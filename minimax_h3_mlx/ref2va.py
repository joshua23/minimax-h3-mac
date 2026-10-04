# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# MLX adaptation of reference/diffusers/modular/packing_ref2va.py.

"""The ``ref2va`` omni-reference task: references, their preparation and their presentation.

The port of ``reference/diffusers/modular/packing_ref2va.py`` and the preparation half of
``before_encoder.py``. A request carries an ordered list of references — at most 9 images, 3 videos
and 3 audio clips, 12 in total — and MiniMax-H3 packs one block per reference ahead of the
generated rows:

    [ text (L) | reference block 1 | ... | target audio (A) | target video (V) ]

The order is semantic twice over: it fixes the ``"<Picture i>"`` / ``"<Audio j>"`` / ``"<Video k>"``
labels of the prompt presentation, and it advances the shared audio/video rotary clock. Unlike an
``fl2va`` keyframe, a reference never binds the target geometry: every reference is prepared at its
own resolution (2048 pixel short edge for images, the 768 pixel canvas of its own aspect ratio for
videos) and carries its own latent geometry.

Media files are decoded with the ``ffmpeg`` binary the release media tooling already requires, not
PyAV: a video is decoded into RGB frames plus its soundtrack as a float32 waveform, an audio file
into a waveform. The soundtrack resample is a numpy FFT resample rather than torchaudio's kaiser
window — a band-limited resample of the same truncation, a few samples different at the edges.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .packing import (
    AUDIO_CHANNELS,
    CANVAS_MULTIPLE,
    FPS,
    FRAMES_PER_CHUNK,
    LATENTS_PER_CHUNK,
    resolve_canvas_size,
)

# Documented per-request limits of the omni-reference task.
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCES = 12

# Reference images are resized to a 2048 pixel short edge — upscaling included — and both axes are
# rounded to a multiple of 32 independently. There is no area cap, so a 4:1 reference is 8192x2048.
REFERENCE_IMAGE_SHORT_EDGE = 2048

# The conditioner sees a reference video at 2 fps, and Qwen3-VL merges every two of those frames
# into one vision block labelled with the mean timestamp of the pair.
QWEN_VIDEO_SAMPLE_FPS = 2.0
QWEN_TEMPORAL_PATCH = 2


def _run_ffmpeg(ffmpeg: str, args: list[str]) -> bytes:
    proc = subprocess.run(
        [ffmpeg, "-v", "error", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(args[:4])}...): {proc.stderr.decode(errors='replace')[:400]}")
    return proc.stdout


def _ffprobe_stream_field(ffprobe: str, path: str, selector: str, field_name: str) -> str:
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", selector, "-show_entries", f"stream={field_name}", "-of", "csv=p=0", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout.decode().strip()


def _ffprobe_geometry(ffprobe: str, path: str) -> tuple[int, int]:
    text = _ffprobe_stream_field(ffprobe, path, "v:0", "width,height")
    try:
        width, height = (int(part) for part in text.split(","))
    except ValueError:
        raise ValueError(f"Could not resolve the frame geometry of {path}.") from None
    return width, height


def _ffprobe_fps(ffprobe: str, path: str) -> float:
    text = _ffprobe_stream_field(ffprobe, path, "v:0", "avg_frame_rate")
    try:
        num, den = text.split("/")
        rate = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return float(FPS)
    return rate if rate > 0 else float(FPS)


def _ffprobe_sample_rate(ffprobe: str, path: str) -> int:
    text = _ffprobe_stream_field(ffprobe, path, "a:0", "sample_rate")
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"No audio stream (or unreadable sample rate) in {path}.") from None


def _decode_video_frames(ffmpeg: str, path: str) -> np.ndarray:
    """Decode a video into ``(num_frames, height, width, 3)`` uint8 RGB at the container's size.

    ffmpeg applies the container's display-matrix rotation on the way out, which is what the
    reference's PyAV decode reproduces by hand.
    """
    width, height = _ffprobe_geometry("ffprobe", path)
    raw = _run_ffmpeg(ffmpeg, ["-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    frame_bytes = width * height * 3
    if not raw or len(raw) % frame_bytes:
        raise ValueError(f"No video frames (or a partial frame) decoded from {path}.")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def _decode_audio_waveform(ffmpeg: str, path: str) -> tuple[np.ndarray, int]:
    """Decode a file's first audio stream into ``(channels, num_samples)`` float32 at its own rate."""
    rate = _ffprobe_sample_rate("ffprobe", path)
    raw = _run_ffmpeg(
        ffmpeg,
        ["-i", path, "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(AUDIO_CHANNELS), "-"],
    )
    samples = np.frombuffer(raw, dtype=np.float32).reshape(-1, AUDIO_CHANNELS)
    return samples.T.copy(), rate


@dataclass
class Reference:
    """One omni-reference: an image, a video (plus, optionally, its own soundtrack), or an audio clip.

    Paths are decoded when the reference is built, so no later stage ever opens a media file. The
    references of a request are passed **in the order the model should read them**.
    """

    image: str | os.PathLike | Image.Image | np.ndarray | None = None
    video: str | os.PathLike | np.ndarray | None = None
    fps: float | None = None
    audio: str | os.PathLike | np.ndarray | None = None
    sample_rate: int | None = None
    ffmpeg: str | None = None

    def __post_init__(self):
        ffmpeg = self.ffmpeg or shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Decoding a reference media file needs the ffmpeg binary on PATH.")
        media = [name for name in ("image", "video", "audio") if getattr(self, name) is not None]
        if media not in (["image"], ["video"], ["audio"], ["video", "audio"]):
            raise ValueError(
                "A Reference must carry exactly one of `image`, `video` or `audio` — plus, for a video, "
                f"the `audio` of its soundtrack — got {media if media else 'none of them'}."
            )
        if isinstance(self.image, (str, os.PathLike)):
            self.image = ImageOps.exif_transpose(Image.open(str(self.image)).convert("RGB"))
        if isinstance(self.video, (str, os.PathLike)):
            path = str(self.video)
            self.video = _decode_video_frames(ffmpeg, path)
            if self.fps is None:
                self.fps = _ffprobe_fps("ffprobe", path)
            # A video reference conditions on its soundtrack too, when the container carries one.
            if self.audio is None and _ffprobe_stream_field("ffprobe", path, "a:0", "codec_name"):
                self.audio, rate = _decode_audio_waveform(ffmpeg, path)
                if self.sample_rate is None:
                    self.sample_rate = rate
        if isinstance(self.audio, (str, os.PathLike)):
            self.audio, rate = _decode_audio_waveform(ffmpeg, str(self.audio))
            if self.sample_rate is None:
                self.sample_rate = rate
        if self.video is not None and self.fps is None:
            self.fps = float(FPS)

    @property
    def kind(self) -> str:
        if self.image is not None:
            return "image"
        return "video" if self.video is not None else "audio"

    @property
    def has_audio(self) -> bool:
        return self.audio is not None


@dataclass
class PreparedReference:
    """One reference prepared for packing and encoding, in packed order."""

    kind: str
    has_audio: bool = False
    image: Image.Image | None = None
    frames: np.ndarray | None = None
    waveform: np.ndarray | None = None
    block_timestamps: list[float] = field(default_factory=list)
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0

    @property
    def num_video_rows(self) -> int:
        return self.num_latent_frames * (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def num_audio_rows(self) -> int:
        return self.num_audio_latents * AUDIO_CHANNELS


def check_references(references: list[Reference]) -> list[str]:
    """Validate the per-modality limits the released task documents; returns the kinds."""
    kinds = [reference.kind for reference in references]
    if not kinds:
        raise ValueError("Ref2VA needs at least one reference.")
    for kind, limit in (("image", MAX_REFERENCE_IMAGES), ("video", MAX_REFERENCE_VIDEOS), ("audio", MAX_REFERENCE_AUDIOS)):
        if kinds.count(kind) > limit:
            raise ValueError(f"MiniMax-H3 accepts at most {limit} {kind} references, got {kinds.count(kind)}.")
    if len(kinds) > MAX_REFERENCES:
        raise ValueError(f"MiniMax-H3 accepts at most {MAX_REFERENCES} references in total, got {len(kinds)}.")
    if set(kinds) == {"audio"}:
        raise ValueError("An audio reference has to be paired with at least one image or video reference.")
    return kinds


def resolve_reference_image_size(width: int, height: int) -> tuple[int, int]:
    """A 2048 pixel short edge, both axes rounded to a multiple of 32; aspect within 1:4 to 4:1."""
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
    scale = REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    multiple = CANVAS_MULTIPLE
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def prepare_reference_image(image: Image.Image, height: int, width: int) -> Image.Image:
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def resample_reference_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    """Resample onto 24 fps the way ffmpeg's CFR filter does: every source frame lands on the
    output slot ``round(index * 24 / fps)`` and holds it until the next frame's slot."""
    if fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
    if fps == FPS:
        return frames
    scale = FPS / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    return np.repeat(frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0)


def prepare_reference_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    """Truncate to the generated frame count and put the video on the canvas of its own aspect."""
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"A reference video must be `(num_frames, height, width, 3)` RGB frames, got {tuple(frames.shape)}."
        )
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
    )


def sample_reference_video_frames(frames: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """Sample the 2 fps frames the conditioner sees, and timestamp every merged vision block."""
    stride = FPS / QWEN_VIDEO_SAMPLE_FPS
    indices, cursor = [], 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    timestamps = [index / QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % QWEN_TEMPORAL_PATCH)
    block_timestamps = [
        (timestamps[index] + timestamps[index + QWEN_TEMPORAL_PATCH - 1]) / 2
        for index in range(0, len(timestamps), QWEN_TEMPORAL_PATCH)
    ]
    return [frames[index] for index in indices], block_timestamps


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen3-VL's smart resize: multiples of ``factor``, aspect kept, pixel budget respected."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_reference_video(
    frames: np.ndarray,
    patch_size: int = 16,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = 4096,
    max_pixels: int = 25_165_824,
) -> tuple[np.ndarray, np.ndarray]:
    """The torch-free video preprocessing the conditioner's vision path expects.

    The released video processor needs torch, so this mirrors its math directly: every frame is
    resized to one shared smart-resize target (BICUBIC, multiples of ``patch * merge``), normalized
    with the 0.5/0.5 convention the release's video preprocessor carries, and patchified with frames
    merged in pairs. Rows come out ``(num_patches, channels * temporal * patch * patch)`` with the
    channels-major element order the tower's ``PatchEmbed`` reshape expects, patches ordered
    ``(temporal block, merged grid row, merged grid column, within-merge row, within-merge column)``.
    """
    if frames.shape[0] % temporal_patch_size:
        padding = (-frames.shape[0]) % temporal_patch_size
        frames = np.concatenate([frames, np.repeat(frames[-1:], padding, axis=0)], axis=0)
    factor = patch_size * merge_size
    height, width = frames.shape[1:3]
    if min(height, width) < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    # Qwen3-VL budgets pixels over the whole sampled video, not per frame.
    h_bar, w_bar = _smart_resize(
        height, width, factor, min_pixels / frames.shape[0], max_pixels / frames.shape[0]
    )
    resized = np.stack(
        [np.asarray(Image.fromarray(frame).resize((w_bar, h_bar), Image.Resampling.BICUBIC)) for frame in frames]
    )
    pixels = resized.astype(np.float32) / 255.0
    pixels = (pixels - 0.5) / 0.5
    t, h, w, c = pixels.shape
    gh, gw = h // patch_size, w // patch_size
    x = pixels.transpose(0, 3, 1, 2).reshape(
        t // temporal_patch_size, temporal_patch_size, c,
        gh // merge_size, merge_size, patch_size,
        gw // merge_size, merge_size, patch_size,
    )
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    patches = np.ascontiguousarray(x.reshape(-1, c * temporal_patch_size * patch_size * patch_size))
    grid = np.array([[t // temporal_patch_size, gh, gw]], dtype=np.int64)
    return patches, grid


def fft_resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Band-limited FFT resample of ``(channels, num_samples)`` float32 audio."""
    if source_rate == target_rate:
        return waveform
    num = int(round(waveform.shape[-1] * target_rate / source_rate))
    spectrum = np.fft.rfft(waveform, axis=-1)
    source_num = waveform.shape[-1]
    resampled = np.zeros((waveform.shape[0], num // 2 + 1), dtype=np.complex128)
    common_num = min(source_num, num)
    keep = common_num // 2 + 1
    resampled[:, :keep] = spectrum[:, :keep]
    if common_num % 2 == 0:
        if num < source_num:
            resampled[:, common_num // 2] *= 2
        elif source_num < num:
            resampled[:, common_num // 2] *= 0.5
    return (np.fft.irfft(resampled, n=num, axis=-1) * (num / source_num)).astype(np.float32)


def prepare_reference_waveform(
    waveform: np.ndarray, sample_rate: int, target_sample_rate: int, max_duration: float
) -> np.ndarray:
    """Truncate at the source rate, upmix mono to stereo, then resample once to the VAE rate."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, AUDIO_CHANNELS):
        raise ValueError(
            "A reference soundtrack must be a `(channels, num_samples)` mono or stereo waveform, got "
            f"{tuple(waveform.shape)}."
        )
    waveform = waveform[:, : int(max_duration * sample_rate)]
    if waveform.shape[0] != AUDIO_CHANNELS:
        waveform = np.repeat(waveform, AUDIO_CHANNELS, axis=0)
    return fft_resample(waveform, sample_rate, target_sample_rate)


def prepare_references(
    references: list[Reference],
    num_frames: int | None,
    audio_sample_rate: int,
) -> tuple[list[PreparedReference], int]:
    """Resolve every reference at its own resolution and, if it was left open, the duration.

    A duration left open is taken from the single audio-bearing reference, as in the reference
    implementation; every other request states ``num_frames`` (the pipeline derives it from the
    requested duration).
    """
    from .packing import align_num_frames

    resolved = [PreparedReference(kind=reference.kind, has_audio=reference.has_audio) for reference in references]
    if num_frames is None:
        audio_bearing = [index for index, reference in enumerate(resolved) if reference.has_audio]
        if len(audio_bearing) != 1:
            raise ValueError(
                "The duration may only be left to the references when exactly one of them carries audio, got "
                f"{len(audio_bearing)}."
            )
        index = audio_bearing[0]
        rate = references[index].sample_rate or audio_sample_rate
        duration = references[index].audio.shape[-1] / rate
        if not 5.0 <= duration <= 15.0:
            raise ValueError(
                f"`references[{index}]` is {duration:g} seconds long, outside the 5 to 15 seconds "
                "MiniMax-H3 generates."
            )
        num_frames = align_num_frames(round(duration * FPS))
    num_frames = align_num_frames(num_frames)

    for reference, entry in zip(resolved, references):
        if reference.kind == "image":
            image = entry.image if isinstance(entry.image, Image.Image) else Image.fromarray(entry.image)
            image = ImageOps.exif_transpose(image).convert("RGB")
            height, width = resolve_reference_image_size(*image.size)
            reference.image = prepare_reference_image(image, height, width)
        elif reference.kind == "video":
            frames = resample_reference_frames(np.asarray(entry.video), float(entry.fps))
            reference.frames = prepare_reference_frames(frames, num_frames)
        if reference.has_audio:
            reference.waveform = prepare_reference_waveform(
                entry.audio,
                entry.sample_rate or audio_sample_rate,
                audio_sample_rate,
                max_duration=num_frames / FPS,
            )
    return resolved, num_frames


def trim_reference_num_frames(num_frames: int) -> int:
    """Snap a reference video's frame count *down* to a ``17 * n + 5`` the video VAE encodes."""
    if num_frames < 1:
        raise ValueError(f"A reference video must have at least one frame, got {num_frames}.")
    return (
        max(1, (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK) * FRAMES_PER_CHUNK + LATENTS_PER_CHUNK
    )


def build_ref2va_presentation(
    tokenizer,
    prompt: str,
    references: list[PreparedReference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
    tags_text: int,
    tags_video: int,
    vision_start_id: int,
    vision_end_id: int,
    image_pad_id: int,
    video_pad_id: int,
) -> tuple[list[int], list[int]]:
    """Tokenize MiniMax-H3's presentation of a ``ref2va`` request.

    Every reference prepends a label, in packed order and numbered per modality: ``"<Picture i>: "``
    plus a vision block for an image, ``"<Audio j>: "`` alone for audio — a waveform never reaches
    the conditioner — and ``"<Video k>: "`` plus one timestamped vision block per merged frame pair
    for a video. A video that carries sound is labelled ``"<Audio j>: "`` *before* ``"<Video k>: "``,
    mirroring the order its rows are packed in. The prompt follows verbatim.
    """

    def text(value: str) -> tuple[list[int], list[int]]:
        token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        return token_ids, [tags_text] * len(token_ids)

    def vision(pad_token_id: int, num_tokens: int) -> tuple[list[int], list[int]]:
        token_ids = [vision_start_id] + [pad_token_id] * num_tokens + [vision_end_id]
        return token_ids, [tags_video] * len(token_ids)

    token_ids, token_tags = [], []

    def emit(segment: tuple[list[int], list[int]]) -> None:
        token_ids.extend(segment[0])
        token_tags.extend(segment[1])

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit(text(f"<Audio {counts['audio']}>: "))
        if reference.kind == "image":
            counts["image"] += 1
            emit(text(f"<Picture {counts['image']}>: "))
            emit(vision(image_pad_id, image_token_counts[counts["image"] - 1]))
        elif reference.kind == "video":
            counts["video"] += 1
            emit(text(f"<Video {counts['video']}>: "))
            for timestamp in reference.block_timestamps:
                # `"{:.1f}"` rounds half to even, so the mean of a 2 fps pair renders as "<0.2 seconds>".
                emit(text(f"<{timestamp:.1f} seconds>"))
                emit(vision(video_pad_id, video_block_token_counts[counts["video"] - 1]))
    emit(text(prompt))
    return token_ids, token_tags
