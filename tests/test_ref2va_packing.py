"""Structural checks of the ``ref2va`` packed layout and request presentation.

The torch reference cannot run here, so this pins the port against the reference's own documented
arithmetic: the row order, the per-modality rotary clock advances (an image takes one integer slot,
a video ``max(audio latents, sequential span)``), the soundtrack-before-video packing of a video
reference, and the presentation's per-modality label numbering.

    ./.venv/bin/python tests/test_ref2va_packing.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO
from minimax_h3_mlx.packing import (
    AUDIO_CHANNELS,
    _temporal_position_span_sequential,
    build_ref2va_packed_sequence,
)
from minimax_h3_mlx.ref2va import (
    PreparedReference,
    build_ref2va_presentation,
    resolve_reference_image_size,
    resample_reference_frames,
    sample_reference_video_frames,
)

FAILURES: list[str] = []

PATCH = (1, 2, 2)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


@dataclass
class StubReference:
    kind: str
    num_latent_frames: int = 1
    latent_height: int = 8
    latent_width: int = 8
    num_audio_latents: int = 0
    has_audio: bool = False
    block_timestamps: list = field(default_factory=list)

    @property
    def num_video_rows(self) -> int:
        return self.num_latent_frames * (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def num_audio_rows(self) -> int:
        return self.num_audio_latents * AUDIO_CHANNELS


class StubTokenizer:
    """Lossless one-codepoint-per-character stand-in with a few special ids."""

    SPECIALS = {"<|vision_start|>": 151652, "<|vision_end|>": 151653, "<|image_pad|>": 151655, "<|video_pad|>": 151656}

    def __call__(self, value: str, add_special_tokens: bool = False):
        return {"input_ids": [ord(ch) for ch in value]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.SPECIALS[token]


def main() -> int:
    num_latent_frames = 7  # target: 7 latent frames, 8x8 latents -> 16 rows per frame
    latent_height = latent_width = 8
    num_audio_latents = 20

    image_ref = StubReference("image")  # 1 latent frame -> 16 video rows
    video_ref = StubReference("video", num_latent_frames=5, num_audio_latents=10, has_audio=True)
    audio_ref = StubReference("audio", num_audio_latents=12, has_audio=True)
    references = [image_ref, video_ref, audio_ref]

    # The conditioner presentation: image label+block (6 tokens), video label + 2 blocks, audio
    # label, prompt. Everything outside a vision block is text-tagged.
    presentation = [TAG_TEXT] * 5 + [TAG_VIDEO] * 8 + [TAG_TEXT] * 4 + [TAG_TEXT] * 30
    layout = build_ref2va_packed_sequence(
        presentation, references, num_latent_frames, latent_height, latent_width, num_audio_latents, PATCH
    )

    tags = np.asarray(layout.token_tags.tolist())
    video_idx = np.asarray(layout.video_indices.tolist())
    audio_idx = np.asarray(layout.audio_indices.tolist())

    num_text = len(presentation)
    num_img_rows = image_ref.num_video_rows
    num_vid_rows = video_ref.num_video_rows
    num_vid_audio_rows = video_ref.num_audio_rows
    num_aud_rows = audio_ref.num_audio_rows
    num_target_video = num_latent_frames * 16
    num_target_audio = num_audio_latents * AUDIO_CHANNELS

    check(
        "sequence length",
        layout.sequence_length
        == num_text + num_img_rows + num_vid_audio_rows + num_vid_rows + num_aud_rows + num_target_audio + num_target_video,
        f"{layout.sequence_length}",
    )
    check("condition video rows", layout.num_condition_video_rows == num_img_rows + num_vid_rows)
    check("condition audio rows", layout.num_condition_audio_rows == num_vid_audio_rows + num_aud_rows)

    # Row order: [text | image | video soundtrack | video frames | audio | target audio | target video]
    image_rows = np.arange(num_text, num_text + num_img_rows)
    vid_audio_rows = np.arange(image_rows[-1] + 1, image_rows[-1] + 1 + num_vid_audio_rows)
    vid_rows = np.arange(vid_audio_rows[-1] + 1, vid_audio_rows[-1] + 1 + num_vid_rows)
    aud_rows = np.arange(vid_rows[-1] + 1, vid_rows[-1] + 1 + num_aud_rows)
    check("soundtrack precedes its video rows", np.array_equal(audio_idx[: num_vid_audio_rows + num_aud_rows], np.concatenate([vid_audio_rows, aud_rows])))
    target_audio_start = vid_rows[-1] + 1 + num_aud_rows
    check(
        "video rows are refs then target",
        np.array_equal(
            video_idx,
            np.concatenate(
                [image_rows, vid_rows, np.arange(target_audio_start + num_target_audio, layout.sequence_length)]
            ),
        ),
    )
    check("image block tagged video", set(tags[image_rows].tolist()) == {TAG_VIDEO})

    # Rotary clock: text rows sit on the time axis at their index; an image takes one slot. The
    # packed sequence stores float32, so the tolerances cover that cast, not semantics.
    pos = np.asarray(layout.position_ids.tolist(), dtype=np.float64)
    eps = 1e-4
    cursor = float(num_text)
    check("image ref takes one integer slot", abs(pos[image_rows, 0][0] - cursor) < eps, f"t={pos[image_rows, 0][0]}")
    cursor += 1.0
    check(
        "video ref shares its origin with its soundtrack",
        abs(pos[vid_audio_rows, 0][0] - cursor) < eps and abs(pos[vid_rows, 0][0] - cursor) < eps,
    )
    cursor += max(float(video_ref.num_audio_latents), _temporal_position_span_sequential(video_ref.num_latent_frames))
    check("audio ref advances by its latents", abs(pos[aud_rows, 0][0] - cursor) < eps)
    cursor += float(audio_ref.num_audio_latents)
    check("target audio starts where refs end", abs(pos[target_audio_start, 0] - cursor) < eps)

    # Audio rows are channel-major: both stereo channels repeat the same latent clock.
    check(
        "audio channel-major",
        abs(pos[aud_rows, 0][1] - pos[aud_rows, 0][0] - 1.0) < eps
        and abs(pos[aud_rows, 0][num_aud_rows // AUDIO_CHANNELS] - pos[aud_rows, 0][0]) < eps,
    )

    # -- presentation ------------------------------------------------------------------------
    tokenizer = StubTokenizer()
    prepared = [
        PreparedReference(kind="image", image="img"),
        PreparedReference(kind="video", frames="vid", has_audio=True, block_timestamps=[0.2, 0.6]),
        PreparedReference(kind="audio", waveform="aud"),
    ]
    token_ids, token_tags = build_ref2va_presentation(
        tokenizer,
        "the prompt",
        prepared,
        image_token_counts=[2],
        video_block_token_counts=[3],
        tags_text=TAG_TEXT,
        tags_video=TAG_VIDEO,
        vision_start_id=tokenizer.SPECIALS["<|vision_start|>"],
        vision_end_id=tokenizer.SPECIALS["<|vision_end|>"],
        image_pad_id=tokenizer.SPECIALS["<|image_pad|>"],
        video_pad_id=tokenizer.SPECIALS["<|video_pad|>"],
    )
    text = "".join(chr(int(i)) for i in token_ids if int(i) < 151000)
    check(
        "presentation labels",
        "<Picture 1>: " in text and "<Audio 1>: " in text and "<Video 1>: " in text,
        text[:48] + "...",
    )
    check(
        "video soundtrack labelled before the video",
        text.index("<Audio 1>: ") > text.index("<Picture 1>: ") and text.index("<Audio 1>: ") < text.index("<Video 1>: "),
    )
    check("prompt comes last", text.endswith("the prompt"))
    check("timestamp labels", "<0.2 seconds>" in text and "<0.6 seconds>" in text)
    check(
        "vision blocks tagged video, labels text",
        all(tag == TAG_VIDEO for tok, tag in zip(token_ids, token_tags) if int(tok) >= 151000)
        and all(tag == TAG_TEXT for tok, tag in zip(token_ids, token_tags) if int(tok) < 151000),
    )

    # -- preparation helpers -----------------------------------------------------------------
    height, width = resolve_reference_image_size(1024, 2048)
    check("reference image size", (height, width) == (4096, 2048), f"{height}x{width}")
    frames = np.zeros((48, 8, 8, 3), dtype=np.uint8)
    resampled = resample_reference_frames(frames, 12.0)
    check("12 fps -> 24 fps doubles frames", resampled.shape[0] == 96, str(resampled.shape[0]))
    check("24 fps flows through", resample_reference_frames(frames, 24.0) is frames)
    sampled, timestamps = sample_reference_video_frames(np.zeros((48, 4, 4, 3), dtype=np.uint8))
    # The raw block timestamp is 0.25; `{:.1f}` renders it "<0.2 seconds>" by rounding half to even.
    check(
        "2 fps sampling",
        len(sampled) == 4 and len(timestamps) == 2 and f"{timestamps[0]:.1f}" == "0.2",
        f"{timestamps}",
    )

    print()
    if FAILURES:
        print(f"SOME CHECKS FAILED: {FAILURES}")
        return 1
    print("all ref2va packing checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
