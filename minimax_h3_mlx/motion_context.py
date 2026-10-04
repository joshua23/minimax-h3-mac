"""Multi-shot motion context: continue a clip from the previous clip's own latents.

The port of the "clip chaining" idea (credit to the ComfyUI h3_motion_context pack for the two
load-bearing tricks). Shot N+1 pins a run of shot N's final frames — and a window of its final
sound — as conditioning, so the model reads real motion and real audio across the cut instead of
guessing them from a still:

* the pinned picture is sliced **straight out of the previous clip's latent**, bit for bit what the
  model produced — no h264 decode, no resize, no VAE re-encode to shift colour at every link;
* the pinned audio window is **end-aligned with the join**: it must END at the cut and reach
  backwards into the sound that already played, which is what makes the model continue the
  soundtrack rather than write a new one that sounds like it.

Both windows are conditioning rows like fl2va keyframes: the video rows sit at the head of the new
timeline pinned at the keyframe conditioning level, the audio rows ride clean, and the pinned run is
trimmed off the delivered clip — the new clip re-generates those frames as context and throws them
away, so only the continuation is delivered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .packing import (
    AUDIO_CHANNELS,
    AUDIO_LATENTS_PER_SECOND,
    FPS,
    FRAMES_PER_CHUNK,
    LATENTS_PER_CHUNK,
    _ROPE_FRAMES_PER_LATENT,
)

# One latent step covers 1, 4, 4, 4, 4 pixel frames by cycle position. A clip is `17g + 5` pixel
# frames = `5g + 2` latent steps, so the tail of any legal clip starts at cycle position 0 and the
# sliced run keeps the phase of a freshly encoded one — asserted, not assumed, at slice time.
FRAME_RESCALE = 5.0 / 3.0

# Pin windows that are whole numbers of latent steps: 2/7/12/17 steps cover 5/22/39/56 pixel
# frames. 5 is barely fluid, 22 is nearly seamless; longer windows pin more motion but come off
# the front of the delivered clip.
CONTEXT_WINDOWS = (5, 22, 39, 56)


def pixel_frames(latent_t: int) -> int:
    """Pixel frames covered by ``latent_t`` latent steps."""
    return sum(_ROPE_FRAMES_PER_LATENT[k % len(_ROPE_FRAMES_PER_LATENT)] for k in range(latent_t))


def step_offsets(latent_t: int) -> list[int]:
    """Pixel-frame index at which each latent step begins."""
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += _ROPE_FRAMES_PER_LATENT[k % len(_ROPE_FRAMES_PER_LATENT)]
    return out


def steps_for_frames(n: int) -> int | None:
    """Latent steps covering exactly ``n`` pixel frames, or ``None`` when none does."""
    k, covered = 0, 0
    while covered < n:
        covered += _ROPE_FRAMES_PER_LATENT[k % len(_ROPE_FRAMES_PER_LATENT)]
        k += 1
    return k if covered == n else None


@dataclass
class ClipLatents:
    """One generated clip's latents, in the normalized space the denoiser works in.

    ``video`` is ``(1, C, latent_frames, latent_height, latent_width)`` float32; ``audio`` is
    ``(2, latent_channels, audio_latents)`` float32, channel-major — the same layouts the decode
    path consumes after denormalization. Saved ``.npz`` files carry both, so a chain can cross
    process boundaries without a decode/re-encode round trip.
    """

    video: np.ndarray
    audio: np.ndarray

    def __post_init__(self):
        if self.video.ndim != 5 or self.video.shape[0] != 1:
            raise ValueError(f"`video` must be (1, C, T, H, W), got {self.video.shape}.")
        if self.audio.ndim != 3 or self.audio.shape[0] != AUDIO_CHANNELS:
            raise ValueError(f"`audio` must be (2, A, T), got {self.audio.shape}.")

    @property
    def latent_frames(self) -> int:
        return int(self.video.shape[2])

    @property
    def frames(self) -> int:
        return pixel_frames(self.latent_frames)

    @property
    def audio_latents(self) -> int:
        return int(self.audio.shape[-1])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, video=self.video, audio=self.audio)

    @classmethod
    def load(cls, path: str | Path) -> "ClipLatents":
        with np.load(path) as data:
            return cls(video=np.asarray(data["video"], np.float32), audio=np.asarray(data["audio"], np.float32))


def resolve_context_window(requested: int, available_frames: int) -> int:
    """Snap a requested pin window onto the nearest whole-latent-step run that fits."""
    if requested <= 0:
        raise ValueError("The context window must be a positive number of frames.")
    run = max(window for window in CONTEXT_WINDOWS if window <= requested) if requested >= CONTEXT_WINDOWS[0] else 0
    if run == 0:
        raise ValueError(
            f"A {requested}-frame context window is not a whole number of latent steps; use one of "
            f"{list(CONTEXT_WINDOWS)}."
        )
    if run > available_frames:
        raise ValueError(
            f"A {run}-frame context window does not fit a {available_frames}-frame clip."
        )
    return run


def video_tail(
    clip: ClipLatents, pixel_frames_requested: int
) -> tuple[mx.array, list[int], int]:
    """Slice the last whole-latent-step run of video straight out of a generated clip.

    Returns ``(tail, offsets, covered)``: the tail as ``(1, C, steps, H, W)`` — the same layout the
    clip's latents use, sliced bit for bit — the pixel-frame index each step sits at on the new
    timeline, and the pixel frames the run covers. The tail of a ``17g + 5`` frame clip always
    starts at cycle position 0 — asserted, because if that ever stopped holding, the pinned content
    would silently disagree with the positions written for it and the join would land at the wrong
    instant.
    """
    steps = steps_for_frames(pixel_frames_requested)
    if steps is None:
        raise ValueError(
            f"A {pixel_frames_requested}-frame window is not a whole number of latent steps; use one "
            f"of {list(CONTEXT_WINDOWS)}."
        )
    total_steps = clip.latent_frames
    if steps > total_steps:
        raise ValueError(f"Asked for {steps} latent steps, the clip has {total_steps}.")
    start = total_steps - steps
    if start % len(_ROPE_FRAMES_PER_LATENT):
        raise RuntimeError(
            f"The {steps}-step tail of a {total_steps}-step clip starts at cycle position "
            f"{start % len(_ROPE_FRAMES_PER_LATENT)}, not 0; refusing rather than rendering a "
            "shifted join."
        )
    covered = pixel_frames(steps)
    if covered != pixel_frames_requested:
        raise RuntimeError(f"{steps} steps cover {covered} frames, expected {pixel_frames_requested}.")
    tail = mx.array(np.ascontiguousarray(clip.video[:, :, start:]))
    offsets = step_offsets(steps)
    return tail, offsets, covered


def audio_tail(clip: ClipLatents, audio_frames_requested: int) -> tuple[np.ndarray, int, float]:
    """Slice the last ``audio_frames_requested`` pixel frames' worth of audio out of a clip.

    Returns ``(tail, rt, overhang)``: the tail as ``(2, latent_channels, rt)`` (``rt`` counts 40 Hz
    latent steps), and the signed fraction of a step by which the clip's audio grid overshoots its
    last pixel frame. H3 rounds the audio grid to the **nearest** step, not up: for ``5/3 * frames``
    landing on .0, .333 or .667 the overhang is exactly 0, +1/3 or -1/3, and the caller shifts the
    window's end coordinate by it so the pinned content lands where its samples actually sit.
    """
    total_t = clip.audio_latents
    frames = clip.frames
    overhang = total_t - FRAME_RESCALE * frames
    if not -0.5 < overhang < 0.5:
        overhang = 0.0
    rt = int(round(audio_frames_requested / float(FPS) * AUDIO_LATENTS_PER_SECOND))
    if rt > total_t:
        rt = total_t
    if rt < 1:
        raise ValueError("The audio context window is empty.")
    return np.ascontiguousarray(clip.audio[..., total_t - rt :]), rt, float(overhang)


@dataclass
class MotionContext:
    """The conditioning a chained clip derives from its predecessor."""

    video_rows: np.ndarray  # (steps * rows_per_frame, C * prod(patch)) — clean, pre-noise
    steps: int
    offsets: list[int]
    covered: int  # pixel frames the pinned video run occupies on the new timeline
    audio_rows: np.ndarray | None  # (2 * rt, latent_channels) — clean
    audio_steps: int  # rt
    audio_start_coord: int  # 40 Hz coordinate the pinned audio window starts at
    trim_frames: int  # == covered; the delivered clip drops these from the front

    @property
    def num_condition_video_rows(self) -> int:
        return int(self.video_rows.shape[0])

    @property
    def num_condition_audio_rows(self) -> int:
        return 0 if self.audio_rows is None else int(self.audio_rows.shape[0])


def build_motion_context(
    clip: ClipLatents,
    patch_size: tuple[int, int, int],
    *,
    context_frames: int = 22,
    context_audio_frames: int = 24,
) -> MotionContext:
    """Derive shot N+1's conditioning from shot N's latents.

    ``context_frames`` pins that many pixel frames of the previous clip's picture (snapped to a
    whole-latent-step window); ``context_audio_frames`` pins that many frames' worth of tail sound,
    end-aligned with the video window — 0 follows the video window, and multiples of 24 are whole
    seconds. Both are sliced from the latents directly.
    """
    _, ph, pw = patch_size
    requested = resolve_context_window(context_frames, clip.frames)
    tail, offsets, covered = video_tail(clip, requested)

    pt, _, _ = patch_size
    if pt != 1:
        raise ValueError(f"The motion-context layout assumes a (1, {ph}, {pw}) patch, got {patch_size}.")
    steps = int(tail.shape[2])
    gh, gw = tail.shape[3] // ph, tail.shape[4] // pw
    # (1, C, steps, H, W) -> (steps * rows_per_frame, C * ph * pw): step-major, and within a step
    # the row element order matches patchify_video_latents — (C, ph, pw) per (gh, gw) raster patch.
    x = tail.reshape(tail.shape[1], steps, gh, ph, gw, pw)
    x = mx.transpose(x, (1, 2, 4, 0, 3, 5))
    video_rows = np.ascontiguousarray(np.array(x.reshape(steps * gh * gw, -1)))

    a_frames = int(context_audio_frames) or covered
    audio_rows, rt, overhang = audio_tail(clip, a_frames)
    # The window is end-aligned with the pinned video: both end at the join (frame `covered`), and
    # the end coordinate moves by the audio grid's overhang before snapping onto the 40 Hz grid.
    end_frame = covered + overhang / FRAME_RESCALE
    end_coord = round(FRAME_RESCALE * end_frame)
    audio_rows = np.ascontiguousarray(audio_rows.transpose(0, 2, 1).reshape(-1, audio_rows.shape[1]))

    return MotionContext(
        video_rows=video_rows,
        steps=steps,
        offsets=offsets,
        covered=covered,
        audio_rows=audio_rows,
        audio_steps=rt,
        audio_start_coord=end_coord - rt,
        trim_frames=covered,
    )
