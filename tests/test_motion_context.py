"""Structural checks of the motion-context chaining port.

Pins the port against the semantics the chaining idea rests on: tail slicing at whole latent steps
with the cycle-position guard, the audio grid's ±1/3-step overhang and its end-aligned window, the
pinned rows sharing the target timeline's own rope coordinates, and the delivered trim.

    ./.venv/bin/python tests/test_motion_context.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO
from minimax_h3_mlx.motion_context import (
    ClipLatents,
    audio_tail,
    build_motion_context,
    pixel_frames,
    step_offsets,
    steps_for_frames,
    video_tail,
)
from minimax_h3_mlx.packing import (
    AUDIO_CHANNELS,
    _temporal_position_grid,
    build_motion_context_packed_sequence,
)

FAILURES: list[str] = []
PATCH = (1, 2, 2)
LATENT_C = 6  # small stand-in channel count; row dim = 6 * 1 * 2 * 2 = 24
AUDIO_A = 4


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def make_clip(latent_frames: int, latent_h: int = 8, latent_w: int = 8, seed: int = 3) -> ClipLatents:
    rng = np.random.default_rng(seed)
    video = rng.standard_normal((1, LATENT_C, latent_frames, latent_h, latent_w)).astype(np.float32)
    # H3 rounds the audio grid to the nearest step: a 37-step (124-frame) clip carries 207 audio
    # latents, overshooting 5/3 * 124 by exactly 1/3 of a step.
    audio_latents = round(pixel_frames(latent_frames) / 24 * 40)
    audio = rng.standard_normal((AUDIO_CHANNELS, AUDIO_A, audio_latents)).astype(np.float32)
    return ClipLatents(video=video, audio=audio)


def main() -> int:
    # -- step arithmetic ------------------------------------------------------------------
    # The cycle is (1, 4, 4, 4, 4) repeating: 22 frames = 1+4+4+4+4+1+4, so the 7th step starts
    # at frame 18, not 21.
    check("step offsets follow the 1,4,4,4,4 cycle", step_offsets(7) == [0, 1, 5, 9, 13, 17, 18])
    check("windows cover their frames", all(pixel_frames(steps_for_frames(n)) == n for n in (5, 22, 39, 56)))
    check("1 frame is covered by one step arithmetically", steps_for_frames(1) == 1)

    # -- video tail slicing ----------------------------------------------------------------
    clip = make_clip(37)  # a 124-frame clip: 37 latent steps
    marker = clip.video[:, :, -2:] + 7.0  # a distinctive tail
    clip.video = np.ascontiguousarray(np.concatenate([clip.video[:, :, :-2], marker], axis=2))
    tail, offsets, covered = video_tail(clip, 22)
    check("22 frames is 7 steps", (tail.shape[2], covered) == (7, 22))
    check("tail is bit-for-bit the latent's tail", bool(mx.array_equal(mx.array(tail), mx.array(clip.video[:, :, -7:]))))
    # The last step starts 4 frames before the tail's end and covers those 4.
    check("offsets end where the clip ends", offsets[-1] == 22 - 4 and offsets[0] == 0)
    try:
        video_tail(clip, 5 + 1)
        check("off-grid window rejected", False)
    except ValueError:
        check("off-grid window rejected", True)
    # A 1-frame window covers exactly one step arithmetically, but a clip tail can never start a
    # 1-step run on cycle position 0 (5g+2-1 = 5g+1), so the slicing guard refuses it.
    try:
        video_tail(clip, 1)
        check("1-frame window rejected by the cycle guard", False)
    except RuntimeError:
        check("1-frame window rejected by the cycle guard", True)

    # -- audio tail and overhang ------------------------------------------------------------
    # 124 frames want 206.67 steps at 40 Hz; H3 rounds to the nearest, so 207 steps overhang +1/3.
    # The stub clip carries exactly that grid for a 124-frame clip.
    tail_audio, rt, overhang = audio_tail(clip, 24)
    check("24 frames -> 40 audio steps", rt == 40)
    check("overhang is exactly +1/3", abs(overhang - 1.0 / 3.0) < 1e-9, f"{overhang}")
    check("tail is the last rt steps", tail_audio.shape == (AUDIO_CHANNELS, AUDIO_A, 40))

    # -- the derived context ----------------------------------------------------------------
    context = build_motion_context(clip, PATCH, context_frames=22, context_audio_frames=24)
    rows_per_frame = (8 // 2) * (8 // 2)
    check("pinned video rows", context.num_condition_video_rows == 7 * rows_per_frame)
    check("pinned audio rows", context.num_condition_audio_rows == 2 * 40)
    check("trim equals the pinned head", context.trim_frames == 22)
    check("audio window ends at the join on the 40 Hz grid",
          context.audio_start_coord + 40 == round(5.0 / 3.0 * (22 + overhang / (5.0 / 3.0))) or
          context.audio_start_coord + 40 == round(5.0 / 3.0 * 22))

    # -- the chained layout -----------------------------------------------------------------
    text_tags = [TAG_TEXT] * 9
    layout = build_motion_context_packed_sequence(
        text_tags,
        context.steps,
        context.offsets,
        context.num_condition_video_rows,
        context.num_condition_audio_rows,
        context.audio_steps,
        context.audio_start_coord,
        37,  # the new clip: also 124 frames
        8, 8,
        207,
        PATCH,
    )
    tags = np.asarray(layout.token_tags.tolist())
    video_idx = np.asarray(layout.video_indices.tolist())
    audio_idx = np.asarray(layout.audio_indices.tolist())
    pos = np.asarray(layout.position_ids.tolist(), dtype=np.float64)
    n_text, n_pin_v, n_pin_a = 9, 7 * rows_per_frame, 2 * 40
    target_video_start = n_text + n_pin_v + n_pin_a + 2 * 207

    check("sequence length",
          layout.sequence_length == n_text + n_pin_v + n_pin_a + 2 * 207 + 37 * rows_per_frame)
    check("pinned rows are the condition counts",
          layout.num_condition_video_rows == n_pin_v and layout.num_condition_audio_rows == n_pin_a)
    check("pinned video tagged video, seam audio tagged audio",
          set(tags[n_text : n_text + n_pin_v].tolist()) == {TAG_VIDEO}
          and set(tags[n_text + n_pin_v : target_video_start - 2 * 207].tolist()) == {TAG_AUDIO})
    check("pinned rows precede target rows in each modality",
          np.array_equal(video_idx[:n_pin_v], np.arange(n_text, n_text + n_pin_v))
          and np.array_equal(audio_idx[:n_pin_a], np.arange(n_text + n_pin_v, n_text + n_pin_v + n_pin_a)))

    # The pinned head shares the target timeline's own rope coordinates: pinned step k sits at
    # origin + 5/3 * pixel offset — the same coordinate the target's own step there occupies.
    eps = 1e-4
    expected = [float(n_text) + (5.0 / 3.0) * off for off in [0, 1, 5, 9, 13, 17, 18]]
    pinned_ok = all(
        abs(pos[n_text + k * rows_per_frame, 0] - expected[k]) < eps for k in range(7)
    )
    check("pinned rows sit at the target's timeline coordinates", pinned_ok,
          f"t[0]={pos[n_text, 0]:.3f}, t[6]={pos[n_text + 6 * rows_per_frame, 0]:.3f}")
    check("target video still covers the full timeline from its origin",
          abs(pos[target_video_start, 0] - float(n_text)) < eps)
    check("seam window ends at the join",
          abs(pos[n_text + n_pin_v + n_pin_a - 1, 0] - (context.audio_start_coord + context.audio_steps - 1)) < eps)

    # -- npz round trip ---------------------------------------------------------------------
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.npz"
        clip.save(path)
        loaded = ClipLatents.load(path)
        check("npz round trip", np.array_equal(loaded.video, clip.video) and np.array_equal(loaded.audio, clip.audio))

    print()
    if FAILURES:
        print(f"SOME CHECKS FAILED: {FAILURES}")
        return 1
    print("all motion context checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
