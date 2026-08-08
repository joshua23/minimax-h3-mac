"""Measure what quantization does to the MiniMax-H3 DiT.

Comparing two independent generations is the obvious approach and it is the wrong one: the
trajectories diverge chaotically from the first step, so the difference you measure is dominated by
divergence amplification rather than by quantization error, and the ranking you get from a handful
of clips is noise. Generations are also ~9 minutes *per step* at the released canvas, so a sample
large enough to rank 2/3/4/6/8-bit would take days.

This script uses **teacher forcing** instead. One bfloat16 trajectory is run and its latents are
recorded at every step; each quantized variant is then asked to predict the velocity *at those same
latents*. Both models see identical inputs at every point, so the difference measured is the
quantization error alone, and every step of every prompt/seed is an independent paired sample.

The reported statistic is the per-step **relative L2** of the velocity error and its cosine
similarity to the reference, aggregated with a **paired bootstrap over shared (prompt, seed, step)
observations** — paired because the variants are compared on exactly the same inputs, which removes
the between-input variance that otherwise swamps the between-variant differences.

Velocity error is also the right quantity to watch rather than final pixels: the scheduler
integrates it, so a bias that is small per step still accumulates coherently over the trajectory.

    ./.venv/bin/python scripts/eval_quant.py --variants 8 6 4 3 --height 256 --width 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.adaln import ModulationCache
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.packing import (
    FPS,
    KEYFRAME_NOISE_AUG,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    video_latent_num_frames,
)
from minimax_h3_mlx.quantize import QuantConfig, quantize_dit
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler

PROMPTS = [
    "A red fox leaps over a mossy log in a misty forest at dawn.",
    "Waves crash against black volcanic rocks under a grey sky.",
    "A street musician plays saxophone on a rainy neon-lit corner.",
    "Steam rises from a bowl of noodles on a wooden table.",
]


def build_case(text_len, height, width, duration, steps, latent_channels, audio_channels, patch_size, seed):
    """One packed layout plus its initial noise and schedules."""
    frames = align_num_frames(int(round(duration * FPS)))
    latent_frames = video_latent_num_frames(frames)
    lh, lw = height // 16, width // 16
    n_audio = audio_latent_num_frames(frames)

    tags = np.ones(text_len, dtype=np.int64)  # all text rows
    layout = build_packed_sequence(tags, latent_frames, lh, lw, n_audio, patch_size)

    mx.random.seed(seed)
    latents = mx.random.normal((1, latent_channels, latent_frames, lh, lw)).astype(mx.float32)
    video_rows = patchify_video_latents(latents, patch_size)
    audio_rows = mx.random.normal((n_audio * 2, audio_channels)).astype(mx.float32)

    video_sched = MiniMaxH3Scheduler(shift=12.0)
    audio_sched = MiniMaxH3Scheduler(shift=3.0)
    video_sched.set_timesteps(steps)
    audio_sched.set_timesteps(steps)
    return layout, video_rows, audio_rows, video_sched, audio_sched


def timestep_plan(layout, video_sched, audio_sched):
    per_step = []
    for t, at in zip(video_sched.timesteps.tolist(), audio_sched.timesteps.tolist()):
        distinct, inverse = build_row_timesteps(
            layout, float(t), float(at), max(float(t), KEYFRAME_NOISE_AUG), 1.0
        )
        per_step.append((np.array(distinct), np.array(inverse)))
    table = sorted({float(v) for d, _ in per_step for v in d})
    lookup = {v: i for i, v in enumerate(table)}
    plan = [mx.array(np.array([lookup[float(v)] for v in d], dtype=np.int32)[inv].astype(np.int32))
            for d, inv in per_step]
    return mx.array(np.array(table, dtype=np.float32)), plan


def forward(model, cache, layout, video_rows, audio_rows, embeds, table, indices):
    return model(
        video_rows[None].astype(mx.bfloat16),
        audio_rows[None].astype(mx.bfloat16),
        embeds,
        table,
        indices,
        layout.token_tags,
        layout.position_ids,
        layout.video_indices,
        layout.audio_indices,
        layout.text_indices,
        modulation_cache=cache,
    )


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a.ravel() @ b.ravel() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def paired_bootstrap(values: dict[str, list[float]], reference: str, iterations: int = 2000, seed: int = 0):
    """Bootstrap over shared observation indices, so variants resample together."""
    rng = np.random.default_rng(seed)
    names = list(values)
    n = len(values[names[0]])
    arrays = {k: np.asarray(v) for k, v in values.items()}
    out = {}
    for name in names:
        draws = np.empty(iterations)
        for i in range(iterations):
            idx = rng.integers(0, n, n)  # one index set shared by every variant this draw
            draws[i] = arrays[name][idx].mean()
        out[name] = (float(draws.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/Volumes/models/MiniMax-H3/FL2VA")
    parser.add_argument("--variants", type=int, nargs="+", default=[8, 6, 4, 3])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--prompts", type=int, default=2, help="how many built-in prompts to use")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    source = Path(args.checkpoint)
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

    print("loading text encoder")
    encoder = MiniMaxH3TextEncoder(source / "text_encoder", load_vision=False)
    embeds_by_prompt = {}
    for prompt in PROMPTS[: args.prompts]:
        h, tags = encoder.encode(prompt)
        embeds_by_prompt[prompt] = (h.astype(mx.bfloat16), len(tags))
    del encoder

    print("loading bfloat16 reference DiT")
    reference = load_dit(source / "transformer")
    cfg = reference.config

    # 1. Reference trajectory: record the latents visited at every step, plus the velocities.
    cases = []
    for prompt, (embeds, text_len) in embeds_by_prompt.items():
        for seed in args.seeds:
            layout, video_rows, audio_rows, vs, aud = build_case(
                text_len, args.height, args.width, args.duration, args.steps,
                24, 32, cfg.patch_size, seed,
            )
            table, plan = timestep_plan(layout, vs, aud)
            cache = ModulationCache.build(reference, table, dtype=mx.bfloat16)
            trajectory = []
            for i, t in enumerate(vs.timesteps.tolist()):
                vp, ap = forward(reference, cache, layout, video_rows, audio_rows, embeds, table, plan[i])
                mx.eval(vp, ap)
                trajectory.append((video_rows, audio_rows, np.array(vp.astype(mx.float32)),
                                   np.array(ap.astype(mx.float32))))
                video_rows = vs.step(vp[0].astype(mx.float32), float(t), video_rows)
                audio_rows = aud.step(ap[0].astype(mx.float32), float(aud.timesteps[i].item()), audio_rows)
                mx.eval(video_rows, audio_rows)
            cases.append({"prompt": prompt, "seed": seed, "layout": layout, "embeds": embeds,
                          "table": table, "plan": plan, "trajectory": trajectory})
            print(f"  reference trajectory: {prompt[:32]}... seed {seed}, "
                  f"{len(trajectory)} steps, {layout.sequence_length:,} rows", flush=True)
    base_weights = dict(tree_flatten(reference.parameters()))
    del reference

    # 2. Each variant re-predicts at the *same* latents. The bfloat16 weights are kept in memory and
    #    the model rebuilt from them per variant: quantization is destructive, but re-reading 62 GB
    #    from disk four times costs far more than holding it once.
    results = {"video_rel_l2": {}, "audio_rel_l2": {}, "video_cos": {}}
    for bits in args.variants:
        started = time.perf_counter()
        model = MiniMaxH3DiT(cfg)
        model.update(tree_unflatten(list(base_weights.items())))
        mx.eval(model.parameters())
        summary = quantize_dit(model, QuantConfig(bits=bits, group_size=args.group_size))
        v_l2, a_l2, v_cos = [], [], []
        for case in cases:
            cache = ModulationCache.build(model, case["table"], dtype=mx.bfloat16)
            for i, (video_rows, audio_rows, ref_v, ref_a) in enumerate(case["trajectory"]):
                vp, ap = forward(model, cache, case["layout"], video_rows, audio_rows,
                                 case["embeds"], case["table"], case["plan"][i])
                mx.eval(vp, ap)
                got_v = np.array(vp.astype(mx.float32))
                got_a = np.array(ap.astype(mx.float32))
                v_l2.append(relative_l2(got_v, ref_v))
                a_l2.append(relative_l2(got_a, ref_a))
                v_cos.append(cosine(got_v, ref_v))
        key = f"{bits}bit"
        results["video_rel_l2"][key] = v_l2
        results["audio_rel_l2"][key] = a_l2
        results["video_cos"][key] = v_cos
        print(f"  {key}: {summary['gb_after']:.1f} GB, mean video rel-L2 {np.mean(v_l2):.4f}, "
              f"cos {np.mean(v_cos):.5f}  ({time.perf_counter() - started:.0f}s)")
        del model

    # 3. Paired bootstrap over the shared (prompt, seed, step) observations.
    print(f"\n{len(cases) * args.steps} paired observations per variant "
          f"({len(embeds_by_prompt)} prompts x {len(args.seeds)} seeds x {args.steps - 1} steps)")
    print(f"\n{'variant':>8}  {'video rel-L2 [95% CI]':>30}  {'audio rel-L2':>14}  {'video cos':>10}")
    boot_v = paired_bootstrap(results["video_rel_l2"], reference="")
    boot_a = paired_bootstrap(results["audio_rel_l2"], reference="")
    for key in results["video_rel_l2"]:
        m, lo, hi = boot_v[key]
        am, _, _ = boot_a[key]
        print(f"{key:>8}  {m:>12.5f} [{lo:.5f}, {hi:.5f}]  {am:>14.5f}  "
              f"{np.mean(results['video_cos'][key]):>10.6f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"config": vars(args), "raw": results,
             "bootstrap": {"video_rel_l2": boot_v, "audio_rel_l2": boot_a}}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
