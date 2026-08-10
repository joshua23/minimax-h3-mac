#!/usr/bin/env python3
"""Interleaved real-path timing for the opt-in refined text-conditioning cache."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from minimax_h3_mlx.adaln import ModulationCache  # noqa: E402
from minimax_h3_mlx.config import PipelineConfig  # noqa: E402
from minimax_h3_mlx.load import read_audio_vae_config, read_video_vae_config  # noqa: E402
from minimax_h3_mlx.packing import (  # noqa: E402
    AUDIO_CHANNELS,
    FPS,
    KEYFRAME_NOISE_AUG,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    video_latent_num_frames,
)
from minimax_h3_mlx.pipeline import detach_bfloat16  # noqa: E402
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler  # noqa: E402
from minimax_h3_mlx.streaming import BLOCK_LOAD_MODES, load_streaming_dit  # noqa: E402
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder  # noqa: E402


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _mlx_call(name: str) -> int | None:
    func = getattr(mx, name, None)
    if func is None and hasattr(mx, "metal"):
        func = getattr(mx.metal, name, None)
    if func is None:
        return None
    try:
        return int(func())
    except Exception:
        return None


def _reset_mlx_peak() -> None:
    for owner in (mx, getattr(mx, "metal", None)):
        func = getattr(owner, "reset_peak_memory", None) if owner is not None else None
        if func is None:
            continue
        try:
            func()
            return
        except Exception:
            continue


def _vm_stat_value(label: str) -> int | None:
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return None
    prefix = f"{label}:"
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            raw = stripped.split(":", 1)[1].strip().rstrip(".").replace(".", "")
            try:
                return int(raw)
            except ValueError:
                return None
    return None


def _swapusage() -> str | None:
    try:
        return subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    except Exception:
        return None


def _rss_kib() -> int | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(out.strip())
    except Exception:
        return None


def _metrics() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "wall_time_seconds": time.perf_counter(),
        "rss_kib": _rss_kib(),
        "ru_maxrss_raw": usage.ru_maxrss,
        "ru_minflt": usage.ru_minflt,
        "ru_majflt": usage.ru_majflt,
        "ru_inblock": usage.ru_inblock,
        "ru_oublock": usage.ru_oublock,
        "ru_nvcsw": usage.ru_nvcsw,
        "ru_nivcsw": usage.ru_nivcsw,
        "vm_pageouts": _vm_stat_value("Pageouts"),
        "vm_swapouts": _vm_stat_value("Swapouts"),
        "mlx_peak_bytes": _mlx_call("get_peak_memory"),
        "mlx_active_bytes": _mlx_call("get_active_memory"),
        "mlx_cache_bytes": _mlx_call("get_cache_memory"),
    }


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(after_value, (int, float)) and isinstance(before_value, (int, float)):
            out[key] = after_value - before_value
    return out


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "samples": []}
    ordered = sorted(values)
    p90 = ordered[max(0, min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1))]
    return {
        "n": len(values),
        "samples": [float(v) for v in values],
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "p90": float(p90),
        "max": float(max(values)),
    }


def _max_abs_rel(a: mx.array, b: mx.array) -> dict[str, float]:
    max_abs = float(mx.max(mx.abs(a - b)).item())
    scale = float(mx.max(mx.abs(a)).item())
    return {"max_abs": max_abs, "max_rel": 0.0 if max_abs == 0.0 or scale == 0.0 else max_abs / scale}


def _build_timestep_plan(layout, video_timesteps, audio_timesteps):
    per_step = []
    for t, at in zip(video_timesteps.tolist(), audio_timesteps.tolist()):
        distinct, inverse = build_row_timesteps(
            layout, float(t), float(at), max(float(t), KEYFRAME_NOISE_AUG), 1.0
        )
        per_step.append((np.array(distinct), np.array(inverse)))
    table = sorted({float(v) for distinct, _ in per_step for v in distinct})
    lookup = {v: i for i, v in enumerate(table)}
    plan = []
    for distinct, inverse in per_step:
        remap = np.array([lookup[float(v)] for v in distinct], dtype=np.int32)
        plan.append(mx.array(remap[inverse].astype(np.int32)))
    return mx.array(np.array(table, dtype=np.float32)), plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", default="models/MiniMax-H3/FL2VA")
    parser.add_argument("--transformer", default="models/MiniMax-H3-MLX-4bit")
    parser.add_argument("--text-encoder", default="models/MiniMax-H3/FL2VA/text_encoder-mlx-4bit")
    parser.add_argument("--turbo-lora", default=None)
    parser.add_argument("--turbo-lora-alpha", type=float, default=8.0)
    parser.add_argument("--turbo-lora-scale", type=float, default=1.0)
    parser.add_argument("--block-load-mode", choices=BLOCK_LOAD_MODES, default="mlx")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-pairs", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--out", default="experiments/text_conditioning_cache/latest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    events: list[dict[str, Any]] = []

    def log(message: str, **fields: Any) -> None:
        record = {"time": _now(), "message": message, **fields}
        events.append(record)
        suffix = " " + json.dumps(fields, sort_keys=True) if fields else ""
        print(f"[{record['time']}] {message}{suffix}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log("starting text-conditioning cache profiler", out=str(out_path))

    mx.set_cache_limit(0)
    mx.set_memory_limit(int(args.memory_limit_gb * 1e9))
    preflight = {"metrics": _metrics(), "swapusage": _swapusage()}

    root = Path(args.checkpoint)
    transformer = Path(args.transformer)
    text_encoder_path = Path(args.text_encoder)
    pipe_cfg = PipelineConfig.from_model_index(root / "model_index.json")
    video_cfg = read_video_vae_config(root / "video_vae")
    audio_cfg = read_audio_vae_config(root / "audio_vae")

    log("loading text encoder")
    text_encoder = MiniMaxH3TextEncoder(text_encoder_path, load_vision=False, verbose=False)
    prompt_embeds, text_token_tags = text_encoder.encode(args.prompt, None)
    prompt_embeds = detach_bfloat16(prompt_embeds)
    text_token_tags = np.array(text_token_tags, copy=True)
    del text_encoder
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    log("encoded prompt", text_tokens=int(text_token_tags.shape[0]))

    log("loading streaming DiT", transformer=str(transformer), block_load_mode=args.block_load_mode)
    dit, provider = load_streaming_dit(
        transformer,
        turbo_lora_path=args.turbo_lora,
        turbo_lora_alpha=args.turbo_lora_alpha,
        turbo_lora_scale=args.turbo_lora_scale,
        block_load_mode=args.block_load_mode,
        verbose=False,
    )

    if args.height % 32 or args.width % 32:
        raise ValueError("height/width must be multiples of 32")
    num_frames = align_num_frames(int(round(args.duration * FPS)))
    num_latent_frames = video_latent_num_frames(num_frames)
    latent_height = args.height // video_cfg.spatial_compression_ratio
    latent_width = args.width // video_cfg.spatial_compression_ratio
    num_audio_latents = audio_latent_num_frames(num_frames)
    layout = build_packed_sequence(
        text_token_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        dit.config.patch_size,
        (),
    )
    video_sched = MiniMaxH3Scheduler(shift=pipe_cfg.sigma_shift_video)
    audio_sched = MiniMaxH3Scheduler(shift=pipe_cfg.sigma_shift_audio)
    video_sched.set_timesteps(args.steps)
    audio_sched.set_timesteps(args.steps)
    timestep_table, plan = _build_timestep_plan(layout, video_sched.timesteps, audio_sched.timesteps)

    mx.random.seed(args.seed)
    latents = mx.random.normal(
        (1, video_cfg.latent_channels, num_latent_frames, latent_height, latent_width)
    ).astype(mx.float32)
    video_rows = patchify_video_latents(latents, dit.config.patch_size)
    audio_rows = mx.random.normal(
        (num_audio_latents * AUDIO_CHANNELS, audio_cfg.latent_channels)
    ).astype(mx.float32)
    embeds = prompt_embeds.astype(mx.bfloat16)
    mx.eval(video_rows, audio_rows, embeds, timestep_table, *plan)

    log("building adaln cache", timesteps=int(timestep_table.shape[0]), sequence_length=int(layout.sequence_length))
    modulation_cache = ModulationCache.build_streaming(dit, provider, timestep_table, dtype=mx.bfloat16)
    log("precomputing refined text")
    refined_text = dit.precompute_text_conditioning(embeds, block_provider=provider)
    mx.eval(refined_text)

    common = (
        video_rows[None].astype(mx.bfloat16),
        audio_rows[None].astype(mx.bfloat16),
        timestep_table,
        plan[0],
        layout.token_tags,
        layout.position_ids,
        layout.video_indices,
        layout.audio_indices,
        layout.text_indices,
    )

    def run_once(label: str):
        use_cache = label == "candidate"
        _reset_mlx_peak()
        before = _metrics()
        call_started = time.perf_counter()
        video_out, audio_out = dit(
            common[0],
            common[1],
            None if use_cache else embeds,
            common[2],
            common[3],
            common[4],
            common[5],
            common[6],
            common[7],
            common[8],
            modulation_cache=modulation_cache,
            block_provider=provider,
            refined_text=refined_text if use_cache else None,
        )
        mx.eval(video_out, audio_out)
        elapsed = time.perf_counter() - call_started
        after = _metrics()
        return {
            "label": label,
            "seconds": elapsed,
            "metrics_before": before,
            "metrics_after": after,
            "metrics_delta": _delta(before, after),
        }, video_out, audio_out

    warmups = []
    for pair in range(args.warmup_pairs):
        order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        for label in order:
            sample, _, _ = run_once(label)
            warmups.append({"pair": pair, **sample})
        log("warmup pair complete", pair=pair + 1, total=args.warmup_pairs)

    samples = []
    parity = []
    for pair in range(args.pairs):
        order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        outputs: dict[str, tuple[mx.array, mx.array]] = {}
        pair_samples: dict[str, dict[str, Any]] = {}
        for label in order:
            sample, video_out, audio_out = run_once(label)
            sample["pair"] = pair
            sample["order"] = list(order)
            samples.append(sample)
            pair_samples[label] = sample
            outputs[label] = (video_out, audio_out)
        video_delta = _max_abs_rel(outputs["baseline"][0], outputs["candidate"][0])
        audio_delta = _max_abs_rel(outputs["baseline"][1], outputs["candidate"][1])
        pair_parity = {"pair": pair, "video": video_delta, "audio": audio_delta}
        parity.append(pair_parity)
        log(
            "measured pair complete",
            pair=pair + 1,
            total=args.pairs,
            baseline_seconds=pair_samples["baseline"]["seconds"],
            candidate_seconds=pair_samples["candidate"]["seconds"],
            video_max_abs=video_delta["max_abs"],
            audio_max_abs=audio_delta["max_abs"],
        )

    baseline_seconds = [s["seconds"] for s in samples if s["label"] == "baseline"]
    candidate_seconds = [s["seconds"] for s in samples if s["label"] == "candidate"]
    paired_deltas = []
    for pair in range(args.pairs):
        b = next(s["seconds"] for s in samples if s["pair"] == pair and s["label"] == "baseline")
        c = next(s["seconds"] for s in samples if s["pair"] == pair and s["label"] == "candidate")
        paired_deltas.append(float(c - b))

    parity_ok = all(p["video"]["max_abs"] == 0.0 and p["audio"]["max_abs"] == 0.0 for p in parity)
    median_delta = float(statistics.median(paired_deltas)) if paired_deltas else float("nan")
    candidate_peak = [s["metrics_after"].get("mlx_peak_bytes") for s in samples if s["label"] == "candidate"]
    baseline_peak = [s["metrics_after"].get("mlx_peak_bytes") for s in samples if s["label"] == "baseline"]
    comparable_peaks = [
        (b, c) for b, c in zip(baseline_peak, candidate_peak) if isinstance(b, int) and isinstance(c, int)
    ]
    peak_delta_bytes = None
    if comparable_peaks:
        peak_delta_bytes = int(statistics.median([c - b for b, c in comparable_peaks]))
    if not parity_ok:
        decision = {"status": "reject_parity", "promote_candidate": False}
    elif median_delta < 0.0 and (peak_delta_bytes is None or peak_delta_bytes <= refined_text.nbytes):
        decision = {
            "status": "retain_disabled_candidate",
            "promote_candidate": False,
            "reason": "strict-equivalent and timed faster in this focused run, but it remains opt-in pending reviewer scrutiny",
        }
    else:
        decision = {
            "status": "retain_disabled_no_clear_win",
            "promote_candidate": False,
            "reason": "strict-equivalent but focused timing or memory did not justify promotion",
        }

    result = {
        "kind": "text_conditioning_cache_profile",
        "created_at": _now(),
        "command": sys.argv,
        "preflight": preflight,
        "config": {
            "checkpoint": str(root),
            "transformer": str(transformer),
            "text_encoder": str(text_encoder_path),
            "height": args.height,
            "width": args.width,
            "duration": args.duration,
            "steps": args.steps,
            "seed": args.seed,
            "warmup_pairs": args.warmup_pairs,
            "measured_pairs": args.pairs,
            "block_load_mode": args.block_load_mode,
            "turbo_lora": args.turbo_lora,
        },
        "layout": {
            "text_tokens": int(text_token_tags.shape[0]),
            "sequence_length": int(layout.sequence_length),
            "video_rows": int(layout.video_indices.shape[0]),
            "audio_rows": int(layout.audio_indices.shape[0]),
            "latent_height": int(latent_height),
            "latent_width": int(latent_width),
            "num_latent_frames": int(num_latent_frames),
            "num_audio_latents": int(num_audio_latents),
        },
        "candidate": {
            "name": "precomputed_refined_text",
            "default_enabled": False,
            "refined_text_shape": list(refined_text.shape),
            "refined_text_nbytes": int(refined_text.nbytes),
            "skips": ["condition_proj", "token_refiner"],
        },
        "warmups": warmups,
        "samples": samples,
        "timing": {
            "baseline": _stats(baseline_seconds),
            "candidate": _stats(candidate_seconds),
            "paired_candidate_minus_baseline_seconds": _stats(paired_deltas),
            "median_candidate_minus_baseline_seconds": median_delta,
        },
        "parity": {"ok": parity_ok, "pairs": parity},
        "memory": {
            "baseline_peak_bytes": baseline_peak,
            "candidate_peak_bytes": candidate_peak,
            "median_candidate_minus_baseline_peak_bytes": peak_delta_bytes,
            "retained_refined_text_bytes": int(refined_text.nbytes),
        },
        "decision": decision,
        "events": events,
        "total_wall_seconds": time.perf_counter() - started,
    }
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    log("wrote profile artifact", out=str(out_path), decision=decision["status"])
    print(json.dumps({"out": str(out_path), "decision": decision, "parity_ok": parity_ok}, indent=2))
    return 0 if parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
