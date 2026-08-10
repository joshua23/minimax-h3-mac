#!/usr/bin/env python3
"""Profile one warm MiniMax-H3 DiT block compute hotpath.

This harness deliberately loads at most one real quantized transformer block, builds a
representative 320x192 packed sequence without running the text encoder or VAEs, and
times the compute sub-segments that the low-memory generation path executes after the
block is already resident.  Optional Metal capture wraps the *uninstrumented* full
block call so kernel-boundary evidence is not inferred from the per-segment mx.eval
barriers used for Amdahl timing.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from minimax_h3_mlx.config import MODALITY_NUM, TAG_TEXT, DiTConfig  # noqa: E402
from minimax_h3_mlx.dit import (  # noqa: E402
    DENSE_DEQUANT_GENERATION_PROFILES,
    DENSE_DEQUANT_PROFILE_OFF,
    DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
    DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED,
    MiniMaxH3DiT,
    RotaryPosEmbed3D,
    TransformerBlock,
    apply_dense_dequant_profile_to_block,
    apply_rotary,
    apply_rotary_qk_metal,
    gather_packed_modulation_rows,
    indexed_adaln_affine_metal,
    indexed_gated_residual_metal,
    materialize_attention_input_contiguous,
    materialize_attention_output_contiguous,
    materialize_ffn_hidden_contiguous,
    materialize_ffn_input_contiguous,
    sdpa_head_batch_rank3,
    sdpa_headgroup_split_rank4,
    sdpa_out_to_bshd_metal,
)
from minimax_h3_mlx.packing import (  # noqa: E402
    AUDIO_CHANNELS,
    FPS,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    video_latent_num_frames,
)
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler  # noqa: E402

DEFAULT_OUT = "experiments/dit_block_hotpath/latest_profile.json"
TRACE_TOKENS = (
    "rms",
    "rmsnorm",
    "layernorm",
    "rope",
    "rotary",
    "scaled_dot_product",
    "attention",
    "softmax",
    "quantized",
    "gemm",
    "steel_gemm",
    "metal_kernel",
    "silu",
    "multiply",
)

# Public profiler surface after the deployability cleanup.  The implementation below still
# preserves older helper functions as provenance/archive, but argparse no longer exposes rejected
# or tiny-only micro-candidates for fresh enumeration.  Keep only the dense "none" baseline plus
# the individual real-block dense-dequant helpers that remain disabled-by-default building blocks.
ACTIVE_HOTPATH_CANDIDATES = (
    "none",
    "ffn_fc2_tiled_dense_dequant",
    "attention_out_dense_dequant",
    "attention_out_tiled_dense_dequant",
    "attention_qkv_tiled_dense_dequant",
)

QUARANTINED_HOTPATH_CANDIDATES = (
    "ffn_hidden_tile_stream",
    "promoted_dense_dequant_combo",
    "compile_block",
    "ffn_projection_2d_qmm",
    "ffn_fc1_rank2_qmm",
    "ffn_fc2_rank2_qmm",
    "ffn_fc2_input_chunked_qmm",
    "ffn_fc1_dense_dequant",
    "ffn_fc1_tiled_dense_dequant",
    "ffn_fc2_dense_dequant",
    "ffn_subgraph_compile",
    "indexed_adaln_affine_metal",
    "indexed_gated_residual_metal",
    "attention_projection_2d_qmm",
    "attention_qkv_headgroup_row_sliced_qmm",
    "attention_qkv_input_chunked_qmm",
    "attention_sdpa_head_batch_rank3",
    "attention_sdpa_headgroup_split_rank4",
)

ARCHIVED_NONPROMOTED_HOTPATH_CANDIDATES = (
    "ffn_mx_split_swiglu",
    "ffn_metal_swiglu",
    "ffn_fused_swiglu_metal",
    "ffn_fc1_split_gate_value_quantized_qmm",
    "ffn_pre_fc1_contiguous",
    "ffn_pre_fc2_contiguous",
    "ffn_sequence_chunked",
    "adaln_packed_gather",
    "attention_pre_qkv_contiguous",
    "attention_qkv_pretranspose_layout",
    "attention_pre_sdpa_contiguous",
    "attention_pre_out_proj_contiguous",
    "attention_qkv_rmsnorm_sdpa_metal",
    "attention_qkv_rmsnorm_rotary_sdpa_metal",
    "attention_sdpa_out_layout_metal",
    "attention_rotary_qk_metal",
)

ARCHIVED_HOTPATH_CANDIDATES = QUARANTINED_HOTPATH_CANDIDATES + ARCHIVED_NONPROMOTED_HOTPATH_CANDIDATES


def _tiny_config() -> DiTConfig:
    hidden = 64
    return DiTConfig(
        hidden_size=hidden,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=16,
        ffn_hidden_size=32,
        latents_dim=4,
        audio_latents_dim=8,
        patch_size=(1, 2, 2),
        text_dim=32,
        timestep_input_dim=16,
        time_embed_hidden_size=hidden,
        time_embed_dim=32,
        adaln_out_features=6 * 3 * hidden,
        final_adaln_out_features=2 * hidden,
        rope_inv_freq_len=2,
    )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(events: list[dict[str, Any]], message: str, **fields: Any) -> None:
    record = {"time": _now(), "message": message, **fields}
    events.append(record)
    suffix = " " + json.dumps(fields, sort_keys=True) if fields else ""
    print(f"[{record['time']}] {message}{suffix}", flush=True)


def _current_rss_kib() -> int | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(out.strip())
    except Exception:
        return None


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
        return subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
    except Exception:
        return None


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


def _metrics() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "wall_time_seconds": time.perf_counter(),
        "current_rss_kib": _current_rss_kib(),
        "ru_maxrss_raw": usage.ru_maxrss,
        "ru_minflt": usage.ru_minflt,
        "ru_majflt": usage.ru_majflt,
        "ru_inblock": usage.ru_inblock,
        "ru_oublock": usage.ru_oublock,
        "ru_nvcsw": usage.ru_nvcsw,
        "ru_nivcsw": usage.ru_nivcsw,
        "vm_pageouts": _vm_stat_value("Pageouts"),
        "vm_swapouts": _vm_stat_value("Swapouts"),
        "vm_swapusage": _swapusage(),
        "mlx_peak_bytes": _mlx_call("get_peak_memory"),
        "mlx_active_bytes": _mlx_call("get_active_memory"),
        "mlx_cache_bytes": _mlx_call("get_cache_memory"),
    }


def _safety_preflight() -> dict[str, Any]:
    """Record process/disk/memory state before loading a real block."""

    stat = os.statvfs(Path.cwd())
    disk_free_bytes = int(stat.f_bavail * stat.f_frsize)
    rows: list[dict[str, Any]] = []
    try:
        ps_out = subprocess.check_output(
            ["ps", "-axo", "pid,ppid,%cpu,%mem,rss,comm,args"],
            text=True,
        )
    except Exception as exc:
        ps_error = f"{type(exc).__name__}: {exc}"
    else:
        ps_error = None
        for line in ps_out.splitlines()[1:]:
            parts = line.split(None, 6)
            if len(parts) < 7:
                continue
            pid, ppid, cpu, mem, rss, comm, args = parts
            haystack = f"{comm} {args}".lower()
            if not any(token in haystack for token in ("python", "mlx", "generate", "ffmpeg", "argus")):
                continue
            try:
                rss_kib = int(rss)
            except ValueError:
                rss_kib = 0
            rows.append(
                {
                    "pid": int(pid),
                    "ppid": int(ppid),
                    "cpu_percent": float(cpu),
                    "mem_percent": float(mem),
                    "rss_kib": rss_kib,
                    "comm": comm,
                    "args": args[:240],
                }
            )
    high_memory_threshold_kib = 8 * 1024 * 1024
    high_memory_rows = [
        row
        for row in rows
        if row["pid"] != os.getpid() and row["rss_kib"] >= high_memory_threshold_kib
    ]
    return {
        "recorded_at": _now(),
        "cwd": str(Path.cwd()),
        "disk_free_bytes": disk_free_bytes,
        "disk_free_gib": disk_free_bytes / (1024 ** 3),
        "meets_100gb_free_contract": disk_free_bytes >= 100 * 1024 ** 3,
        "memory_metrics_before_load": _metrics(),
        "process_scan_error": ps_error,
        "process_scan_filter": "python|mlx|generate|ffmpeg|argus",
        "high_memory_threshold_rss_kib": high_memory_threshold_kib,
        "other_high_memory_process_count": len(high_memory_rows),
        "other_high_memory_processes": high_memory_rows,
        "relevant_processes": rows[:20],
    }


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(after_value, (int, float)) and isinstance(before_value, (int, float)):
            out[key] = after_value - before_value
    return out


def _eval_tree(value: Any) -> None:
    if isinstance(value, mx.array):
        mx.eval(value)
    elif isinstance(value, (tuple, list)):
        mx.eval(*[v for v in value if isinstance(v, mx.array)])
    elif isinstance(value, dict):
        mx.eval(*[v for v in value.values() if isinstance(v, mx.array)])


def _p90(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1))
    return float(ordered[index])


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "samples": [float(v) for v in values],
        "n": len(values),
        "median_seconds": float(statistics.median(values)) if values else None,
        "p90_seconds": _p90(values) if values else None,
        "min_seconds": float(min(values)) if values else None,
        "max_seconds": float(max(values)) if values else None,
    }


def _bootstrap_median_ci(
    values: list[float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile bootstrap CI for a median, used only as noise evidence."""
    if len(values) < 2 or resamples <= 0:
        return {
            "statistic": "median",
            "confidence": confidence,
            "resamples": 0,
            "low": None,
            "high": None,
            "reason": "need at least two samples and positive resamples",
        }
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.shape[0])
    boot = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        boot[i] = float(np.median(arr[rng.integers(0, n, size=n)]))
    alpha = (1.0 - confidence) / 2.0
    return {
        "statistic": "median",
        "confidence": confidence,
        "resamples": int(resamples),
        "low": float(np.quantile(boot, alpha)),
        "high": float(np.quantile(boot, 1.0 - alpha)),
    }


def _noise_decision(delta_ci: dict[str, Any]) -> str:
    """Interpret candidate-baseline timing deltas without any fixed percent threshold."""
    low = delta_ci.get("low")
    high = delta_ci.get("high")
    if low is None or high is None:
        return "insufficient_samples"
    if high < 0.0:
        return "candidate_faster_than_noise"
    if low > 0.0:
        return "candidate_slower_than_noise"
    return "indistinguishable_from_noise"


def _time_callable(fn: Callable[[], mx.array], *, warmups: int, repeats: int) -> tuple[dict[str, Any], mx.array]:
    samples: list[float] = []
    last: mx.array | None = None
    for i in range(warmups + repeats):
        mx.synchronize()
        started = time.perf_counter()
        out = fn()
        _eval_tree(out)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        if i >= warmups:
            samples.append(elapsed)
        last = out
    assert last is not None
    return _stats(samples), last


def _time_interleaved_pairwise(
    baseline_fn: Callable[[], mx.array],
    candidate_fn: Callable[[], mx.array],
    *,
    warmups: int,
    repeats: int,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[dict[str, Any], mx.array, mx.array]:
    """Alternate baseline/candidate order and bootstrap paired timing deltas."""

    def timed(fn: Callable[[], mx.array]) -> tuple[float, mx.array]:
        mx.synchronize()
        started = time.perf_counter()
        out = fn()
        _eval_tree(out)
        mx.synchronize()
        return time.perf_counter() - started, out

    measured_pairs: list[dict[str, Any]] = []
    warmup_pairs: list[dict[str, Any]] = []
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    deltas: list[float] = []
    ratios: list[float] = []
    last_baseline: mx.array | None = None
    last_candidate: mx.array | None = None

    for i in range(warmups + repeats):
        order = ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
        pair: dict[str, Any] = {"pair_index": i, "order": list(order)}
        for name in order:
            elapsed, out = timed(baseline_fn if name == "baseline" else candidate_fn)
            pair[f"{name}_seconds"] = float(elapsed)
            if name == "baseline":
                last_baseline = out
            else:
                last_candidate = out
        pair["candidate_minus_baseline_seconds"] = (
            float(pair["candidate_seconds"]) - float(pair["baseline_seconds"])
        )
        pair["baseline_over_candidate_speedup"] = (
            float(pair["baseline_seconds"]) / float(pair["candidate_seconds"])
            if float(pair["candidate_seconds"]) > 0.0
            else None
        )
        if i < warmups:
            warmup_pairs.append(pair)
            continue
        measured_index = i - warmups
        pair["pair_index"] = measured_index
        measured_pairs.append(pair)
        baseline_samples.append(float(pair["baseline_seconds"]))
        candidate_samples.append(float(pair["candidate_seconds"]))
        deltas.append(float(pair["candidate_minus_baseline_seconds"]))
        if pair["baseline_over_candidate_speedup"] is not None:
            ratios.append(float(pair["baseline_over_candidate_speedup"]))

    assert last_baseline is not None and last_candidate is not None
    delta_ci = _bootstrap_median_ci(deltas, resamples=bootstrap_resamples, seed=seed)
    baseline_stats = _stats(baseline_samples)
    candidate_stats = _stats(candidate_samples)
    baseline_median = baseline_stats.get("median_seconds")
    candidate_median = candidate_stats.get("median_seconds")
    median_delta = (
        float(candidate_median) - float(baseline_median)
        if baseline_median is not None and candidate_median is not None
        else None
    )
    protocol = {
        "protocol": "paired_interleaved_alternating_order",
        "warmup_pairs": int(warmups),
        "measured_pairs": int(repeats),
        "order_rule": "even pairs baseline->candidate; odd pairs candidate->baseline",
        "baseline_timing": baseline_stats,
        "candidate_timing": candidate_stats,
        "paired_samples": measured_pairs,
        "warmup_samples": warmup_pairs,
        "candidate_minus_baseline_delta_seconds": {
            **_stats(deltas),
            "median_delta_from_medians_seconds": median_delta,
        },
        "baseline_over_candidate_speedup": _stats(ratios),
        "bootstrap_ci_candidate_minus_baseline_median_seconds": delta_ci,
        "noise_decision": _noise_decision(delta_ci),
    }
    return protocol, last_baseline, last_candidate


def _diff_stats(reference: mx.array, candidate: mx.array) -> dict[str, float]:
    diff = (candidate.astype(mx.float32) - reference.astype(mx.float32)).astype(mx.float32)
    ref = reference.astype(mx.float32)
    abs_diff = mx.abs(diff)
    abs_ref = mx.abs(ref)
    max_abs = float(mx.max(abs_diff).item())
    max_rel = 0.0 if max_abs == 0.0 else float(mx.max(abs_diff / mx.maximum(abs_ref, 1e-12)).item())
    numerator = float(mx.sqrt(mx.sum(diff * diff)).item())
    denominator = float(mx.sqrt(mx.sum(ref * ref)).item())
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rel_l2": numerator / denominator if denominator else (0.0 if numerator == 0.0 else float("inf")),
    }


def _load_block(args: argparse.Namespace, events: list[dict[str, Any]]) -> tuple[TransformerBlock, DiTConfig, dict[str, Any]]:
    if args.tiny:
        cfg = _tiny_config()
        mx.random.seed(args.seed)
        model = MiniMaxH3DiT(cfg)
        mx.eval(model.parameters())
        _log(events, "loaded tiny random block", hidden_size=cfg.hidden_size)
        return model.blocks[0], cfg, {"mode": "tiny_random", "real_4bit": False}

    from mlx.utils import tree_flatten
    from minimax_h3_mlx.streaming import QuantizedBlockProvider

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    _log(events, "loading real quantized block", model_dir=str(model_dir), block_index=args.block_index)
    started = time.perf_counter()
    provider = QuantizedBlockProvider(model_dir, block_load_mode=args.block_load_mode)
    block = provider.load_block(args.block_index, include_adaln=True)
    flat = dict(tree_flatten(provider.slot.parameters()))
    mx.eval(*flat.values())
    mx.synchronize()
    elapsed = time.perf_counter() - started
    recipe: dict[str, Any] = {}
    quant_path = model_dir / "quant_config.json"
    if quant_path.exists():
        recipe = json.loads(quant_path.read_text())
    _log(events, "loaded and materialized block", seconds=elapsed, tensors=len(flat))
    return block, provider.config, {
        "mode": "quantized_provider",
        "real_4bit": True,
        "model_dir": str(model_dir),
        "block_load_mode": args.block_load_mode,
        "block_index": args.block_index,
        "load_materialize_seconds": elapsed,
        "logical_bytes_loaded": provider.logical_bytes_loaded,
        "quant_config": recipe,
    }


def _build_inputs(
    cfg: DiTConfig,
    args: argparse.Namespace,
    block: TransformerBlock,
    events: list[dict[str, Any]],
) -> tuple[mx.array, tuple[mx.array, ...], mx.array, tuple[mx.array, mx.array], dict[str, Any]]:
    height = int(args.height)
    width = int(args.width)
    if height % 32 or width % 32:
        raise ValueError(f"height/width must be multiples of 32, got {height}x{width}")
    if height % args.spatial_compression or width % args.spatial_compression:
        raise ValueError(
            f"height/width must be divisible by spatial compression {args.spatial_compression}, "
            f"got {height}x{width}"
        )
    num_frames = align_num_frames(int(round(args.duration * FPS)))
    num_latent_frames = video_latent_num_frames(num_frames)
    latent_height = height // args.spatial_compression
    latent_width = width // args.spatial_compression
    num_audio_latents = audio_latent_num_frames(num_frames)
    text_tags = np.full((args.text_tokens,), TAG_TEXT, dtype=np.int64)
    layout = build_packed_sequence(
        text_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        cfg.patch_size,
    )

    video_sched = MiniMaxH3Scheduler(shift=args.video_sigma_shift)
    audio_sched = MiniMaxH3Scheduler(shift=args.audio_sigma_shift)
    video_sched.set_timesteps(args.sigma_grid_points)
    audio_sched.set_timesteps(args.sigma_grid_points)
    step_index = min(max(args.step_index, 0), len(video_sched.timesteps.tolist()) - 1)
    distinct, timestep_indices = build_row_timesteps(
        layout,
        float(video_sched.timesteps[step_index].item()),
        float(audio_sched.timesteps[step_index].item()),
        max(float(video_sched.timesteps[step_index].item()), 0.999),
        1.0,
    )
    adaln_indices = timestep_indices * MODALITY_NUM + mx.maximum(layout.token_tags, 0)
    # Keep construction explicit in the artifact: 3D RoPE, no cached trace shortcut.
    rotary = RotaryPosEmbed3D(cfg)(layout.position_ids)
    # This one-block harness does not load the static time_embedder weights.  The hot block path
    # sees only the precomputed modulation table, so representative random vectors at the real
    # time_embed_dim exercise the same AdaLN projection shape to build that table once outside the
    # timed region.
    mx.random.seed(args.seed + 1)
    temb = mx.random.normal((int(distinct.shape[0]), cfg.time_embed_dim)).astype(mx.float32)
    modulation = block.adaln_proj(temb)
    mx.eval(*modulation, adaln_indices, *rotary)

    dtype = mx.bfloat16 if args.activation_dtype == "bf16" else mx.float32
    mx.random.seed(args.seed)
    x = mx.random.normal((1, layout.sequence_length, cfg.hidden_size)).astype(dtype)
    mx.eval(x)
    _log(
        events,
        "built representative packed block input",
        sequence_length=layout.sequence_length,
        height=height,
        width=width,
        duration=args.duration,
        activation_dtype=args.activation_dtype,
    )
    _, ph, pw = cfg.patch_size
    rows_per_frame = (latent_height // ph) * (latent_width // pw)
    metadata = {
        "height": height,
        "width": width,
        "duration_seconds": args.duration,
        "fps": FPS,
        "aligned_num_frames": num_frames,
        "num_latent_frames": num_latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "rows_per_video_latent_frame": rows_per_frame,
        "num_audio_latents": num_audio_latents,
        "audio_channels": AUDIO_CHANNELS,
        "text_tokens": args.text_tokens,
        "sequence_length": layout.sequence_length,
        "video_rows": int(len(layout.video_indices.tolist())),
        "audio_rows": int(len(layout.audio_indices.tolist())),
        "text_rows": int(len(layout.text_indices.tolist())),
        "condition_video_rows": layout.num_condition_video_rows,
        "sigma_grid_points": args.sigma_grid_points,
        "denoiser_evaluations_for_grid": args.sigma_grid_points - 1,
        "profiled_step_index": step_index,
        "distinct_timestep_count_for_step": int(distinct.shape[0]),
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "estimated_full_generation_block_calls": cfg.num_layers * (args.sigma_grid_points - 1),
        "heads": cfg.num_attention_heads,
        "head_dim": cfg.attention_head_dim,
        "inner_dim": cfg.inner_dim,
        "ffn_hidden_size": cfg.ffn_hidden_size,
        "rotary_dim": cfg.rotary_dim,
        "patch_size": list(cfg.patch_size),
        "activation_dtype": args.activation_dtype,
        "modulation_projection_in_timed_hotpath": False,
        "modulation_source": "block.adaln_proj(random time_embed_dim vectors); time_embedder static weights are intentionally not loaded in the one-block harness",
        "packed_order": "[text | keyframe conditions | target audio | target video]",
    }
    return x, modulation, adaln_indices, rotary, metadata


def _run_segmented_once(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
) -> tuple[dict[str, float], mx.array]:
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
    samples: dict[str, float] = {}

    def timed(label: str, fn: Callable[[], mx.array]) -> mx.array:
        mx.synchronize()
        started = time.perf_counter()
        value = fn()
        _eval_tree(value)
        mx.synchronize()
        samples[label] = time.perf_counter() - started
        return value

    B, S, _ = x.shape
    h = timed(
        "adaln_msa_indexed_norm_affine",
        lambda: block.norm1(x) * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices],
    )
    qkv = timed(
        "qkv_quantized_matmul",
        lambda: block.attn.qkv_proj(h).reshape(B, S, block.attn.heads, 3, block.attn.head_dim),
    )

    def qk_rope_layout() -> tuple[mx.array, mx.array, mx.array]:
        q = qkv[:, :, :, 0]
        k = qkv[:, :, :, 1]
        v = qkv[:, :, :, 2]
        q = block.attn.q_norm(q).transpose(0, 2, 1, 3)
        k = block.attn.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = apply_rotary(q, *rotary)
        k = apply_rotary(k, *rotary)
        return q, k, v

    q, k, v = timed("qk_rmsnorm_rope_layout", qk_rope_layout)
    attn = timed(
        "mlx_fast_sdpa",
        lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=block.attn.scale, mask=None),
    )
    attn_out = timed(
        "out_projection",
        lambda: block.attn.out_proj(
            attn.transpose(0, 2, 1, 3).reshape(B, S, block.attn.heads * block.attn.head_dim).astype(h.dtype)
        ),
    )
    x_attn = timed(
        "msa_gated_residual",
        lambda: x + gate_msa[adaln_indices] * attn_out,
    )
    h2 = timed(
        "adaln_mlp_indexed_norm_affine",
        lambda: block.norm2(x_attn) * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices],
    )

    def ffn() -> mx.array:
        fused = block.mlp.fc1(h2)
        gate, value = fused[..., : block.mlp._ffn], fused[..., block.mlp._ffn :]
        hidden = nn.silu(gate) * value
        return block.mlp.fc2(hidden)

    mlp_out = timed("fc1_swiglu_fc2", ffn)
    out = timed("mlp_gated_residual", lambda: x_attn + gate_mlp[adaln_indices] * mlp_out)
    return samples, out


def _time_segmented(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    *,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, dict[str, Any]], mx.array]:
    per_label: dict[str, list[float]] = {}
    last: mx.array | None = None
    for i in range(warmups + repeats):
        samples, out = _run_segmented_once(block, x, modulation, adaln_indices, rotary)
        if i >= warmups:
            for label, value in samples.items():
                per_label.setdefault(label, []).append(value)
        last = out
    assert last is not None
    return {label: _stats(values) for label, values in per_label.items()}, last


def _path_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _shape_bytes(shape: tuple[int, ...], *, itemsize: int = 2) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return int(n * itemsize)


def _function_role(name: str) -> str:
    lower = name.lower()
    if "affine_qmm" in lower and "_b_4" in lower:
        return "4bit_quantized_dequant_gemm"
    if "affine_qmv" in lower:
        return "quantized_vector_affine"
    if lower.startswith("rms"):
        return "rmsnorm_reduction"
    if "gather" in lower:
        return "indexed_materialization_gather"
    if "sigmoid" in lower or "silu" in lower:
        return "swiglu_pointwise"
    if "multiply" in lower or "add" in lower or "negative" in lower or "copy" in lower:
        return "pointwise_or_layout_materialization"
    if "softmax" in lower or "attention" in lower:
        return "attention"
    if re.fullmatch(r"[0-9A-F]{15,16}", name):
        return "generated_or_hashed_mlx_kernel"
    return "other_mlx_kernel"


def _extract_resource_functions(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_dir():
        return {"active": [], "unused": [], "counts_by_role": {}}
    active: list[dict[str, Any]] = []
    unused: list[dict[str, Any]] = []
    for resource in sorted(path.glob("*device-resources*")):
        try:
            strings = [
                s.decode("utf-8", "replace")
                for s in re.findall(rb"[ -~]{4,}", resource.read_bytes())
            ]
        except OSError:
            continue
        rows = unused if resource.name.startswith("unused-") else active
        for i, token in enumerate(strings):
            if token != "function" or i + 1 >= len(strings):
                continue
            name = strings[i + 1]
            source = None
            for candidate in strings[i + 2 : i + 8]:
                if candidate.startswith("/") or candidate.endswith(".h") or candidate.endswith(".metal"):
                    source = candidate
                    break
            rows.append(
                {
                    "name": name,
                    "role": _function_role(name),
                    "source": source,
                    "resource_file": resource.name,
                }
            )
    counts: dict[str, int] = {}
    for row in active:
        counts[row["role"]] = counts.get(row["role"], 0) + 1
    return {"active": active, "unused": unused, "counts_by_role": counts}


def _buffer_sizes(path: Path) -> dict[str, int]:
    if not path.exists() or not path.is_dir():
        return {}
    out: dict[str, int] = {}
    for child in path.iterdir():
        if not child.is_file() or not child.name.startswith("MTLBuffer-"):
            continue
        try:
            out[child.name] = int(child.stat().st_size)
        except OSError:
            continue
    return out


def _buffer_matches(buffers: dict[str, int], expected_bytes: int) -> list[dict[str, Any]]:
    tolerance = max(4096, int(expected_bytes * 0.01))
    matches = []
    for name, size in buffers.items():
        delta = int(size) - int(expected_bytes)
        if abs(delta) <= tolerance:
            matches.append({"name": name, "size_bytes": size, "delta_bytes": delta})
    return sorted(matches, key=lambda row: (abs(row["delta_bytes"]), row["name"]))[:8]


def _expected_materialization_boundaries(
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    buffers: dict[str, int],
) -> list[dict[str, Any]]:
    s = int(sequence_meta["sequence_length"])
    h = int(cfg.hidden_size)
    inner = int(cfg.inner_dim)
    heads = int(cfg.num_attention_heads)
    head_dim = int(cfg.attention_head_dim)
    ffn = int(cfg.ffn_hidden_size)
    specs = [
        ("hidden_residual_or_normed_rows", (1, s, h), "block input/output, AdaLN normalized h, gated residual tensors"),
        ("qkv_projected_interleaved", (1, s, heads, 3, head_dim), "qkv projection output before split/layout"),
        ("q_or_k_or_v_attention_layout", (1, heads, s, head_dim), "one materialized q/k/v tensor after transpose/normalization/rotary"),
        ("attention_merged_heads", (1, s, inner), "attention output before out projection"),
        ("ffn_fc1_gate_value", (1, s, 2 * ffn), "fused [gate; value] projection"),
        ("ffn_swiglu_hidden", (1, s, ffn), "silu(gate) * value hidden before fc2"),
    ]
    rows: list[dict[str, Any]] = []
    for name, shape, note in specs:
        expected = _shape_bytes(shape)
        rows.append(
            {
                "boundary": name,
                "shape": list(shape),
                "dtype_assumption": "bf16 activation (2 bytes)",
                "expected_bytes": expected,
                "observed_mtlbuffer_matches": _buffer_matches(buffers, expected),
                "note": note,
            }
        )
    return rows


def _matmul_shape_summary(
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    s = int(sequence_meta["sequence_length"])
    h = int(cfg.hidden_size)
    inner = int(cfg.inner_dim)
    ffn = int(cfg.ffn_hidden_size)

    def qmm_row(name: str, m: int, k: int, n: int, segment: str) -> dict[str, Any]:
        measured = segment_stats.get(segment, {}).get("median_seconds")
        flops = 2.0 * float(m) * float(k) * float(n)
        return {
            "op": name,
            "segment": segment,
            "M": int(m),
            "K": int(k),
            "N": int(n),
            "observed_kernel_family": "affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0",
            "weight_bits": 4,
            "group_size": 64,
            "estimated_output_bytes_bf16": int(m * n * 2),
            "quant_weight_packed_bytes_lower_bound": int(k * n // 2),
            "estimated_flops_if_dequantized_gemm": flops,
            "segment_median_seconds": measured,
            "estimated_tflop_per_second_at_segment_median": (
                flops / float(measured) / 1e12 if measured else None
            ),
        }

    rows = [
        qmm_row("qkv_proj", s, h, 3 * inner, "qkv_quantized_matmul"),
        qmm_row("attn_out_proj", s, inner, h, "out_projection"),
        {
            **qmm_row("ffn_fc1_plus_fc2_combined", s, h, 2 * ffn, "fc1_swiglu_fc2"),
            "second_matmul": {"M": s, "K": ffn, "N": h},
        },
    ]
    # Correct the combined FFN FLOP and packed-weight estimates to include fc2 as well as fc1.
    ffn_flops = 2.0 * s * (h * 2 * ffn + ffn * h)
    ffn_packed = h * 2 * ffn // 2 + ffn * h // 2
    ffn_measured = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    rows[-1]["estimated_flops_if_dequantized_gemm"] = ffn_flops
    rows[-1]["quant_weight_packed_bytes_lower_bound"] = int(ffn_packed)
    rows[-1]["estimated_tflop_per_second_at_segment_median"] = (
        ffn_flops / float(ffn_measured) / 1e12 if ffn_measured else None
    )
    return rows


def _scan_capture(
    path: Path,
    *,
    cfg: DiTConfig | None = None,
    sequence_meta: dict[str, Any] | None = None,
    segment_stats: dict[str, dict[str, Any]] | None = None,
    max_bytes_per_file: int = 2_000_000,
) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "token_counts": {}}
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    counts = {token: 0 for token in TRACE_TOKENS}
    scanned_files = 0
    scanned_bytes = 0
    for file in files[:256]:
        try:
            data = file.read_bytes()[:max_bytes_per_file].lower()
        except OSError:
            continue
        scanned_files += 1
        scanned_bytes += len(data)
        for token in TRACE_TOKENS:
            counts[token] += data.count(token.encode())
    resources = _extract_resource_functions(path)
    buffers = _buffer_sizes(path)
    boundary_rows = (
        _expected_materialization_boundaries(cfg, sequence_meta, buffers)
        if cfg is not None and sequence_meta is not None
        else []
    )
    matmul_rows = (
        _matmul_shape_summary(cfg, sequence_meta, segment_stats or {})
        if cfg is not None and sequence_meta is not None
        else []
    )
    active_function_names = {row["name"] for row in resources["active"]}
    return {
        "exists": True,
        "is_dir": path.is_dir(),
        "path_size_bytes": _path_size(path),
        "scanned_files": scanned_files,
        "scanned_bytes": scanned_bytes,
        "token_counts": counts,
        "nonzero_tokens": {k: v for k, v in counts.items() if v},
        "active_device_resource_functions": resources["active"],
        "unused_device_resource_functions": resources["unused"],
        "active_function_counts_by_role": resources["counts_by_role"],
        "observed_mtlbuffer_count": len(buffers),
        "largest_mtlbuffers": [
            {"name": name, "size_bytes": size}
            for name, size in sorted(buffers.items(), key=lambda item: item[1], reverse=True)[:12]
        ],
        "materialization_boundaries": boundary_rows,
        "quantized_matmul_shape_summary": matmul_rows,
        "kernel_boundary_conclusion": {
            "trace_identifies_4bit_qmm_kernel": "affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0" in active_function_names,
            "trace_identifies_separate_rmsnorm_or_layout_kernels": any(
                row["role"] in {"rmsnorm_reduction", "pointwise_or_layout_materialization"}
                for row in resources["active"]
            ),
            "trace_identifies_separate_gather_kernel": any(
                row["role"] == "indexed_materialization_gather" for row in resources["active"]
            ),
            "interpretation": (
                "The capture package exposes MLX device-resource functions and MTLBuffer sizes, "
                "not just token counts: the real block uses MLX's 4-bit affine_qmm dequant+GEMM "
                "kernel family for the large qkv/ffn/out projections, while RMSNorm/gather/copy/"
                "pointwise kernels materialize smaller boundaries around those matmuls."
            ),
        },
        "note": "Token counts are retained as a weak sanity check; function names, buffer-size matches, and shape summaries drive the boundary interpretation.",
    }


def _capture_full_block(
    capture_path: Path,
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    *,
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    if capture_path.exists():
        capture_path = capture_path.with_name(
            f"{capture_path.stem}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}{capture_path.suffix}"
        )
    started = time.perf_counter()
    try:
        mx.metal.start_capture(str(capture_path))
        out = block(x, modulation, adaln_indices, rotary)
        mx.eval(out)
        mx.synchronize()
        mx.metal.stop_capture()
    except Exception as exc:  # capture support is environment-dependent; report, do not fake it.
        try:
            mx.metal.stop_capture()
        except Exception:
            pass
        return {
            "ok": False,
            "path": str(capture_path),
            "seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "ok": True,
        "path": str(capture_path),
        "seconds": time.perf_counter() - started,
        "scope": "one uninstrumented full TransformerBlock call with one mx.eval at the output",
        **_scan_capture(capture_path, cfg=cfg, sequence_meta=sequence_meta, segment_stats=segment_stats),
    }


def _rank_segments(segment_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    total = sum(float(s.get("median_seconds") or 0.0) for s in segment_stats.values())
    for label, stats in segment_stats.items():
        median = float(stats.get("median_seconds") or 0.0)
        rows.append(
            {
                "segment": label,
                "median_seconds": median,
                "p90_seconds": stats.get("p90_seconds"),
                "share_of_segmented_sum": median / total if total else None,
            }
        )
    return sorted(rows, key=lambda row: row["median_seconds"], reverse=True)


def _candidate_ffn_mx_split(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe the dominant FFN projection boundary with ``mx.split`` gate/value views."""

    original_flag = bool(getattr(block.mlp, "use_mx_split_swiglu_candidate", False))

    def baseline_forward() -> mx.array:
        block.mlp.use_mx_split_swiglu_candidate = False
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        block.mlp.use_mx_split_swiglu_candidate = True
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 2603,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        block.mlp.use_mx_split_swiglu_candidate = original_flag

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    if not parity_ok or not first_parity_ok:
        decision = "reject_parity"
        reason = "strict-equivalence check failed for the mx.split FFN boundary candidate"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says mx.split at the FFN boundary is slower than slice baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        reason = (
            "interleaved timing and bootstrap CI do not separate the mx.split FFN boundary candidate "
            "from measurement noise; no fixed percentage cutoff was used"
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "mx.split FFN boundary candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "mx.split FFN boundary candidate is strictly equivalent, disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    return {
        "name": "ffn_mx_split_swiglu",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "fused fc1 [gate; value] output before SwiGLU and fc2",
        "selection_rationale": (
            "The certified warm 4-bit block evidence ranks fc1_swiglu_fc2 as the dominant segment "
            f"(sequence length {sequence_meta.get('sequence_length')}, hidden {sequence_meta.get('hidden_size')}, "
            "52.6% of the segmented median in the referenced run).  The trace materializes the "
            "ffn_fc1_gate_value and ffn_swiglu_hidden buffers around MLX affine_qmm, so this probe "
            "changes only the gate/value split boundary from two slices to equal mx.split views."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "parity_ok": parity_ok and first_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default branch in FeedForward; no weight, quantization, sigma/NFE, or LoRA path change",
            "compile_cost": "no mx.compile or persistent compiler cache is introduced by this candidate",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; larger S/M/N must be remeasured",
            "strict_equivalence": "checked against both the pre-candidate baseline and the last interleaved baseline output",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, Amdahl contribution, complexity, memory, and strict parity; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_ffn_fused_swiglu_metal(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
    candidate_name: str = "ffn_fused_swiglu_metal",
) -> dict[str, Any]:
    """Probe the FFN SwiGLU activation/multiply boundary with one custom Metal kernel."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    block_flag_names = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    attention_flag_names = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_out_layout_metal_candidate",
    )
    original_ffn_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_block_flags = {name: bool(getattr(block, name, False)) for name in block_flag_names}
    original_attention_flags = {name: bool(getattr(block.attn, name, False)) for name in attention_flag_names}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: force off every other block, Attention, and FFN candidate.
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        for name in block_flag_names:
            setattr(block, name, False)
        for name in attention_flag_names:
            setattr(block.attn, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_tile_size
        block.mlp.use_ffn_metal_swiglu_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        fused = block.mlp._fc1_project(fc1_input)
        gate, value = fused[..., : block.mlp._ffn], fused[..., block.mlp._ffn :]
        baseline_hidden = nn.silu(gate) * value
        set_candidate(True)
        candidate_hidden = block.mlp._swiglu_hidden(fused)
        mx.eval(fc1_input, fused, baseline_hidden, candidate_hidden)
        mx.synchronize()
        activation_parity = _diff_stats(baseline_hidden, candidate_hidden)
        activation_shape_dtype = {
            "fused_shape": list(fused.shape),
            "fused_dtype": str(fused.dtype),
            "baseline_hidden_shape": list(baseline_hidden.shape),
            "candidate_hidden_shape": list(candidate_hidden.shape),
            "baseline_hidden_dtype": str(baseline_hidden.dtype),
            "candidate_hidden_dtype": str(candidate_hidden.dtype),
            "shape_matches_baseline": bool(baseline_hidden.shape == candidate_hidden.shape),
            "dtype_matches_baseline": bool(baseline_hidden.dtype == candidate_hidden.dtype),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 9011,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    except Exception as exc:
        after = _metrics()
        return {
            "name": candidate_name,
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "SwiGLU activation/multiply after fused fc1 [gate; value] projection",
            "opt_in_only": True,
            "disabled_by_default": True,
            "default_behavior_unchanged": True,
            "production_integrated": True,
            "strict_exact_semantics": False,
            "candidate_available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics_before": before,
            "metrics_after": after,
            "metrics_delta": _delta(before, after),
            "promote": False,
            "decision": "blocked_unsupported",
            "decision_reason": "custom Metal SwiGLU kernel could not be constructed or executed locally",
        }
    finally:
        for name, value in original_ffn_flags.items():
            setattr(block.mlp, name, value)
        for name, value in original_block_flags.items():
            setattr(block, name, value)
        for name, value in original_attention_flags.items():
            setattr(block.attn, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_tile_size

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    activation_parity_ok = (
        activation_parity["max_abs"] <= args.parity_atol
        and activation_parity["rel_l2"] <= args.parity_rel_l2
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(activation_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = (
            "Metal SwiGLU multiply repair changed activation or full-block outputs beyond configured parity bounds; "
            "the opt-in path remains disabled and is not promoted"
        )
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the fused Metal SwiGLU candidate is slower than baseline"
        promoted = False
    elif not stable_faster:
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        decision = "reject_unproven_noise"
        reason = (
            "interleaved timing and bootstrap CI do not separate the fused Metal SwiGLU candidate "
            "from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "fused Metal SwiGLU candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "fused Metal SwiGLU candidate satisfies parity bounds, is disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    return {
        "name": candidate_name,
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "SwiGLU activation/multiply after fused fc1 [gate; value] projection and before fc2",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior FFN probes changed gate/value splitting and "
            "projection rank; this repaired single-variable candidate leaves both QMM projections, slicing policy, "
            "and native nn.silu math alone and replaces only the final activated-gate/value multiply with one custom "
            "MLX/Metal kernel."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "parity_bounded_not_assumed_exact": False,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {
                **{name: False for name in ffn_flag_names},
                **{name: False for name in block_flag_names},
                **{name: False for name in attention_flag_names},
            },
            "enabled_flag_for_this_run": "use_ffn_metal_swiglu_candidate",
            "single_variable_guard": "all other block, Attention, and FFN candidate flags are forced off during this probe",
            "lora_path": "falls back to the existing MLX nn.silu path whenever lora is not None",
            "kernel": "native nn.silu(gate) is preserved for parity; mx.fast.metal_kernel multiplies activated_gate by value",
        },
        "activation_shape_dtype_contract": activation_shape_dtype,
        "activation_parity": activation_parity,
        "activation_parity_ok": activation_parity_ok,
        "strict_activation_parity_zero": bool(activation_parity["max_abs"] == 0.0 and activation_parity["rel_l2"] == 0.0),
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": bool(
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward flag plus a cached custom Metal multiply helper; no weights, quantization, sigma/NFE, cache, or projection-rank change",
            "compile_cost": "first candidate call records custom Metal JIT/setup cost; warm interleaved samples measure the cached kernel path",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other shapes and dtypes must be remeasured",
            "strict_equivalence": "native nn.silu is preserved and activation/full-block parity are explicitly measured before any timing promotion decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, Amdahl contribution, memory, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _ffn_projection_input(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
) -> mx.array:
    """Return the exact rank-3 input tensor consumed by FeedForward.fc1."""
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, _gate_mlp = modulation
    h = block.norm1(x) * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
    x_attn = x + gate_msa[adaln_indices] * block.attn(h, rotary)
    return block.norm2(x_attn) * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]


def _candidate_ffn_projection_2d_qmm(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe FeedForward fc1/fc2 by flattening rank-3 inputs before quantized QMM."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    original_projection_flag = bool(getattr(block.mlp, "use_ffn_2d_projection_candidate", False))
    original_split_flag = bool(getattr(block.mlp, "use_mx_split_swiglu_candidate", False))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine it with the prior mx.split candidate.
        block.mlp.use_mx_split_swiglu_candidate = False
        block.mlp.use_ffn_2d_projection_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        baseline_fc1 = block.mlp._fc1_project(fc1_input)
        gate, value = baseline_fc1[..., : block.mlp._ffn], baseline_fc1[..., block.mlp._ffn :]
        fc2_input = nn.silu(gate) * value
        baseline_fc2 = block.mlp._fc2_project(fc2_input)
        set_candidate(True)
        candidate_fc1 = block.mlp._fc1_project(fc1_input)
        candidate_fc2 = block.mlp._fc2_project(fc2_input)
        mx.eval(fc1_input, fc2_input, baseline_fc1, baseline_fc2, candidate_fc1, candidate_fc2)
        mx.synchronize()
        projection_parity = {
            "fc1": _diff_stats(baseline_fc1, candidate_fc1),
            "fc2": _diff_stats(baseline_fc2, candidate_fc2),
        }
        projection_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "fc1_flattened_shape": [int(fc1_input.shape[0]) * int(fc1_input.shape[1]), int(fc1_input.shape[2])],
            "fc1_baseline_output_shape": list(baseline_fc1.shape),
            "fc1_candidate_output_shape": list(candidate_fc1.shape),
            "fc1_baseline_output_dtype": str(baseline_fc1.dtype),
            "fc1_candidate_output_dtype": str(candidate_fc1.dtype),
            "fc2_input_shape": list(fc2_input.shape),
            "fc2_input_dtype": str(fc2_input.dtype),
            "fc2_flattened_shape": [int(fc2_input.shape[0]) * int(fc2_input.shape[1]), int(fc2_input.shape[2])],
            "fc2_baseline_output_shape": list(baseline_fc2.shape),
            "fc2_candidate_output_shape": list(candidate_fc2.shape),
            "fc2_baseline_output_dtype": str(baseline_fc2.dtype),
            "fc2_candidate_output_dtype": str(candidate_fc2.dtype),
            "restores_original_leading_shape": bool(
                tuple(baseline_fc1.shape[:-1])
                == tuple(candidate_fc1.shape[:-1])
                == tuple(fc1_input.shape[:-1])
                and tuple(baseline_fc2.shape[:-1])
                == tuple(candidate_fc2.shape[:-1])
                == tuple(fc2_input.shape[:-1])
            ),
            "dtypes_match_baseline": bool(
                baseline_fc1.dtype == candidate_fc1.dtype and baseline_fc2.dtype == candidate_fc2.dtype
            ),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 6829,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        block.mlp.use_ffn_2d_projection_candidate = original_projection_flag
        block.mlp.use_mx_split_swiglu_candidate = original_split_flag

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "strict-equivalence check failed for the FFN fc1/fc2 2D projection QMM candidate"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the FFN fc1/fc2 2D projection candidate is slower than rank-3 baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the FFN fc1/fc2 2D projection candidate "
            "from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "FFN fc1/fc2 2D projection candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "FFN fc1/fc2 2D projection candidate is strictly equivalent, disabled by default, memory-clean, "
            "and faster than interleaved rank-3 baseline outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    return {
        "name": "ffn_fc1_fc2_2d_qmm",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "rank-3 [B,S,H] fc1 input and rank-3 [B,S,F] fc2 input flattened to [B*S,*] before quantized nn.Linear, then reshaped back",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. This is the dominant FFN QMM/GEMM-heavy boundary; "
            "unlike the prior mx.split route, this probe changes only the dispatch rank presented to both "
            "FeedForward.fc1 and FeedForward.fc2 while preserving SwiGLU slicing and all weights."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {
                "use_mx_split_swiglu_candidate": False,
                "use_ffn_2d_projection_candidate": False,
            },
            "enabled_flag_for_this_run": "use_ffn_2d_projection_candidate",
            "helper": "linear_rank3_input_as_rank2",
            "single_variable_guard": "use_mx_split_swiglu_candidate is forced off during this probe",
            "lora_path": "falls back to existing rank-3 base projections whenever lora is not None; LoRA deltas keep their existing path",
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one helper reuse plus disabled-by-default FeedForward flag and two local projection wrappers; no weights, quantization, sigma/NFE, cache, or default generation path changes",
            "compile_cost": "no mx.compile or persistent compiler cache is introduced by this candidate; first candidate call records ordinary MLX lazy/kernel setup cost",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; larger S/M/N must be remeasured before release-profile promotion",
            "maintainability": "simple local rank-layout switch, but it affects both FFN projections and is intentionally not combined with the prior mx.split route",
            "strict_equivalence": "fc1 projection, fc2 projection, and full block output are checked against rank-3 baselines",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, Amdahl contribution, memory, maintainability, resolution scaling, and strict parity; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_ffn_fc1_rank2_qmm(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe only FeedForward ``fc1`` by flattening its rank-3 input before QMM."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    original_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: force off prior FFN split, full-rank2, fc2-rank2,
        # Metal, chunking, and explicit-contiguity candidates. Only fc1 sees rank-2 scheduling.
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.use_ffn_fc1_rank2_qmm_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        baseline_fc1 = block.mlp._fc1_project(fc1_input)
        baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=fc1_input)
        baseline_fc2 = block.mlp._fc2_project(baseline_hidden)
        set_candidate(True)
        candidate_fc1 = block.mlp._fc1_project(fc1_input)
        candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=fc1_input)
        candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
        mx.eval(
            fc1_input,
            baseline_fc1,
            candidate_fc1,
            baseline_hidden,
            candidate_hidden,
            baseline_fc2,
            candidate_fc2,
        )
        mx.synchronize()
        projection_parity = {
            "fc1": _diff_stats(baseline_fc1, candidate_fc1),
            "swiglu_hidden": _diff_stats(baseline_hidden, candidate_hidden),
            "fc2_unchanged": _diff_stats(baseline_fc2, candidate_fc2),
        }
        projection_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "fc1_flattened_shape": [
                int(fc1_input.shape[0]) * int(fc1_input.shape[1]),
                int(fc1_input.shape[2]),
            ],
            "fc1_baseline_output_shape": list(baseline_fc1.shape),
            "fc1_candidate_output_shape": list(candidate_fc1.shape),
            "fc1_baseline_output_dtype": str(baseline_fc1.dtype),
            "fc1_candidate_output_dtype": str(candidate_fc1.dtype),
            "hidden_baseline_shape": list(baseline_hidden.shape),
            "hidden_candidate_shape": list(candidate_hidden.shape),
            "hidden_baseline_dtype": str(baseline_hidden.dtype),
            "hidden_candidate_dtype": str(candidate_hidden.dtype),
            "fc2_baseline_output_shape": list(baseline_fc2.shape),
            "fc2_candidate_output_shape": list(candidate_fc2.shape),
            "fc2_baseline_output_dtype": str(baseline_fc2.dtype),
            "fc2_candidate_output_dtype": str(candidate_fc2.dtype),
            "fc1_restores_original_leading_shape": bool(
                tuple(baseline_fc1.shape[:-1]) == tuple(candidate_fc1.shape[:-1]) == tuple(fc1_input.shape[:-1])
            ),
            "hidden_shape_dtype_matches_baseline": bool(
                baseline_hidden.shape == candidate_hidden.shape and baseline_hidden.dtype == candidate_hidden.dtype
            ),
            "fc2_shape_dtype_unchanged_by_candidate": bool(
                baseline_fc2.shape == candidate_fc2.shape and baseline_fc2.dtype == candidate_fc2.dtype
            ),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 7867,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        for name, value in original_flags.items():
            setattr(block.mlp, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "strict-equivalence check failed for the FFN fc1-only rank-2 QMM candidate"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the FFN fc1-only rank-2 QMM candidate is slower than rank-3 fc1 baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the FFN fc1-only rank-2 QMM candidate "
            "from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "FFN fc1-only rank-2 QMM candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "FFN fc1-only rank-2 QMM candidate is strictly equivalent, disabled by default, memory-clean, "
            "and faster than interleaved rank-3 fc1 baseline outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    return {
        "name": "ffn_fc1_rank2_qmm",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "rank-3 [B,S,H] FFN input to FeedForward.fc1 flattened to [B*S,H] before quantized nn.Linear, then reshaped back to [B,S,2F]",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. This candidate isolates the previously-confounded full "
            "FFN fc1/fc2 rank-2 route: it leaves fc2 projection rank, SwiGLU slicing, weights, and LoRA fallback "
            "unchanged while testing whether only the fc1 QMM dispatch prefers explicit rank-2 scheduling."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {name: False for name in ffn_flag_names},
            "enabled_flag_for_this_run": "use_ffn_fc1_rank2_qmm_candidate",
            "helper": "linear_rank3_input_as_rank2",
            "single_variable_guard": "prior FFN split, full fc1/fc2 rank-2, fc2-only rank-2, Metal SwiGLU, chunking, and explicit-contiguity candidates are forced off during this probe",
            "lora_path": "falls back to the existing rank-3 base fc1 projection whenever lora is not None; LoRA deltas keep their existing path",
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward flag reusing the rank-2 linear helper only at fc1; no weights, quantization, sigma/NFE, cache, or default generation path changes",
            "compile_cost": "no mx.compile, custom Metal kernel, or persistent compiler cache is introduced by this candidate; first candidate call records ordinary MLX lazy/kernel setup cost",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas; reshape scheduling may change transient materialization",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; larger S/M/N must be remeasured before release-profile promotion",
            "maintainability": "local fc1-only rank-layout switch with explicit LoRA fallback; prior broader FFN rank-2 route remains separate and is not combined here",
            "strict_equivalence": "fc1 projection, SwiGLU hidden, fc2 projection unchanged under rank-3 scheduling, and full block output are checked against rank-3 baselines",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, Amdahl contribution, memory, maintainability, resolution scaling, and strict parity; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_ffn_fc1_split_gate_value_quantized_qmm(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe ``fc1`` as two quantized output-row projections for gate and value."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    source_is_quantized = getattr(block.mlp.fc1, "scales", None) is not None
    quantized_matmul_available = getattr(mx, "quantized_matmul", None) is not None
    if not source_is_quantized and not args.tiny:
        return {
            "name": "ffn_fc1_split_gate_value_quantized_qmm",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc1 fused [gate; value] output rows split into two QMM projections",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_fc1_not_quantized",
            "decision_reason": "real block FeedForward.fc1 is not an MLX quantized linear with public scales; split quantized QMM was not attempted",
            "fc1_split_gate_value_qmm": block.mlp.fc1_split_gate_value_quantized_qmm_info(),
        }
    if source_is_quantized and not quantized_matmul_available:
        return {
            "name": "ffn_fc1_split_gate_value_quantized_qmm",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc1 fused [gate; value] output rows split into two QMM projections",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_quantized_matmul_unavailable",
            "decision_reason": "mx.quantized_matmul is unavailable, so quantized fc1 output-row slices cannot be projected safely",
            "fc1_split_gate_value_qmm": block.mlp.fc1_split_gate_value_quantized_qmm_info(),
        }

    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc1_split_gate_value_quantized_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
        "use_ffn_subgraph_compile_candidate",
    )
    original_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_fc1_tile_size = int(getattr(block.mlp, "ffn_fc1_tiled_output_channels", 2048))
    original_fc2_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: all prior FFN projection, SwiGLU, dense-dequant,
        # chunking, contiguity, and compile candidates are off. Only fc1 output-row splitting toggles.
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = original_fc1_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.mlp.use_ffn_fc1_split_gate_value_quantized_qmm_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        baseline_fc1 = block.mlp._fc1_project(fc1_input)
        baseline_gate = baseline_fc1[..., : block.mlp._ffn]
        baseline_value = baseline_fc1[..., block.mlp._ffn :]
        baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=fc1_input)
        baseline_fc2 = block.mlp._fc2_project(baseline_hidden)

        set_candidate(True)
        qmm_before_projection = block.mlp.fc1_split_gate_value_quantized_qmm_info()
        split_started = time.perf_counter()
        candidate_gate, candidate_value = block.mlp._fc1_gate_value_from_prepared_input(fc1_input)
        candidate_hidden = nn.silu(candidate_gate) * candidate_value
        candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
        candidate_fc1_reassembled = block.mlp._fc1_project(fc1_input)
        candidate_hidden_from_reassembled = block.mlp._swiglu_hidden(candidate_fc1_reassembled, x=fc1_input)
        fallback_fc1 = block.mlp._fc1_project(fc1_input, lora=object())
        mx.eval(
            fc1_input,
            baseline_fc1,
            baseline_gate,
            baseline_value,
            baseline_hidden,
            baseline_fc2,
            candidate_gate,
            candidate_value,
            candidate_hidden,
            candidate_fc2,
            candidate_fc1_reassembled,
            candidate_hidden_from_reassembled,
            fallback_fc1,
        )
        mx.synchronize()
        first_split_fc1_seconds = time.perf_counter() - split_started
        qmm_after_projection = block.mlp.fc1_split_gate_value_quantized_qmm_info()
        projection_parity = {
            "gate_rows": _diff_stats(baseline_gate, candidate_gate),
            "value_rows": _diff_stats(baseline_value, candidate_value),
            "fc1_reassembled": _diff_stats(baseline_fc1, candidate_fc1_reassembled),
            "swiglu_hidden_direct": _diff_stats(baseline_hidden, candidate_hidden),
            "swiglu_hidden_reassembled": _diff_stats(baseline_hidden, candidate_hidden_from_reassembled),
            "fc2_unchanged": _diff_stats(baseline_fc2, candidate_fc2),
            "lora_fallback_fused_fc1": _diff_stats(baseline_fc1, fallback_fc1),
        }
        projection_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "baseline_fused_fc1_shape": list(baseline_fc1.shape),
            "candidate_gate_shape": list(candidate_gate.shape),
            "candidate_value_shape": list(candidate_value.shape),
            "candidate_reassembled_fc1_shape": list(candidate_fc1_reassembled.shape),
            "baseline_fused_fc1_dtype": str(baseline_fc1.dtype),
            "candidate_gate_dtype": str(candidate_gate.dtype),
            "candidate_value_dtype": str(candidate_value.dtype),
            "candidate_reassembled_fc1_dtype": str(candidate_fc1_reassembled.dtype),
            "hidden_baseline_shape": list(baseline_hidden.shape),
            "hidden_candidate_shape": list(candidate_hidden.shape),
            "hidden_baseline_dtype": str(baseline_hidden.dtype),
            "hidden_candidate_dtype": str(candidate_hidden.dtype),
            "fc2_baseline_output_shape": list(baseline_fc2.shape),
            "fc2_candidate_output_shape": list(candidate_fc2.shape),
            "fc2_baseline_output_dtype": str(baseline_fc2.dtype),
            "fc2_candidate_output_dtype": str(candidate_fc2.dtype),
            "gate_value_shapes_match_fused_halves": bool(
                candidate_gate.shape == baseline_gate.shape and candidate_value.shape == baseline_value.shape
            ),
            "gate_value_dtypes_match_fused_halves": bool(
                candidate_gate.dtype == baseline_gate.dtype and candidate_value.dtype == baseline_value.dtype
            ),
            "reassembled_shape_dtype_matches_baseline": bool(
                candidate_fc1_reassembled.shape == baseline_fc1.shape
                and candidate_fc1_reassembled.dtype == baseline_fc1.dtype
            ),
            "fc2_shape_dtype_unchanged_by_candidate": bool(
                baseline_fc2.shape == candidate_fc2.shape and baseline_fc2.dtype == candidate_fc2.dtype
            ),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 18719,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        qmm_after_interleaved = block.mlp.fc1_split_gate_value_quantized_qmm_info()
    finally:
        for name, value in original_flags.items():
            setattr(block.mlp, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = original_fc1_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "split gate/value quantized fc1 output-row QMM changed direct FFN or full-block outputs beyond configured parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says split gate/value quantized fc1 output-row QMM is slower than fused fc1 baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate split gate/value quantized fc1 output-row QMM "
            "from fused fc1 baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "split gate/value quantized fc1 output-row QMM is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "split gate/value quantized fc1 output-row QMM is strictly equivalent, disabled by default, "
            "memory-clean, and faster than fused fc1 baseline outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    return {
        "name": "ffn_fc1_split_gate_value_quantized_qmm",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "FeedForward.fc1 fused [gate; value] output rows are projected as two separate gate/value QMMs before the same SwiGLU and existing fc2",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior FFN candidates changed split syntax, rank layout, "
            "dense dequantization, pre-contiguity, chunking, or mx.compile scheduling. This probe changes only "
            "the fc1 output-row boundary: it issues one QMM for [gate] rows and one QMM for [value] rows, then "
            "runs the same native SwiGLU and unchanged fc2."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {name: False for name in ffn_flag_names},
            "enabled_flag_for_this_run": "use_ffn_fc1_split_gate_value_quantized_qmm_candidate",
            "helper": "linear_output_row_slice_projection + FeedForward._fc1_gate_value_from_prepared_input",
            "single_variable_guard": "prior FFN projection-rank, dense-dequant, SwiGLU Metal/split, chunking, contiguity, and subgraph-compile candidates are forced off during this probe",
            "lora_path": "falls back to the existing fused fc1 base projection whenever lora is not None; LoRA gate/value deltas keep the existing _swiglu_hidden path",
            "forward_materialization": "candidate forward does not concatenate [gate; value] before SwiGLU; direct projection parity reassembles only for checking",
        },
        "fc1_split_gate_value_qmm": {
            "source_is_quantized": source_is_quantized,
            "quantized_matmul_available": quantized_matmul_available,
            "qmm_before_projection": qmm_before_projection,
            "qmm_after_projection": qmm_after_projection,
            "qmm_after_interleaved": qmm_after_interleaved,
            "first_split_fc1_seconds_includes_two_projection_launches": first_split_fc1_seconds,
            "row_order_contract": "gate rows are fc1[0:F] and value rows are fc1[F:2F], matching the raw MiniMax-H3 fused [gate; value] SwiGLU layout",
            "dense_dequantization": False,
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "memory_gate_components": {
            "pageouts_delta": pageouts_delta,
            "swapouts_delta": swapouts_delta,
            "mlx_cache_bytes_delta": metrics_delta.get("mlx_cache_bytes"),
            "mlx_peak_bytes_delta": metrics_delta.get("mlx_peak_bytes"),
            "rss_kib_delta": metrics_delta.get("current_rss_kib"),
        },
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward flag plus a small quantized output-row projection helper; no default generation path, sigma/NFE, cache, or weight changes",
            "compile_cost": "no mx.compile or custom Metal kernel is introduced; first candidate call records ordinary MLX lazy/kernel setup plus two fc1 QMM launches",
            "memory": "candidate avoids constructing the fused [B,S,2F] fc1 tensor in the forward path but creates separate gate/value outputs; artifact records RSS/MLX deltas, pageouts, and swapouts",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence and one real 4-bit resident block; larger shapes must be remeasured before promotion",
            "maintainability": "local fc1-only output-row split with explicit LoRA fallback; prior FFN candidates are forced off and not combined",
            "strict_equivalence": "gate rows, value rows, reassembled fc1, SwiGLU hidden, fc2 output, LoRA fallback base projection, and full block output are checked before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_ffn_fc2_rank2_qmm(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe only FeedForward ``fc2`` by flattening its rank-3 hidden input before QMM."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    original_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: force off prior FFN split, full-rank2, fc1-rank2, Metal, chunking,
        # and explicit-contiguity candidates. Only the fc2 projection sees the rank-2 schedule.
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.use_ffn_fc2_rank2_qmm_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        baseline_fc1 = block.mlp._fc1_project(fc1_input)
        baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=fc1_input)
        baseline_fc2 = block.mlp._fc2_project(baseline_hidden)
        set_candidate(True)
        candidate_fc1 = block.mlp._fc1_project(fc1_input)
        candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=fc1_input)
        candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
        mx.eval(
            fc1_input,
            baseline_fc1,
            candidate_fc1,
            baseline_hidden,
            candidate_hidden,
            baseline_fc2,
            candidate_fc2,
        )
        mx.synchronize()
        projection_parity = {
            "fc1_unchanged": _diff_stats(baseline_fc1, candidate_fc1),
            "swiglu_hidden_unchanged": _diff_stats(baseline_hidden, candidate_hidden),
            "fc2": _diff_stats(baseline_fc2, candidate_fc2),
        }
        projection_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "fc1_baseline_output_shape": list(baseline_fc1.shape),
            "fc1_candidate_output_shape": list(candidate_fc1.shape),
            "fc1_baseline_output_dtype": str(baseline_fc1.dtype),
            "fc1_candidate_output_dtype": str(candidate_fc1.dtype),
            "fc2_input_shape": list(baseline_hidden.shape),
            "fc2_input_dtype": str(baseline_hidden.dtype),
            "fc2_flattened_shape": [
                int(baseline_hidden.shape[0]) * int(baseline_hidden.shape[1]),
                int(baseline_hidden.shape[2]),
            ],
            "fc2_baseline_output_shape": list(baseline_fc2.shape),
            "fc2_candidate_output_shape": list(candidate_fc2.shape),
            "fc2_baseline_output_dtype": str(baseline_fc2.dtype),
            "fc2_candidate_output_dtype": str(candidate_fc2.dtype),
            "fc1_shape_dtype_unchanged_by_candidate": bool(
                baseline_fc1.shape == candidate_fc1.shape and baseline_fc1.dtype == candidate_fc1.dtype
            ),
            "fc2_restores_original_leading_shape": bool(
                tuple(baseline_fc2.shape[:-1]) == tuple(candidate_fc2.shape[:-1]) == tuple(baseline_hidden.shape[:-1])
            ),
            "fc2_dtype_matches_baseline": bool(baseline_fc2.dtype == candidate_fc2.dtype),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 7841,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        for name, value in original_flags.items():
            setattr(block.mlp, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "strict-equivalence check failed for the FFN fc2-only rank-2 QMM candidate"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the FFN fc2-only rank-2 QMM candidate is slower than rank-3 fc2 baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the FFN fc2-only rank-2 QMM candidate "
            "from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "FFN fc2-only rank-2 QMM candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "FFN fc2-only rank-2 QMM candidate is strictly equivalent, disabled by default, memory-clean, "
            "and faster than interleaved rank-3 fc2 baseline outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    return {
        "name": "ffn_fc2_rank2_qmm",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "rank-3 [B,S,F] SwiGLU hidden input to FeedForward.fc2 flattened to [B*S,F] before quantized nn.Linear, then reshaped back to [B,S,H]",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. This candidate is narrower than the prior full "
            "FFN fc1/fc2 rank-2 route: it leaves fc1 projection rank, SwiGLU slicing, weights, and LoRA fallback "
            "unchanged while testing whether only the fc2 QMM dispatch prefers explicit rank-2 scheduling."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {name: False for name in ffn_flag_names},
            "enabled_flag_for_this_run": "use_ffn_fc2_rank2_qmm_candidate",
            "helper": "linear_rank3_input_as_rank2",
            "single_variable_guard": "prior FFN split, full fc1/fc2 rank-2, Metal SwiGLU, chunking, and explicit-contiguity candidates are forced off during this probe",
            "lora_path": "falls back to the existing rank-3 base fc2 projection whenever lora is not None; LoRA deltas keep their existing path",
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward flag reusing the rank-2 linear helper only at fc2; no weights, quantization, sigma/NFE, cache, or default generation path changes",
            "compile_cost": "no mx.compile, custom Metal kernel, or persistent compiler cache is introduced by this candidate; first candidate call records ordinary MLX lazy/kernel setup cost",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas; reshape scheduling may change transient materialization",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; larger S/M/N must be remeasured before release-profile promotion",
            "maintainability": "local fc2-only rank-layout switch with explicit LoRA fallback; prior broader FFN rank-2 route remains separate and is not combined here",
            "strict_equivalence": "fc1 projection unchanged, SwiGLU hidden unchanged, fc2 projection, and full block output are checked against rank-3 baselines",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, Amdahl contribution, memory, maintainability, resolution scaling, and strict parity; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_ffn_fc1_dense_dequant(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe resident dense reconstruction of only ``FeedForward.fc1`` quantized weights."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    source_is_quantized = getattr(block.mlp.fc1, "scales", None) is not None
    dequantize_available = getattr(mx, "dequantize", None) is not None
    if not source_is_quantized and not args.tiny:
        return {
            "name": "ffn_fc1_dense_dequant",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc1 quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_fc1_reconstruction",
            "decision_reason": (
                "real block fc1 is not an MLX quantized linear with public scales/biases; "
                "dense reconstruction was not attempted"
            ),
            "fc1_cache_info": block.mlp.fc1_dense_dequant_cache_info(),
        }
    if source_is_quantized and not dequantize_available:
        return {
            "name": "ffn_fc1_dense_dequant",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc1 quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_fc1_reconstruction",
            "decision_reason": "mx.dequantize is unavailable, so quantized fc1 weights cannot be reconstructed safely",
            "fc1_cache_info": block.mlp.fc1_dense_dequant_cache_info(),
        }

    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    original_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: fc2, SwiGLU, projection rank, chunking,
        # tiling, Metal, and explicit-contiguity candidates are forced off. Only fc1
        # swaps from quantized QMM to the resident dense dequantized weight.
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_tile_size
        block.mlp.use_ffn_fc1_dense_dequant_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        block.mlp.clear_fc1_dense_dequant_cache()
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        baseline_fc1 = block.mlp._fc1_project(fc1_input)
        baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=fc1_input)
        baseline_fc2 = block.mlp._fc2_project(baseline_hidden)

        set_candidate(True)
        cache_before_projection = block.mlp.fc1_dense_dequant_cache_info()
        dense_projection_started = time.perf_counter()
        candidate_fc1 = block.mlp._fc1_project(fc1_input)
        candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=fc1_input)
        candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
        mx.eval(
            fc1_input,
            baseline_fc1,
            candidate_fc1,
            baseline_hidden,
            candidate_hidden,
            baseline_fc2,
            candidate_fc2,
        )
        mx.synchronize()
        first_dense_fc1_projection_seconds = time.perf_counter() - dense_projection_started
        cache_after_projection = block.mlp.fc1_dense_dequant_cache_info()
        projection_parity = {
            "fc1_dense_dequant": _diff_stats(baseline_fc1, candidate_fc1),
            "swiglu_hidden_after_fc1": _diff_stats(baseline_hidden, candidate_hidden),
            "fc2_after_fc1": _diff_stats(baseline_fc2, candidate_fc2),
        }
        projection_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "fc1_baseline_output_shape": list(baseline_fc1.shape),
            "fc1_candidate_output_shape": list(candidate_fc1.shape),
            "fc1_baseline_output_dtype": str(baseline_fc1.dtype),
            "fc1_candidate_output_dtype": str(candidate_fc1.dtype),
            "hidden_baseline_shape": list(baseline_hidden.shape),
            "hidden_candidate_shape": list(candidate_hidden.shape),
            "hidden_baseline_dtype": str(baseline_hidden.dtype),
            "hidden_candidate_dtype": str(candidate_hidden.dtype),
            "fc2_baseline_output_shape": list(baseline_fc2.shape),
            "fc2_candidate_output_shape": list(candidate_fc2.shape),
            "fc2_baseline_output_dtype": str(baseline_fc2.dtype),
            "fc2_candidate_output_dtype": str(candidate_fc2.dtype),
            "fc1_shape_matches_baseline": bool(baseline_fc1.shape == candidate_fc1.shape),
            "fc1_dtype_matches_baseline": bool(baseline_fc1.dtype == candidate_fc1.dtype),
            "hidden_shape_matches_baseline": bool(baseline_hidden.shape == candidate_hidden.shape),
            "hidden_dtype_matches_baseline": bool(baseline_hidden.dtype == candidate_hidden.dtype),
            "fc2_shape_matches_baseline": bool(baseline_fc2.shape == candidate_fc2.shape),
            "fc2_dtype_matches_baseline": bool(baseline_fc2.dtype == candidate_fc2.dtype),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_candidate_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 11813,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        cache_after_interleaved = block.mlp.fc1_dense_dequant_cache_info()
    finally:
        for name, value in original_flags.items():
            setattr(block.mlp, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_tile_size
        if not original_flags.get("use_ffn_fc1_dense_dequant_candidate", False):
            block.mlp.clear_fc1_dense_dequant_cache()

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "resident dense-dequantized fc1 changed projection or full-block outputs beyond configured parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says resident dense-dequantized fc1 is slower than baseline quantized fc1"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate resident dense-dequantized fc1 from baseline; "
            "no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "resident dense-dequantized fc1 is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "resident dense-dequantized fc1 is within parity bounds, disabled by default, memory-clean, "
            "and faster than baseline quantized fc1 outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    dense_resident_nbytes = int(cache_after_interleaved.get("dense_nbytes") or 0)
    packed_source_nbytes = int(cache_after_interleaved.get("source_weight_nbytes") or 0) + int(
        cache_after_interleaved.get("source_scales_nbytes") or 0
    ) + int(cache_after_interleaved.get("source_biases_nbytes") or 0)
    return {
        "name": "ffn_fc1_dense_dequant",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "FeedForward.fc1 fused gate/value projection uses a resident dense weight reconstructed from the quantized fc1 pack/scales/biases",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior fc2 dense probes changed the downstream projection; "
            "this single-variable probe changes only the larger fused fc1 weight representation after the real 4-bit "
            "block is resident, replacing repeated QMM dequant/GEMM dispatch with dense MLX matmul."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {name: False for name in ffn_flag_names},
            "enabled_flag_for_this_run": "use_ffn_fc1_dense_dequant_candidate",
            "helper": "FeedForward._fc1_dense_dequant_weight + dense_linear_projection",
            "single_variable_guard": "prior FFN split, projection-rank, resident/tiled dense fc2, Metal SwiGLU, sequence-chunk, and explicit-contiguity candidates are forced off during this probe",
            "lora_path": "falls back to the existing quantized fc1 base projection whenever lora is not None; LoRA fc1 deltas keep their existing path",
            "cache_lifecycle": "cache key follows the loaded fc1 weight/scales/biases arrays; QuantizedBlockProvider clears it whenever the reusable resident block slot is rebound",
        },
        "fc1_source_reconstruction": {
            "api": "mx.dequantize(weight, scales, biases, group_size, bits, mode, dtype=scales.dtype)",
            "source_is_quantized": source_is_quantized,
            "dequantize_available": dequantize_available,
            "cache_before_projection": cache_before_projection,
            "cache_after_projection": cache_after_projection,
            "cache_after_interleaved": cache_after_interleaved,
            "first_dense_fc1_projection_seconds_includes_materialization": first_dense_fc1_projection_seconds,
            "resident_dense_nbytes": dense_resident_nbytes,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "extra_resident_nbytes_vs_packed_source": dense_resident_nbytes - packed_source_nbytes,
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_candidate_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward flag plus resident dense fc1 cache; no default generation path, sigma/NFE, or non-fc1 weight changes",
            "compile_cost": "no mx.compile or custom Metal kernel is introduced; first dense fc1 projection records dequantization/materialization cost separately from warm interleaved timing",
            "memory": "candidate holds an extra dense fc1 weight while the quantized block is resident; artifact records dense bytes, packed-source bytes, RSS/MLX deltas, pageouts, and swapouts",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence and one resident block; larger shapes and all-block lifecycle need remeasurement before promotion",
            "maintainability": "local fc1-only switch with explicit LoRA fallback and provider cache invalidation on block-slot rebinding",
            "strict_equivalence": "fc1 projection, downstream SwiGLU hidden, downstream fc2 projection, and full block output are checked against the quantized baseline before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_ffn_fc1_tiled_dense_dequant(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe transient output-channel tiled dense reconstruction of ``FeedForward.fc1``."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    tile_size = int(args.ffn_fc1_tile_size)
    if tile_size <= 0:
        raise ValueError(f"--ffn-fc1-tile-size must be positive, got {tile_size}")

    source_is_quantized = getattr(block.mlp.fc1, "scales", None) is not None
    dequantize_available = getattr(mx, "dequantize", None) is not None
    if not source_is_quantized and not args.tiny:
        return {
            "name": "ffn_fc1_tiled_dense_dequant",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc1 output-channel tiled quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_fc1_reconstruction",
            "decision_reason": (
                "real block fc1 is not an MLX quantized linear with public scales/biases; "
                "tiled dense reconstruction was not attempted"
            ),
            "fc1_tiling": block.mlp.fc1_tiled_dense_dequant_info(tile_size),
        }
    if source_is_quantized and not dequantize_available:
        return {
            "name": "ffn_fc1_tiled_dense_dequant",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc1 output-channel tiled quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_fc1_reconstruction",
            "decision_reason": "mx.dequantize is unavailable, so quantized fc1 weight tiles cannot be reconstructed safely",
            "fc1_tiling": block.mlp.fc1_tiled_dense_dequant_info(tile_size),
        }

    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    original_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_fc1_tile_size = int(getattr(block.mlp, "ffn_fc1_tiled_output_channels", 512))
    original_fc2_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: resident dense fc1, fc2, SwiGLU, projection rank,
        # chunking, Metal, and explicit-contiguity candidates are forced off. Only fc1 uses
        # transient output-channel dense-dequant tiles.
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.mlp.use_ffn_fc1_tiled_dense_dequant_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        if not original_flags.get("use_ffn_fc1_dense_dequant_candidate", False):
            block.mlp.clear_fc1_dense_dequant_cache()
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        baseline_fc1 = block.mlp._fc1_project(fc1_input)
        baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=fc1_input)
        baseline_fc2 = block.mlp._fc2_project(baseline_hidden)

        set_candidate(True)
        tiling_before_projection = block.mlp.fc1_tiled_dense_dequant_info(tile_size)
        tiled_projection_started = time.perf_counter()
        candidate_fc1 = block.mlp._fc1_project(fc1_input)
        candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=fc1_input)
        candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
        mx.eval(
            fc1_input,
            baseline_fc1,
            candidate_fc1,
            baseline_hidden,
            candidate_hidden,
            baseline_fc2,
            candidate_fc2,
        )
        mx.synchronize()
        first_tiled_fc1_projection_seconds = time.perf_counter() - tiled_projection_started
        tiling_after_projection = block.mlp.fc1_tiled_dense_dequant_info(tile_size)
        projection_parity = {
            "fc1_tiled_dense_dequant": _diff_stats(baseline_fc1, candidate_fc1),
            "swiglu_hidden_after_fc1": _diff_stats(baseline_hidden, candidate_hidden),
            "fc2_after_fc1": _diff_stats(baseline_fc2, candidate_fc2),
        }
        projection_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "fc1_baseline_output_shape": list(baseline_fc1.shape),
            "fc1_candidate_output_shape": list(candidate_fc1.shape),
            "fc1_baseline_output_dtype": str(baseline_fc1.dtype),
            "fc1_candidate_output_dtype": str(candidate_fc1.dtype),
            "hidden_baseline_shape": list(baseline_hidden.shape),
            "hidden_candidate_shape": list(candidate_hidden.shape),
            "hidden_baseline_dtype": str(baseline_hidden.dtype),
            "hidden_candidate_dtype": str(candidate_hidden.dtype),
            "fc2_baseline_output_shape": list(baseline_fc2.shape),
            "fc2_candidate_output_shape": list(candidate_fc2.shape),
            "fc2_baseline_output_dtype": str(baseline_fc2.dtype),
            "fc2_candidate_output_dtype": str(candidate_fc2.dtype),
            "fc1_shape_matches_baseline": bool(baseline_fc1.shape == candidate_fc1.shape),
            "fc1_dtype_matches_baseline": bool(baseline_fc1.dtype == candidate_fc1.dtype),
            "hidden_shape_matches_baseline": bool(baseline_hidden.shape == candidate_hidden.shape),
            "hidden_dtype_matches_baseline": bool(baseline_hidden.dtype == candidate_hidden.dtype),
            "fc2_shape_matches_baseline": bool(baseline_fc2.shape == candidate_fc2.shape),
            "fc2_dtype_matches_baseline": bool(baseline_fc2.dtype == candidate_fc2.dtype),
            "fc1_output_tile_channels": tile_size,
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_candidate_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 15101,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        tiling_after_interleaved = block.mlp.fc1_tiled_dense_dequant_info(tile_size)
        resident_cache_after_interleaved = block.mlp.fc1_dense_dequant_cache_info()
    finally:
        for name, value in original_flags.items():
            setattr(block.mlp, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = original_fc1_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        if not original_flags.get("use_ffn_fc1_dense_dequant_candidate", False):
            block.mlp.clear_fc1_dense_dequant_cache()

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "transient tiled dense-dequantized fc1 changed projection or full-block outputs beyond configured parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says transient tiled dense-dequantized fc1 is slower than baseline quantized fc1"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate transient tiled dense-dequantized fc1 "
            "from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "transient tiled dense-dequantized fc1 is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "transient tiled dense-dequantized fc1 is within parity bounds, disabled by default, "
            "does not keep a full dense fc1 resident, is memory-clean, and is faster than baseline outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    max_tile_nbytes = int(tiling_after_interleaved.get("max_dense_tile_nbytes") or 0)
    full_dense_nbytes = int(tiling_after_interleaved.get("full_dense_nbytes_if_resident") or 0)
    packed_source_nbytes = int(tiling_after_interleaved.get("packed_quantized_source_nbytes") or 0)
    return {
        "name": "ffn_fc1_tiled_dense_dequant",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "FeedForward.fc1 fused gate/value projection transiently dequantizes output-channel tiles and runs dense matmul per tile",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. The prior resident dense fc1 probe was faster but "
            "rejected because the full dense fused gate/value matrix stayed live and caused pageouts. This "
            "single-variable probe changes only that fc1 memory boundary: it keeps fc2, SwiGLU, projection rank, "
            "weights, and LoRA fallback unchanged while testing whether per-output-channel dense tiles preserve "
            "enough GEMM benefit without persistent full-dense residency."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {name: False for name in ffn_flag_names},
            "enabled_flag_for_this_run": "use_ffn_fc1_tiled_dense_dequant_candidate",
            "helper": "tiled_dense_linear_projection + FeedForward.fc1_tiled_dense_dequant_info",
            "single_variable_guard": "prior FFN split, projection-rank, resident dense, fc2 tiled dense, Metal SwiGLU, sequence-chunk, and explicit-contiguity candidates are forced off during this probe",
            "lora_path": "falls back to the existing quantized fc1 base projection whenever lora is not None; LoRA fc1 deltas keep their existing path",
            "resident_full_dense_cache": False,
        },
        "fc1_tiling": {
            "api": "for each output row tile: mx.dequantize(weight[start:stop], scales[start:stop], biases[start:stop], group_size, bits, mode, dtype=scales.dtype); fc1_input @ tile.T; concatenate fused gate/value outputs",
            "source_is_quantized": source_is_quantized,
            "dequantize_available": dequantize_available,
            "tile_size_argument": tile_size,
            "tiling_before_projection": tiling_before_projection,
            "tiling_after_projection": tiling_after_projection,
            "tiling_after_interleaved": tiling_after_interleaved,
            "resident_dense_cache_after_interleaved": resident_cache_after_interleaved,
            "first_tiled_fc1_projection_seconds_includes_tile_materialization": first_tiled_fc1_projection_seconds,
            "max_transient_dense_tile_nbytes": max_tile_nbytes,
            "full_dense_nbytes_if_resident": full_dense_nbytes,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "max_transient_tile_nbytes_vs_full_dense": (max_tile_nbytes / full_dense_nbytes) if full_dense_nbytes else None,
            "extra_persistent_dense_nbytes_vs_packed_source": 0,
            "fused_gate_value_row_order_preserved": True,
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_candidate_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward flag plus one tiled dense projection helper reused for fc1; no default generation path, sigma/NFE, or non-fc1 weight changes",
            "compile_cost": "no mx.compile or custom Metal kernel is introduced; timing includes per-tile dequantization/materialization and synchronization needed to keep tiles transient",
            "memory": "candidate never stores a resident full dense fc1; artifact records max tile bytes, full-dense equivalent bytes, RSS/MLX deltas, pageouts, and swapouts",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence and chosen output tile size; tile size must be remeasured for other shapes before promotion",
            "maintainability": "local fc1-only switch with explicit LoRA fallback; prior resident dense cache remains separate and is forced off during this probe",
            "strict_equivalence": "fc1 projection, downstream SwiGLU hidden, downstream fc2 projection, and full block output are checked against the quantized baseline before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_ffn_fc2_input_chunked_qmm(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe ``FeedForward.fc2`` input-feature quantization-group chunked QMM."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    chunk_groups = int(args.ffn_fc2_input_chunk_groups)
    if chunk_groups <= 0:
        raise ValueError(f"--ffn-fc2-input-chunk-groups must be positive, got {chunk_groups}")

    source_is_quantized = getattr(block.mlp.fc2, "scales", None) is not None
    quantized_matmul_available = getattr(mx, "quantized_matmul", None) is not None
    if not source_is_quantized and not args.tiny:
        return {
            "name": "ffn_fc2_input_chunked_qmm",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc2 input-feature/group chunked quantized matmul",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_fc2_not_quantized",
            "decision_reason": (
                "real block fc2 is not an MLX quantized linear with public scales/biases; "
                "input-group chunked quantized_matmul was not attempted"
            ),
            "fc2_input_chunked_qmm": block.mlp.fc2_input_chunked_qmm_info(chunk_groups),
        }
    if source_is_quantized and not quantized_matmul_available:
        return {
            "name": "ffn_fc2_input_chunked_qmm",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc2 input-feature/group chunked quantized matmul",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_quantized_matmul_unavailable",
            "decision_reason": "mx.quantized_matmul is unavailable, so quantized fc2 input chunks cannot be run faithfully",
            "fc2_input_chunked_qmm": block.mlp.fc2_input_chunked_qmm_info(chunk_groups),
        }

    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc1_split_gate_value_quantized_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_fc2_input_chunked_qmm_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
        "use_ffn_subgraph_compile_candidate",
    )
    original_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_fc1_tile_size = int(getattr(block.mlp, "ffn_fc1_tiled_output_channels", 512))
    original_fc2_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))
    original_fc2_input_chunk_groups = int(getattr(block.mlp, "ffn_fc2_input_chunk_groups", 56))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: fc1, SwiGLU, output-channel tiling, dense-dequant,
        # projection-rank, sequence chunking, compile, Metal, and explicit-contiguity routes are
        # forced off. Only fc2 splits its input quantization groups across partial QMM launches.
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = original_fc1_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.mlp.ffn_fc2_input_chunk_groups = chunk_groups
        block.mlp.use_ffn_fc2_input_chunked_qmm_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        if not original_flags.get("use_ffn_fc1_dense_dequant_candidate", False):
            block.mlp.clear_fc1_dense_dequant_cache()
        if not original_flags.get("use_ffn_fc2_dense_dequant_candidate", False):
            block.mlp.clear_fc2_dense_dequant_cache()
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        baseline_fc1 = block.mlp._fc1_project(fc1_input)
        baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=fc1_input)
        baseline_fc2 = block.mlp._fc2_project(baseline_hidden)

        set_candidate(True)
        candidate_fc1 = block.mlp._fc1_project(fc1_input)
        candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=fc1_input)
        qmm_before_projection = block.mlp.fc2_input_chunked_qmm_info(chunk_groups)
        chunked_projection_started = time.perf_counter()
        candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
        mx.eval(
            fc1_input,
            baseline_fc1,
            candidate_fc1,
            baseline_hidden,
            candidate_hidden,
            baseline_fc2,
            candidate_fc2,
        )
        mx.synchronize()
        first_chunked_fc2_projection_seconds = time.perf_counter() - chunked_projection_started
        qmm_after_projection = block.mlp.fc2_input_chunked_qmm_info(chunk_groups)
        projection_parity = {
            "fc1_unchanged": _diff_stats(baseline_fc1, candidate_fc1),
            "swiglu_hidden_unchanged": _diff_stats(baseline_hidden, candidate_hidden),
            "fc2_input_chunked_qmm": _diff_stats(baseline_fc2, candidate_fc2),
        }
        projection_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "fc1_baseline_output_shape": list(baseline_fc1.shape),
            "fc1_candidate_output_shape": list(candidate_fc1.shape),
            "fc1_baseline_output_dtype": str(baseline_fc1.dtype),
            "fc1_candidate_output_dtype": str(candidate_fc1.dtype),
            "hidden_baseline_shape": list(baseline_hidden.shape),
            "hidden_candidate_shape": list(candidate_hidden.shape),
            "hidden_baseline_dtype": str(baseline_hidden.dtype),
            "hidden_candidate_dtype": str(candidate_hidden.dtype),
            "fc2_baseline_output_shape": list(baseline_fc2.shape),
            "fc2_candidate_output_shape": list(candidate_fc2.shape),
            "fc2_baseline_output_dtype": str(baseline_fc2.dtype),
            "fc2_candidate_output_dtype": str(candidate_fc2.dtype),
            "fc2_shape_matches_baseline": bool(baseline_fc2.shape == candidate_fc2.shape),
            "fc2_dtype_matches_baseline": bool(baseline_fc2.dtype == candidate_fc2.dtype),
            "fc2_input_chunk_groups": chunk_groups,
            "fc2_input_chunk_count": qmm_after_projection.get("chunk_count"),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_candidate_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 17011,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        qmm_after_interleaved = block.mlp.fc2_input_chunked_qmm_info(chunk_groups)
        resident_fc1_cache_after_interleaved = block.mlp.fc1_dense_dequant_cache_info()
        resident_fc2_cache_after_interleaved = block.mlp.fc2_dense_dequant_cache_info()
    finally:
        for name, value in original_flags.items():
            setattr(block.mlp, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = original_fc1_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.mlp.ffn_fc2_input_chunk_groups = original_fc2_input_chunk_groups
        if not original_flags.get("use_ffn_fc1_dense_dequant_candidate", False):
            block.mlp.clear_fc1_dense_dequant_cache()
        if not original_flags.get("use_ffn_fc2_dense_dequant_candidate", False):
            block.mlp.clear_fc2_dense_dequant_cache()

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "fc2 input/group chunked QMM changed projection or full-block outputs beyond configured parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says fc2 input/group chunked QMM is slower than baseline quantized fc2"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate fc2 input/group chunked QMM from baseline; "
            "no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "fc2 input/group chunked QMM is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "fc2 input/group chunked QMM is within parity bounds, disabled by default, memory-clean, "
            "and faster than baseline quantized fc2 outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    chunk_count = int(qmm_after_interleaved.get("chunk_count") or 0)
    partial_output_nbytes = int(getattr(candidate_fc2, "nbytes", 0)) if "candidate_fc2" in locals() else None
    return {
        "name": "ffn_fc2_input_chunked_qmm",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "FeedForward.fc2 hidden projection split along input quantization groups with partial mx.quantized_matmul accumulation",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior fc2 routes changed dispatch rank, dense weight "
            "materialization, output-channel tiling, or pre-fc2 materialization. This single-variable probe keeps "
            "fc1, SwiGLU, output rows, weights, and LoRA fallback unchanged while testing whether group-aligned "
            "input chunks improve the fc2 QMM scheduler/cache boundary without dense dequantization."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {name: False for name in ffn_flag_names},
            "enabled_flag_for_this_run": "use_ffn_fc2_input_chunked_qmm_candidate",
            "helper": "quantized_matmul_input_chunked_projection + FeedForward.fc2_input_chunked_qmm_info",
            "single_variable_guard": "all known FFN candidate flags are forced off except use_ffn_fc2_input_chunked_qmm_candidate during this probe",
            "lora_path": "falls back to the existing quantized fc2 base projection whenever lora is not None; LoRA fc2 delta consumes the baseline hidden tensor",
            "dense_dequantization": False,
            "resident_full_dense_cache": False,
        },
        "fc2_input_chunked_qmm": {
            "api": "for each input group chunk: mx.quantized_matmul(hidden[..., feature_start:feature_stop], weight[:, packed_start:packed_stop], sliced scales/biases, transpose=True); sum partial outputs; add learned bias once",
            "source_is_quantized": source_is_quantized,
            "quantized_matmul_available": quantized_matmul_available,
            "chunk_groups_argument": chunk_groups,
            "qmm_before_projection": qmm_before_projection,
            "qmm_after_projection": qmm_after_projection,
            "qmm_after_interleaved": qmm_after_interleaved,
            "resident_fc1_dense_cache_after_interleaved": resident_fc1_cache_after_interleaved,
            "resident_fc2_dense_cache_after_interleaved": resident_fc2_cache_after_interleaved,
            "first_chunked_fc2_projection_seconds": first_chunked_fc2_projection_seconds,
            "partial_qmm_launches_per_fc2": chunk_count,
            "partial_output_nbytes_each": partial_output_nbytes,
            "learned_bias_added_once": True,
            "dense_weight_materialization": False,
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_candidate_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
            "known_risk": "partial QMM accumulation can change floating-point reduction order versus the monolithic fc2 QMM",
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward flag plus a packed-weight input-group slicing helper; no default generation path, sigma/NFE, or non-fc2 weight changes",
            "compile_cost": "no mx.compile or custom Metal kernel is introduced; candidate increases the number of fc2 QMM launches by the chunk count",
            "memory": "candidate never reconstructs dense fc2 weights; artifact records RSS/MLX deltas, pageouts, swapouts, chunk metadata, and partial-output size",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence and chosen input chunk groups; chunk size must be remeasured for other shapes before promotion",
            "maintainability": "local fc2-only switch with explicit LoRA fallback; dense-dequant and output-channel tiled fc2 probes remain separate and are forced off during this probe",
            "strict_equivalence": "fc1 projection, SwiGLU hidden, fc2 projection, and full block output are checked against the monolithic quantized baseline before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_ffn_fc2_dense_dequant(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe resident dense reconstruction of only ``FeedForward.fc2`` quantized weights."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    source_is_quantized = getattr(block.mlp.fc2, "scales", None) is not None
    dequantize_available = getattr(mx, "dequantize", None) is not None
    if not source_is_quantized and not args.tiny:
        return {
            "name": "ffn_fc2_dense_dequant",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc2 quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_fc2_reconstruction",
            "decision_reason": (
                "real block fc2 is not an MLX quantized linear with public scales/biases; "
                "dense reconstruction was not attempted"
            ),
            "fc2_cache_info": block.mlp.fc2_dense_dequant_cache_info(),
        }
    if source_is_quantized and not dequantize_available:
        return {
            "name": "ffn_fc2_dense_dequant",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc2 quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_fc2_reconstruction",
            "decision_reason": "mx.dequantize is unavailable, so quantized fc2 weights cannot be reconstructed safely",
            "fc2_cache_info": block.mlp.fc2_dense_dequant_cache_info(),
        }

    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    original_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: fc1, SwiGLU, projection rank, explicit contiguity,
        # chunking, and prior Metal/2D routes are forced off; only fc2 uses the resident dense weight.
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.use_ffn_fc2_dense_dequant_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        block.mlp.clear_fc2_dense_dequant_cache()
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        baseline_fc1 = block.mlp._fc1_project(fc1_input)
        baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=fc1_input)
        baseline_fc2 = block.mlp._fc2_project(baseline_hidden)

        set_candidate(True)
        candidate_fc1 = block.mlp._fc1_project(fc1_input)
        candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=fc1_input)
        cache_before_projection = block.mlp.fc2_dense_dequant_cache_info()
        dense_projection_started = time.perf_counter()
        candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
        mx.eval(
            fc1_input,
            baseline_fc1,
            candidate_fc1,
            baseline_hidden,
            candidate_hidden,
            baseline_fc2,
            candidate_fc2,
        )
        mx.synchronize()
        first_dense_fc2_projection_seconds = time.perf_counter() - dense_projection_started
        cache_after_projection = block.mlp.fc2_dense_dequant_cache_info()
        projection_parity = {
            "fc1_unchanged": _diff_stats(baseline_fc1, candidate_fc1),
            "swiglu_hidden_unchanged": _diff_stats(baseline_hidden, candidate_hidden),
            "fc2_dense_dequant": _diff_stats(baseline_fc2, candidate_fc2),
        }
        projection_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "fc1_baseline_output_shape": list(baseline_fc1.shape),
            "fc1_candidate_output_shape": list(candidate_fc1.shape),
            "fc1_baseline_output_dtype": str(baseline_fc1.dtype),
            "fc1_candidate_output_dtype": str(candidate_fc1.dtype),
            "fc2_input_shape": list(baseline_hidden.shape),
            "fc2_input_dtype": str(baseline_hidden.dtype),
            "fc2_baseline_output_shape": list(baseline_fc2.shape),
            "fc2_candidate_output_shape": list(candidate_fc2.shape),
            "fc2_baseline_output_dtype": str(baseline_fc2.dtype),
            "fc2_candidate_output_dtype": str(candidate_fc2.dtype),
            "fc1_shape_dtype_unchanged_by_candidate": bool(
                baseline_fc1.shape == candidate_fc1.shape and baseline_fc1.dtype == candidate_fc1.dtype
            ),
            "hidden_shape_dtype_unchanged_by_candidate": bool(
                baseline_hidden.shape == candidate_hidden.shape and baseline_hidden.dtype == candidate_hidden.dtype
            ),
            "fc2_shape_matches_baseline": bool(baseline_fc2.shape == candidate_fc2.shape),
            "fc2_dtype_matches_baseline": bool(baseline_fc2.dtype == candidate_fc2.dtype),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_candidate_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 12011,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        cache_after_interleaved = block.mlp.fc2_dense_dequant_cache_info()
    finally:
        for name, value in original_flags.items():
            setattr(block.mlp, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        if not original_flags.get("use_ffn_fc2_dense_dequant_candidate", False):
            block.mlp.clear_fc2_dense_dequant_cache()

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "resident dense-dequantized fc2 changed projection or full-block outputs beyond configured parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says resident dense-dequantized fc2 is slower than baseline quantized fc2"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate resident dense-dequantized fc2 from baseline; "
            "no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "resident dense-dequantized fc2 is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "resident dense-dequantized fc2 is within parity bounds, disabled by default, memory-clean, "
            "and faster than baseline quantized fc2 outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    dense_resident_nbytes = int(cache_after_interleaved.get("dense_nbytes") or 0)
    packed_source_nbytes = int(cache_after_interleaved.get("source_weight_nbytes") or 0) + int(
        cache_after_interleaved.get("source_scales_nbytes") or 0
    ) + int(cache_after_interleaved.get("source_biases_nbytes") or 0)
    return {
        "name": "ffn_fc2_dense_dequant",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "FeedForward.fc2 hidden projection uses a resident dense weight reconstructed from the quantized fc2 pack/scales/biases",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior fc2 probes changed rank scheduling or hidden "
            "materialization; this probe changes only the fc2 weight representation after the real 4-bit "
            "block is resident, replacing repeated QMM dequant/GEMM dispatch with dense MLX matmul."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {name: False for name in ffn_flag_names},
            "enabled_flag_for_this_run": "use_ffn_fc2_dense_dequant_candidate",
            "helper": "FeedForward._fc2_dense_dequant_weight + dense_linear_projection",
            "single_variable_guard": "prior FFN split, projection-rank, Metal SwiGLU, sequence-chunk, and explicit-contiguity candidates are forced off during this probe",
            "lora_path": "falls back to the existing quantized fc2 base projection whenever lora is not None; LoRA fc2 deltas keep their existing path",
            "cache_lifecycle": "cache key follows the loaded fc2 weight/scales/biases arrays; QuantizedBlockProvider clears it whenever the reusable resident block slot is rebound",
        },
        "fc2_source_reconstruction": {
            "api": "mx.dequantize(weight, scales, biases, group_size, bits, mode, dtype=scales.dtype)",
            "source_is_quantized": source_is_quantized,
            "dequantize_available": dequantize_available,
            "cache_before_projection": cache_before_projection,
            "cache_after_projection": cache_after_projection,
            "cache_after_interleaved": cache_after_interleaved,
            "first_dense_fc2_projection_seconds_includes_materialization": first_dense_fc2_projection_seconds,
            "resident_dense_nbytes": dense_resident_nbytes,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "extra_resident_nbytes_vs_packed_source": dense_resident_nbytes - packed_source_nbytes,
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_candidate_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward flag plus resident dense fc2 cache; no default generation path, sigma/NFE, or non-fc2 weight changes",
            "compile_cost": "no mx.compile or custom Metal kernel is introduced; first dense fc2 projection records dequantization/materialization cost separately from warm interleaved timing",
            "memory": "candidate holds an extra dense fc2 weight while the quantized block is resident; artifact records dense bytes, packed-source bytes, RSS/MLX deltas, pageouts, and swapouts",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence and one resident block; larger shapes and all-block lifecycle need remeasurement before promotion",
            "maintainability": "local fc2-only switch with explicit LoRA fallback and provider cache invalidation on block-slot rebinding",
            "strict_equivalence": "fc1 projection, SwiGLU hidden, fc2 projection, and full block output are checked against the quantized baseline before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_ffn_hidden_tile_stream(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe hidden-channel streamed FFN tiles through FC1/SwiGLU/FC2 partials."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    tile_groups = int(args.ffn_hidden_tile_groups)
    if tile_groups <= 0:
        raise ValueError(f"--ffn-hidden-tile-groups must be positive, got {tile_groups}")

    fc2_scales = getattr(block.mlp.fc2, "scales", None)
    source_is_quantized = fc2_scales is not None
    if not source_is_quantized and not args.tiny:
        return {
            "name": "ffn_hidden_tile_stream",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward hidden-channel tiles between fc1 gate/value rows and fc2 input groups",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_fc2_not_quantized",
            "decision_reason": "real-block hidden-tile streaming requires FC2 quantization-group metadata for packed input-column slices",
        }

    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc1_split_gate_value_quantized_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_fc2_input_chunked_qmm_candidate",
        "use_ffn_hidden_tile_stream_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
        "use_ffn_subgraph_compile_candidate",
    )
    original_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_sequence_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_fc1_tile_size = int(getattr(block.mlp, "ffn_fc1_tiled_output_channels", 512))
    original_fc2_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))
    original_fc2_chunk_groups = int(getattr(block.mlp, "ffn_fc2_input_chunk_groups", 56))
    original_hidden_tile_groups = int(getattr(block.mlp, "ffn_hidden_tile_groups", 56))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: every other FFN schedule/dequant/Metal/compile flag is off.
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        block.mlp.ffn_sequence_chunk_size = original_sequence_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = original_fc1_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.mlp.ffn_fc2_input_chunk_groups = original_fc2_chunk_groups
        block.mlp.ffn_hidden_tile_groups = tile_groups
        block.mlp.use_ffn_hidden_tile_stream_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    def component_call(enabled: bool, fc1_input: mx.array) -> tuple[mx.array, dict[str, Any]]:
        set_candidate(enabled)
        _reset_mlx_peak()
        metrics_before = _metrics()
        started = time.perf_counter()
        out = block.mlp(fc1_input)
        mx.eval(out)
        mx.synchronize()
        metrics_after = _metrics()
        return out, {
            "seconds": time.perf_counter() - started,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "metrics_delta": _delta(metrics_before, metrics_after),
        }

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        mx.eval(fc1_input)
        mx.synchronize()
        batch = int(fc1_input.shape[0]) if len(fc1_input.shape) >= 3 else None
        sequence = int(fc1_input.shape[1]) if len(fc1_input.shape) >= 3 else None
        activation_nbytes = int(mx.zeros((1,), dtype=fc1_input.dtype).nbytes)
        tiling_before = block.mlp.hidden_tile_stream_info(
            tile_groups,
            batch_size=batch,
            sequence_length=sequence,
            activation_dtype_nbytes=activation_nbytes,
        )
        baseline_mlp, baseline_component_probe = component_call(False, fc1_input)
        candidate_mlp, candidate_component_probe = component_call(True, fc1_input)
        tiling_after_component = block.mlp.hidden_tile_stream_info(
            tile_groups,
            batch_size=batch,
            sequence_length=sequence,
            activation_dtype_nbytes=activation_nbytes,
        )
        mx.eval(baseline_mlp, candidate_mlp)
        mx.synchronize()
        mlp_parity = _diff_stats(baseline_mlp, candidate_mlp)
        baseline_component_peak = baseline_component_probe["metrics_after"].get("mlx_peak_bytes")
        candidate_component_peak = candidate_component_probe["metrics_after"].get("mlx_peak_bytes")
        measured_component_peak_delta = (
            int(candidate_component_peak) - int(baseline_component_peak)
            if isinstance(candidate_component_peak, int) and isinstance(baseline_component_peak, int)
            else None
        )
        component_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "baseline_output_shape": list(baseline_mlp.shape),
            "candidate_output_shape": list(candidate_mlp.shape),
            "baseline_output_dtype": str(baseline_mlp.dtype),
            "candidate_output_dtype": str(candidate_mlp.dtype),
            "shape_matches_baseline": bool(baseline_mlp.shape == candidate_mlp.shape),
            "dtype_matches_baseline": bool(baseline_mlp.dtype == candidate_mlp.dtype),
            "tile_groups": tile_groups,
            "effective_tile_groups": tiling_after_component.get("effective_tile_groups"),
            "tile_count": tiling_after_component.get("tile_count"),
            "baseline_full_gate_value_nbytes": tiling_after_component.get("full_gate_value_nbytes_baseline"),
            "baseline_full_hidden_nbytes": tiling_after_component.get("full_hidden_nbytes_baseline"),
            "candidate_max_tile_gate_value_nbytes": tiling_after_component.get("max_tile_gate_value_nbytes"),
            "candidate_max_tile_hidden_nbytes": tiling_after_component.get("max_tile_hidden_nbytes"),
            "candidate_accumulator_nbytes": tiling_after_component.get("accumulator_nbytes"),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 19037,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        tiling_after_interleaved = block.mlp.hidden_tile_stream_info(
            tile_groups,
            batch_size=batch,
            sequence_length=sequence,
            activation_dtype_nbytes=activation_nbytes,
        )
    finally:
        for name, value in original_flags.items():
            setattr(block.mlp, name, value)
        block.mlp.ffn_sequence_chunk_size = original_sequence_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = original_fc1_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.mlp.ffn_fc2_input_chunk_groups = original_fc2_chunk_groups
        block.mlp.ffn_hidden_tile_groups = original_hidden_tile_groups

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    mlp_parity_ok = mlp_parity["max_abs"] <= args.parity_atol and mlp_parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    all_parity_ok = bool(mlp_parity_ok and first_parity_ok and parity_ok)
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    measured_component_peak_lower = (
        measured_component_peak_delta is not None and measured_component_peak_delta < 0 and memory_ok
    )

    if not all_parity_ok:
        decision = "reject_parity"
        reason = (
            "hidden-channel streamed FFN changed MLP or full-block outputs beyond strict parity bounds; "
            "the likely cause is FC2 input-group partial reduction/rounding order"
        )
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says hidden-channel streamed FFN is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        reason = "hidden-channel streamed FFN passed parity but was not faster than noise; generation A/B remains blocked"
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "hidden-channel streamed FFN was faster than noise but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = "hidden-channel streamed FFN is strict-parity, disabled by default, memory-clean, and faster than noise"
        promoted = True

    strict_full_zero = (
        mlp_parity["max_abs"] == 0.0
        and mlp_parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
        and parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
    )
    return {
        "name": "ffn_hidden_tile_stream",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "FeedForward hidden-channel streaming: FC1 gate/value row slices -> tile SwiGLU -> FC2 input-group partial projection accumulator",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. The trace materializes full ffn_fc1_gate_value and "
            "ffn_swiglu_hidden buffers; this candidate attacks those HBM/unified-memory round trips directly."
        ),
        "opt_in_only": True,
        "strict_exact_semantics_required_for_promotion": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "profiler_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "enabled_flag_for_this_run": "use_ffn_hidden_tile_stream_candidate",
            "tile_groups_attribute": "ffn_hidden_tile_groups",
            "tile_groups": tile_groups,
            "helper": "linear_output_row_slice_projection(fc1 gate/value rows) + linear_input_range_partial_projection(fc2 packed input groups)",
            "single_variable_guard": "all other FFN split/dequant/chunk/Metal/compile flags are forced off during this probe",
            "lora_path": "falls back to the existing unstreamed path whenever lora is not None",
        },
        "tiling": {
            "source_is_quantized_fc2": source_is_quantized,
            "tiling_before_component": tiling_before,
            "tiling_after_component": tiling_after_component,
            "tiling_after_interleaved": tiling_after_interleaved,
            "group_alignment_contract": "tile boundaries are whole FC2 quantization groups; FC1 gate/value rows use the same hidden-channel interval",
            "materialization_policy": "mx.eval fc1 input once, then mx.eval the float32 accumulator after each hidden tile so tile-local gate/value/hidden can retire",
        },
        "component_shape_dtype_contract": component_shape_dtype,
        "component_memory_probe": {
            "baseline": baseline_component_probe,
            "candidate": candidate_component_probe,
            "candidate_minus_baseline_peak_bytes": measured_component_peak_delta,
            "measured_component_peak_lower": measured_component_peak_lower,
        },
        "mlp_parity": mlp_parity,
        "mlp_parity_ok": mlp_parity_ok,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "parity_vs_interleaved_baseline": parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block delta to every DiT block and denoiser evaluation; generation A/B remains blocked unless this block gate is faster-than-noise and memory-clean",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward branch plus a packed-input-range projection helper; no scalar generation knobs changed",
            "compile_cost": "no custom Metal or mx.compile kernel; warm samples include multiple FC1 row-slice QMM and FC2 partial-QMM launches per tile",
            "memory": "candidate targets full FFN gate/value and hidden materialization; artifact records theoretical tile bytes plus real MLX/RSS/pageout/swapout deltas",
            "parity_risk": "FC2 K-reduction is split over input groups, so partial-output rounding can differ from one full MLX QMM",
            "promotion_gate": "strict parity, faster-than-noise real-block timing, and pageout/swapout-clean memory are all required before any generation A/B",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_ffn_fc2_tiled_dense_dequant(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe transient output-channel tiled dense reconstruction of ``FeedForward.fc2``."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    tile_size = int(args.ffn_fc2_tile_size)
    if tile_size <= 0:
        raise ValueError(f"--ffn-fc2-tile-size must be positive, got {tile_size}")

    source_is_quantized = getattr(block.mlp.fc2, "scales", None) is not None
    dequantize_available = getattr(mx, "dequantize", None) is not None
    if not source_is_quantized and not args.tiny:
        return {
            "name": "ffn_fc2_tiled_dense_dequant",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc2 output-channel tiled quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_fc2_reconstruction",
            "decision_reason": (
                "real block fc2 is not an MLX quantized linear with public scales/biases; "
                "tiled dense reconstruction was not attempted"
            ),
            "fc2_tiling": block.mlp.fc2_tiled_dense_dequant_info(tile_size),
        }
    if source_is_quantized and not dequantize_available:
        return {
            "name": "ffn_fc2_tiled_dense_dequant",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward.fc2 output-channel tiled quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_fc2_reconstruction",
            "decision_reason": "mx.dequantize is unavailable, so quantized fc2 weight tiles cannot be reconstructed safely",
            "fc2_tiling": block.mlp.fc2_tiled_dense_dequant_info(tile_size),
        }

    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    original_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: fc1, SwiGLU, projection rank, resident dense fc2,
        # chunking, Metal, and explicit-contiguity candidates are forced off. Only fc2 uses
        # transient output-channel dense-dequant tiles.
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = tile_size
        block.mlp.use_ffn_fc2_tiled_dense_dequant_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        if not original_flags.get("use_ffn_fc2_dense_dequant_candidate", False):
            block.mlp.clear_fc2_dense_dequant_cache()
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        baseline_fc1 = block.mlp._fc1_project(fc1_input)
        baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=fc1_input)
        baseline_fc2 = block.mlp._fc2_project(baseline_hidden)

        set_candidate(True)
        candidate_fc1 = block.mlp._fc1_project(fc1_input)
        candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=fc1_input)
        tiling_before_projection = block.mlp.fc2_tiled_dense_dequant_info(tile_size)
        tiled_projection_started = time.perf_counter()
        candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
        mx.eval(
            fc1_input,
            baseline_fc1,
            candidate_fc1,
            baseline_hidden,
            candidate_hidden,
            baseline_fc2,
            candidate_fc2,
        )
        mx.synchronize()
        first_tiled_fc2_projection_seconds = time.perf_counter() - tiled_projection_started
        tiling_after_projection = block.mlp.fc2_tiled_dense_dequant_info(tile_size)
        projection_parity = {
            "fc1_unchanged": _diff_stats(baseline_fc1, candidate_fc1),
            "swiglu_hidden_unchanged": _diff_stats(baseline_hidden, candidate_hidden),
            "fc2_tiled_dense_dequant": _diff_stats(baseline_fc2, candidate_fc2),
        }
        projection_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "fc1_baseline_output_shape": list(baseline_fc1.shape),
            "fc1_candidate_output_shape": list(candidate_fc1.shape),
            "fc1_baseline_output_dtype": str(baseline_fc1.dtype),
            "fc1_candidate_output_dtype": str(candidate_fc1.dtype),
            "fc2_input_shape": list(baseline_hidden.shape),
            "fc2_input_dtype": str(baseline_hidden.dtype),
            "fc2_baseline_output_shape": list(baseline_fc2.shape),
            "fc2_candidate_output_shape": list(candidate_fc2.shape),
            "fc2_baseline_output_dtype": str(baseline_fc2.dtype),
            "fc2_candidate_output_dtype": str(candidate_fc2.dtype),
            "fc1_shape_dtype_unchanged_by_candidate": bool(
                baseline_fc1.shape == candidate_fc1.shape and baseline_fc1.dtype == candidate_fc1.dtype
            ),
            "hidden_shape_dtype_unchanged_by_candidate": bool(
                baseline_hidden.shape == candidate_hidden.shape and baseline_hidden.dtype == candidate_hidden.dtype
            ),
            "fc2_shape_matches_baseline": bool(baseline_fc2.shape == candidate_fc2.shape),
            "fc2_dtype_matches_baseline": bool(baseline_fc2.dtype == candidate_fc2.dtype),
            "fc2_output_tile_channels": tile_size,
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_candidate_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 15013,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        tiling_after_interleaved = block.mlp.fc2_tiled_dense_dequant_info(tile_size)
        resident_cache_after_interleaved = block.mlp.fc2_dense_dequant_cache_info()
    finally:
        for name, value in original_flags.items():
            setattr(block.mlp, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_tile_size
        if not original_flags.get("use_ffn_fc2_dense_dequant_candidate", False):
            block.mlp.clear_fc2_dense_dequant_cache()

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "transient tiled dense-dequantized fc2 changed projection or full-block outputs beyond configured parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says transient tiled dense-dequantized fc2 is slower than baseline quantized fc2"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate transient tiled dense-dequantized fc2 "
            "from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "transient tiled dense-dequantized fc2 is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "transient tiled dense-dequantized fc2 is within parity bounds, disabled by default, "
            "does not keep a full dense fc2 resident, is memory-clean, and is faster than baseline outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    max_tile_nbytes = int(tiling_after_interleaved.get("max_dense_tile_nbytes") or 0)
    full_dense_nbytes = int(tiling_after_interleaved.get("full_dense_nbytes_if_resident") or 0)
    packed_source_nbytes = int(tiling_after_interleaved.get("packed_quantized_source_nbytes") or 0)
    return {
        "name": "ffn_fc2_tiled_dense_dequant",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "FeedForward.fc2 hidden projection transiently dequantizes output-channel tiles and runs dense matmul per tile",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. The prior resident dense fc2 probe showed a compute win "
            "but introduced pageouts by keeping the full dense fc2 matrix live. This probe changes only that "
            "memory boundary: it keeps fc1, SwiGLU, projection rank, weights, and LoRA fallback unchanged while "
            "testing whether per-output-channel dense tiles retain enough GEMM benefit without persistent full-dense residency."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {name: False for name in ffn_flag_names},
            "enabled_flag_for_this_run": "use_ffn_fc2_tiled_dense_dequant_candidate",
            "helper": "tiled_dense_linear_projection + FeedForward.fc2_tiled_dense_dequant_info",
            "single_variable_guard": "prior FFN split, projection-rank, resident dense, Metal SwiGLU, sequence-chunk, and explicit-contiguity candidates are forced off during this probe",
            "lora_path": "falls back to the existing quantized fc2 base projection whenever lora is not None; LoRA fc2 deltas keep their existing path",
            "resident_full_dense_cache": False,
        },
        "fc2_tiling": {
            "api": "for each output row tile: mx.dequantize(weight[start:stop], scales[start:stop], biases[start:stop], group_size, bits, mode, dtype=scales.dtype); hidden @ tile.T; concatenate outputs",
            "source_is_quantized": source_is_quantized,
            "dequantize_available": dequantize_available,
            "tile_size_argument": tile_size,
            "tiling_before_projection": tiling_before_projection,
            "tiling_after_projection": tiling_after_projection,
            "tiling_after_interleaved": tiling_after_interleaved,
            "resident_dense_cache_after_interleaved": resident_cache_after_interleaved,
            "first_tiled_fc2_projection_seconds_includes_tile_materialization": first_tiled_fc2_projection_seconds,
            "max_transient_dense_tile_nbytes": max_tile_nbytes,
            "full_dense_nbytes_if_resident": full_dense_nbytes,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "max_transient_tile_nbytes_vs_full_dense": (max_tile_nbytes / full_dense_nbytes) if full_dense_nbytes else None,
            "extra_persistent_dense_nbytes_vs_packed_source": 0,
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_candidate_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward flag plus one tiled dense projection helper; no default generation path, sigma/NFE, or non-fc2 weight changes",
            "compile_cost": "no mx.compile or custom Metal kernel is introduced; timing includes per-tile dequantization/materialization and synchronization needed to keep tiles transient",
            "memory": "candidate never stores a resident full dense fc2; artifact records max tile bytes, full-dense equivalent bytes, RSS/MLX deltas, pageouts, and swapouts",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence and chosen output tile size; tile size must be remeasured for other shapes before promotion",
            "maintainability": "local fc2-only switch with explicit LoRA fallback; prior resident dense cache remains separate and is forced off during this probe",
            "strict_equivalence": "fc1 projection, SwiGLU hidden, fc2 projection, and full block output are checked against the quantized baseline before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_ffn_pre_fc1_contiguous(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe explicit ``mx.contiguous`` materialization of the FFN input before ``fc1``."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    mlp_candidate_flags = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    original_mlp_flags = {flag: bool(getattr(block.mlp, flag, False)) for flag in mlp_candidate_flags}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    block_candidate_flags = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    original_block_flags = {flag: bool(getattr(block, flag, False)) for flag in block_candidate_flags}
    attention_candidate_flags = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_qkv_tiled_dense_dequant_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    original_attn_flags = {flag: bool(getattr(block.attn, flag, False)) for flag in attention_candidate_flags}

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine with any prior block, Attention, or FFN candidate.
        for flag in block_candidate_flags:
            setattr(block, flag, False)
        for flag in attention_candidate_flags:
            setattr(block.attn, flag, False)
        for flag in mlp_candidate_flags:
            setattr(block.mlp, flag, False)
        block.mlp.use_ffn_pre_fc1_contiguous_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        baseline_fc1 = block.mlp._fc1_project(fc1_input)
        hidden_size = int(getattr(block.mlp, "_hidden", fc1_input.shape[-1]))
        set_candidate(True)
        candidate_input = block.mlp._pre_fc1_input(fc1_input)
        direct_input = materialize_ffn_input_contiguous(fc1_input, hidden_size)
        candidate_fc1 = block.mlp._fc1_project(fc1_input)
        mx.eval(fc1_input, baseline_fc1, candidate_input, direct_input, candidate_fc1)
        mx.synchronize()
        input_materialization_parity = {
            "method": _diff_stats(fc1_input, candidate_input),
            "direct_helper": _diff_stats(fc1_input, direct_input),
        }
        fc1_parity = _diff_stats(baseline_fc1, candidate_fc1)
        input_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "candidate_input_shape": list(candidate_input.shape),
            "candidate_input_dtype": str(candidate_input.dtype),
            "direct_input_shape": list(direct_input.shape),
            "direct_input_dtype": str(direct_input.dtype),
            "baseline_fc1_shape": list(baseline_fc1.shape),
            "candidate_fc1_shape": list(candidate_fc1.shape),
            "baseline_fc1_dtype": str(baseline_fc1.dtype),
            "candidate_fc1_dtype": str(candidate_fc1.dtype),
            "input_shape_matches_baseline": bool(fc1_input.shape == candidate_input.shape == direct_input.shape),
            "input_dtype_matches_baseline": bool(fc1_input.dtype == candidate_input.dtype == direct_input.dtype),
            "fc1_shape_matches_baseline": bool(baseline_fc1.shape == candidate_fc1.shape),
            "fc1_dtype_matches_baseline": bool(baseline_fc1.dtype == candidate_fc1.dtype),
            "materialization_api": "mx.contiguous(ffn_input) immediately before FeedForward.fc1",
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 9949,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        for flag, value in original_block_flags.items():
            setattr(block, flag, value)
        for flag, value in original_attn_flags.items():
            setattr(block.attn, flag, value)
        for flag, value in original_mlp_flags.items():
            setattr(block.mlp, flag, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    input_parity_ok = all(
        stats_["max_abs"] <= args.parity_atol and stats_["rel_l2"] <= args.parity_rel_l2
        for stats_ in input_materialization_parity.values()
    )
    fc1_parity_ok = fc1_parity["max_abs"] <= args.parity_atol and fc1_parity["rel_l2"] <= args.parity_rel_l2
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(input_parity_ok and fc1_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "pre-fc1 contiguous materialization changed input, fc1, or full-block outputs beyond configured bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says pre-fc1 input contiguous materialization is slower than baseline"
        promoted = False
    elif not stable_faster:
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        decision = "reject_unproven_noise"
        reason = (
            "interleaved timing and bootstrap CI do not separate pre-fc1 input contiguous materialization "
            "from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "pre-fc1 input contiguous materialization is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "pre-fc1 input contiguous materialization is strictly equivalent, disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_input_zero = all(
        stats_["max_abs"] == 0.0 and stats_["rel_l2"] == 0.0
        for stats_ in input_materialization_parity.values()
    )
    strict_fc1_zero = fc1_parity["max_abs"] == 0.0 and fc1_parity["rel_l2"] == 0.0
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    return {
        "name": "ffn_pre_fc1_contiguous",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "FFN activation input materialized with mx.contiguous immediately before FeedForward.fc1 quantized projection",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior FFN probes changed split policy, SwiGLU scheduling, "
            "projection rank, fc2-input materialization, or sequence chunking; this single-variable probe leaves "
            "SwiGLU, fc2, weights, projection rank, row order, and all Attention/residual candidates unchanged/off while "
            "testing whether explicitly materializing the normalized FFN input helps the dominant fc1 QMM scheduler."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {flag: False for flag in mlp_candidate_flags},
            "enabled_flag_for_this_run": "use_ffn_pre_fc1_contiguous_candidate",
            "helper": "FeedForward._pre_fc1_input -> materialize_ffn_input_contiguous",
            "single_variable_guard": (
                "prior block, Attention, and FFN candidate flags are forced off during this probe; "
                "only use_ffn_pre_fc1_contiguous_candidate is toggled"
            ),
            "lora_path": "falls back to the existing non-contiguous LoRA path whenever lora is not None",
        },
        "input_shape_dtype_contract": input_shape_dtype,
        "input_materialization_parity": input_materialization_parity,
        "input_materialization_parity_ok": input_parity_ok,
        "fc1_parity": fc1_parity,
        "fc1_parity_ok": fc1_parity_ok,
        "strict_input_parity_zero": strict_input_zero,
        "strict_fc1_parity_zero": strict_fc1_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one pure mx.contiguous helper plus one disabled-by-default FeedForward flag; no weights, quantization, sigma/NFE, cache, projection rank, or default generation path changes",
            "compile_cost": "no mx.compile, custom Metal kernel, or persistent compiler cache is introduced; first call records ordinary MLX lazy/kernel setup cost",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas; candidate may allocate one explicit fc1 input buffer",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other shapes and dtypes must be remeasured before promotion",
            "maintainability": "local pre-fc1 materialization switch with explicit LoRA fallback; other candidate flags are forced off during this probe",
            "strict_equivalence": "input materialization, fc1 output, and full-block output are checked against the baseline before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_ffn_pre_fc2_contiguous(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe explicit ``mx.contiguous`` materialization of SwiGLU hidden before ``fc2``."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    mlp_candidate_flags = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    original_mlp_flags = {flag: bool(getattr(block.mlp, flag, False)) for flag in mlp_candidate_flags}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))
    block_candidate_flags = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    original_block_flags = {flag: bool(getattr(block, flag, False)) for flag in block_candidate_flags}
    attention_candidate_flags = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    original_attn_flags = {flag: bool(getattr(block.attn, flag, False)) for flag in attention_candidate_flags}

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine with any prior block, Attention, or FFN candidate.
        for flag in block_candidate_flags:
            setattr(block, flag, False)
        for flag in attention_candidate_flags:
            setattr(block.attn, flag, False)
        for flag in mlp_candidate_flags:
            setattr(block.mlp, flag, False)
        block.mlp.use_ffn_pre_fc2_contiguous_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        fused = block.mlp._fc1_project(fc1_input)
        hidden = block.mlp._swiglu_hidden(fused)
        baseline_fc2 = block.mlp._fc2_project(hidden)
        set_candidate(True)
        candidate_hidden = block.mlp._pre_fc2_hidden(hidden)
        direct_hidden = materialize_ffn_hidden_contiguous(hidden, block.mlp._ffn)
        candidate_fc2 = block.mlp._fc2_project(hidden)
        mx.eval(fc1_input, fused, hidden, baseline_fc2, candidate_hidden, direct_hidden, candidate_fc2)
        mx.synchronize()
        hidden_materialization_parity = {
            "method": _diff_stats(hidden, candidate_hidden),
            "direct_helper": _diff_stats(hidden, direct_hidden),
        }
        fc2_parity = _diff_stats(baseline_fc2, candidate_fc2)
        hidden_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "fused_shape": list(fused.shape),
            "fused_dtype": str(fused.dtype),
            "hidden_shape": list(hidden.shape),
            "hidden_dtype": str(hidden.dtype),
            "candidate_hidden_shape": list(candidate_hidden.shape),
            "candidate_hidden_dtype": str(candidate_hidden.dtype),
            "direct_hidden_shape": list(direct_hidden.shape),
            "direct_hidden_dtype": str(direct_hidden.dtype),
            "baseline_fc2_shape": list(baseline_fc2.shape),
            "candidate_fc2_shape": list(candidate_fc2.shape),
            "baseline_fc2_dtype": str(baseline_fc2.dtype),
            "candidate_fc2_dtype": str(candidate_fc2.dtype),
            "hidden_shape_matches_baseline": bool(hidden.shape == candidate_hidden.shape == direct_hidden.shape),
            "hidden_dtype_matches_baseline": bool(hidden.dtype == candidate_hidden.dtype == direct_hidden.dtype),
            "fc2_shape_matches_baseline": bool(baseline_fc2.shape == candidate_fc2.shape),
            "fc2_dtype_matches_baseline": bool(baseline_fc2.dtype == candidate_fc2.dtype),
            "materialization_api": "mx.contiguous(hidden) immediately before FeedForward.fc2",
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 9931,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        for flag, value in original_block_flags.items():
            setattr(block, flag, value)
        for flag, value in original_attn_flags.items():
            setattr(block.attn, flag, value)
        for flag, value in original_mlp_flags.items():
            setattr(block.mlp, flag, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_tile_size

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    hidden_parity_ok = all(
        stats_["max_abs"] <= args.parity_atol and stats_["rel_l2"] <= args.parity_rel_l2
        for stats_ in hidden_materialization_parity.values()
    )
    fc2_parity_ok = fc2_parity["max_abs"] <= args.parity_atol and fc2_parity["rel_l2"] <= args.parity_rel_l2
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(hidden_parity_ok and fc2_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "pre-fc2 contiguous materialization changed hidden, fc2, or full-block outputs beyond configured bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says pre-fc2 hidden contiguous materialization is slower than baseline"
        promoted = False
    elif not stable_faster:
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        decision = "reject_unproven_noise"
        reason = (
            "interleaved timing and bootstrap CI do not separate pre-fc2 hidden contiguous materialization "
            "from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "pre-fc2 hidden contiguous materialization is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "pre-fc2 hidden contiguous materialization is strictly equivalent, disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_hidden_zero = all(
        stats_["max_abs"] == 0.0 and stats_["rel_l2"] == 0.0
        for stats_ in hidden_materialization_parity.values()
    )
    strict_fc2_zero = fc2_parity["max_abs"] == 0.0 and fc2_parity["rel_l2"] == 0.0
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    return {
        "name": "ffn_pre_fc2_contiguous",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "SwiGLU hidden tensor materialized with mx.contiguous immediately before FeedForward.fc2 quantized projection",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior FFN probes changed gate/value splitting, "
            "SwiGLU arithmetic scheduling, projection rank, or sequence chunking; this single-variable "
            "probe leaves fc1, SwiGLU math, fc2 weights, projection rank, and row order unchanged while "
            "testing whether an explicit hidden-buffer materialization helps the fc2 QMM scheduler."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {flag: False for flag in mlp_candidate_flags},
            "block_default_flags": {flag: False for flag in block_candidate_flags},
            "attention_default_flags": {flag: False for flag in attention_candidate_flags},
            "enabled_flag_for_this_run": "use_ffn_pre_fc2_contiguous_candidate",
            "helper": "FeedForward._pre_fc2_hidden -> materialize_ffn_hidden_contiguous",
            "single_variable_guard": "all known block, Attention, and FFN candidate flags are forced off except use_ffn_pre_fc2_contiguous_candidate during this probe",
            "lora_path": "falls back to the existing non-contiguous LoRA path whenever lora is not None; LoRA fc2 delta consumes the baseline hidden tensor",
        },
        "hidden_shape_dtype_contract": hidden_shape_dtype,
        "hidden_materialization_parity": hidden_materialization_parity,
        "hidden_materialization_parity_ok": hidden_parity_ok,
        "fc2_parity": fc2_parity,
        "fc2_parity_ok": fc2_parity_ok,
        "strict_hidden_parity_zero": strict_hidden_zero,
        "strict_fc2_parity_zero": strict_fc2_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one pure mx.contiguous helper plus one disabled-by-default FeedForward flag; no weights, quantization, sigma/NFE, cache, projection rank, or default generation path changes",
            "compile_cost": "no mx.compile, custom Metal kernel, or persistent compiler cache is introduced; first call records ordinary MLX lazy/kernel setup cost",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas; candidate may allocate one explicit hidden buffer before fc2",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other shapes and dtypes must be remeasured before promotion",
            "maintainability": "local pre-fc2 materialization switch with explicit LoRA fallback; all known block, Attention, and FFN candidates are forced off during this probe",
            "strict_equivalence": "hidden materialization, fc2 output, and full-block output are checked against the non-contiguous baseline before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_ffn_sequence_chunked(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe sequence-axis chunking across the FFN fc1/SwiGLU/fc2 row-independent path."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    chunk_size = int(args.ffn_sequence_chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"--ffn-sequence-chunk-size must be positive, got {chunk_size}")

    original_sequence_flag = bool(getattr(block.mlp, "use_ffn_sequence_chunk_candidate", False))
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_projection_flag = bool(getattr(block.mlp, "use_ffn_2d_projection_candidate", False))
    original_split_flag = bool(getattr(block.mlp, "use_mx_split_swiglu_candidate", False))
    original_metal_flag = bool(getattr(block.mlp, "use_ffn_metal_swiglu_candidate", False))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine with prior FFN split/projection/Metal routes.
        block.mlp.use_mx_split_swiglu_candidate = False
        block.mlp.use_ffn_2d_projection_candidate = False
        block.mlp.use_ffn_metal_swiglu_candidate = False
        block.mlp.ffn_sequence_chunk_size = chunk_size
        block.mlp.use_ffn_sequence_chunk_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    def component_call(enabled: bool) -> tuple[mx.array, dict[str, Any]]:
        set_candidate(enabled)
        _reset_mlx_peak()
        metrics_before = _metrics()
        started = time.perf_counter()
        out = block.mlp(fc1_input)
        mx.eval(out)
        mx.synchronize()
        metrics_after = _metrics()
        return out, {
            "seconds": time.perf_counter() - started,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "metrics_delta": _delta(metrics_before, metrics_after),
        }

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        fc1_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        sequence = int(fc1_input.shape[1]) if len(fc1_input.shape) == 3 else 0
        max_chunk_rows = min(chunk_size, sequence) if sequence else None
        chunk_count = int(math.ceil(sequence / chunk_size)) if sequence else None
        baseline_mlp, baseline_component_probe = component_call(False)
        candidate_mlp, candidate_component_probe = component_call(True)
        mx.eval(fc1_input, baseline_mlp, candidate_mlp)
        mx.synchronize()
        mlp_parity = _diff_stats(baseline_mlp, candidate_mlp)
        baseline_component_peak = baseline_component_probe["metrics_after"].get("mlx_peak_bytes")
        candidate_component_peak = candidate_component_probe["metrics_after"].get("mlx_peak_bytes")
        measured_component_peak_delta = (
            int(candidate_component_peak) - int(baseline_component_peak)
            if isinstance(candidate_component_peak, int) and isinstance(baseline_component_peak, int)
            else None
        )
        component_shape_dtype = {
            "fc1_input_shape": list(fc1_input.shape),
            "fc1_input_dtype": str(fc1_input.dtype),
            "chunk_size": chunk_size,
            "sequence_length": sequence,
            "chunk_count": chunk_count,
            "max_chunk_rows": max_chunk_rows,
            "tail_chunk_rows": (sequence % chunk_size) or chunk_size if sequence else None,
            "baseline_output_shape": list(baseline_mlp.shape),
            "candidate_output_shape": list(candidate_mlp.shape),
            "baseline_output_dtype": str(baseline_mlp.dtype),
            "candidate_output_dtype": str(candidate_mlp.dtype),
            "shape_matches_baseline": bool(baseline_mlp.shape == candidate_mlp.shape),
            "dtype_matches_baseline": bool(baseline_mlp.dtype == candidate_mlp.dtype),
            "baseline_largest_fc1_temporary_elements": int(fc1_input.shape[0]) * sequence * 2 * int(block.mlp._ffn)
            if len(fc1_input.shape) == 3
            else None,
            "chunked_largest_fc1_temporary_elements": int(fc1_input.shape[0]) * int(max_chunk_rows or 0) * 2 * int(block.mlp._ffn)
            if len(fc1_input.shape) == 3
            else None,
            "baseline_largest_hidden_temporary_elements": int(fc1_input.shape[0]) * sequence * int(block.mlp._ffn)
            if len(fc1_input.shape) == 3
            else None,
            "chunked_largest_hidden_temporary_elements": int(fc1_input.shape[0]) * int(max_chunk_rows or 0) * int(block.mlp._ffn)
            if len(fc1_input.shape) == 3
            else None,
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 7759,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        block.mlp.use_ffn_sequence_chunk_candidate = original_sequence_flag
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.use_ffn_2d_projection_candidate = original_projection_flag
        block.mlp.use_mx_split_swiglu_candidate = original_split_flag
        block.mlp.use_ffn_metal_swiglu_candidate = original_metal_flag

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    mlp_parity_ok = mlp_parity["max_abs"] <= args.parity_atol and mlp_parity["rel_l2"] <= args.parity_rel_l2
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    measured_component_peak_lower = (
        measured_component_peak_delta is not None and measured_component_peak_delta < 0 and memory_ok
    )
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(mlp_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "strict-equivalence check failed for the FFN sequence-chunked candidate"
        promoted = False
    elif stable_slower and not measured_component_peak_lower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the FFN sequence-chunked candidate is slower than baseline"
        promoted = False
    elif not stable_faster and not measured_component_peak_lower:
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        decision = "reject_unproven_noise"
        reason = (
            "interleaved timing and measured component-memory evidence do not separate the FFN sequence-chunked "
            "candidate from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "FFN sequence-chunked candidate showed a possible win, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        if stable_faster:
            reason = (
                "FFN sequence-chunked candidate is strictly equivalent, disabled by default, memory-clean, "
                "and faster than interleaved baseline outside measured noise"
            )
        else:
            reason = (
                "FFN sequence-chunked candidate is strictly equivalent, disabled by default, memory-clean, "
                "and lowers the measured component MLX peak in the focused memory probe"
            )
        promoted = True

    strict_full_zero = (
        mlp_parity["max_abs"] == 0.0
        and mlp_parity["rel_l2"] == 0.0
        and parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    return {
        "name": "ffn_sequence_chunked_qmm_memory",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "contiguous sequence-axis chunks through FeedForward fc1, baseline slice SwiGLU, and fc2",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. FFN rows are independent along the packed sequence, "
            "so this single-variable memory-boundary probe slices only S rows while preserving projection rank, "
            "baseline SwiGLU slicing, weights, quantization, and row order."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "default_flags": {
                "use_mx_split_swiglu_candidate": False,
                "use_ffn_2d_projection_candidate": False,
                "use_ffn_metal_swiglu_candidate": False,
                "use_ffn_sequence_chunk_candidate": False,
                "ffn_sequence_chunk_size": 512,
            },
            "enabled_flag_for_this_run": "use_ffn_sequence_chunk_candidate",
            "chunk_size_attribute": "ffn_sequence_chunk_size",
            "chunk_size_rows": chunk_size,
            "single_variable_guard": "prior FFN split, projection-rank, and Metal SwiGLU candidate flags are forced off during this probe",
            "lora_path": "falls back to the existing unchunked LoRA path whenever lora is not None",
        },
        "component_shape_dtype_contract": component_shape_dtype,
        "component_memory_probe": {
            "baseline": baseline_component_probe,
            "candidate": candidate_component_probe,
            "candidate_minus_baseline_peak_bytes": measured_component_peak_delta,
            "measured_component_peak_lower": measured_component_peak_lower,
            "note": "focused MLP-only probe after resetting MLX peak memory; full-block interleaved phase still provides timing and pageout/swapout deltas",
        },
        "mlp_parity": mlp_parity,
        "mlp_parity_ok": mlp_parity_ok,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward branch plus a chunk-size attribute; no weights, quantization, sigma/NFE, cache, or default generation path changes",
            "compile_cost": "no mx.compile or persistent compiler cache is introduced; warm samples include extra per-chunk QMM launches and concat scheduling",
            "memory": "candidate reports a focused MLP peak probe plus interleaved pageout/swapout, MLX peak/cache, and RSS deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence and chunk size; other shapes/chunks must be remeasured",
            "maintainability": "local row-chunk loop is simple but can add launch/concat overhead and is intentionally not combined with prior FFN candidates",
            "strict_equivalence": "MLP output and full-block output are checked against the unchunked baseline before any keep decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, measured memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_ffn_subgraph_compile(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe a disabled FFN-only ``mx.compile`` wrapper around ``fc1 -> SwiGLU -> fc2``."""

    ffn_median = segment_stats.get("fc1_swiglu_fc2", {}).get("median_seconds")
    if getattr(mx, "compile", None) is None:
        return {
            "name": "ffn_subgraph_compile",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward-only fc1 -> native SwiGLU -> fc2 branch scheduling",
            "opt_in_only": True,
            "disabled_by_default": True,
            "candidate_available": False,
            "parity_ok": False,
            "promote": False,
            "decision": "reject_unavailable",
            "decision_reason": "mx.compile is unavailable in this MLX build, so the FFN subgraph compile candidate cannot run",
        }

    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
        "use_ffn_subgraph_compile_candidate",
    )
    block_flag_names = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    attn_flag_names = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_qkv_tiled_dense_dequant_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    original_ffn_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_block_flags = {name: bool(getattr(block, name, False)) for name in block_flag_names}
    original_attn_flags = {name: bool(getattr(block.attn, name, False)) for name in attn_flag_names}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_fc1_tile_size = int(getattr(block.mlp, "ffn_fc1_tiled_output_channels", 512))
    original_fc2_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))
    original_qkv_tile_size = int(getattr(block.attn, "qkv_tiled_output_channels", 2048))

    def restore_original_state() -> None:
        for name, value in original_ffn_flags.items():
            setattr(block.mlp, name, value)
        for name, value in original_block_flags.items():
            setattr(block, name, value)
        for name, value in original_attn_flags.items():
            setattr(block.attn, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = original_fc1_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.attn.qkv_tiled_output_channels = original_qkv_tile_size
        block.mlp.clear_ffn_subgraph_compile_cache()

    def set_candidate(enabled: bool) -> None:
        # Keep the probe single-variable: only the FeedForward-local compile flag may differ.
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        for name in block_flag_names:
            setattr(block, name, False)
        for name in attn_flag_names:
            setattr(block.attn, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = original_fc1_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.attn.qkv_tiled_output_channels = original_qkv_tile_size
        block.mlp.use_ffn_subgraph_compile_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    error_result: dict[str, Any] | None = None
    try:
        block.mlp.clear_ffn_subgraph_compile_cache()
        set_candidate(False)
        ffn_input = _ffn_projection_input(block, x, modulation, adaln_indices, rotary)
        baseline_branch = block.mlp._ffn_baseline_subgraph(ffn_input)
        mx.eval(ffn_input, baseline_branch)
        mx.synchronize()

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        set_candidate(True)
        candidate_branch = block.mlp(ffn_input)
        mx.eval(candidate_branch)
        mx.synchronize()
        branch_parity = _diff_stats(baseline_branch, candidate_branch)
        branch_shape_dtype = {
            "ffn_input_shape": list(ffn_input.shape),
            "ffn_input_dtype": str(ffn_input.dtype),
            "baseline_branch_shape": list(baseline_branch.shape),
            "candidate_branch_shape": list(candidate_branch.shape),
            "baseline_branch_dtype": str(baseline_branch.dtype),
            "candidate_branch_dtype": str(candidate_branch.dtype),
            "shape_matches_baseline": bool(candidate_branch.shape == baseline_branch.shape),
            "dtype_matches_baseline": bool(candidate_branch.dtype == baseline_branch.dtype),
            "compiled_cache_after_branch_probe": block.mlp.ffn_subgraph_compile_cache_info(),
        }

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 9197,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        cache_info_after = block.mlp.ffn_subgraph_compile_cache_info()
    except Exception as exc:
        after_error = _metrics()
        error_result = {
            "name": "ffn_subgraph_compile",
            "target_segment": "fc1_swiglu_fc2",
            "target_boundary": "FeedForward-only fc1 -> native SwiGLU -> fc2 branch scheduling",
            "selection_rationale": (
                f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
                f"{sequence_meta.get('sequence_length')}. This probe intentionally narrows the previously rejected "
                "full-block mx.compile route to only the FFN branch."
            ),
            "opt_in_only": True,
            "strict_exact_semantics": True,
            "disabled_by_default": True,
            "production_integrated": True,
            "candidate_available": True,
            "compile_attempted": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "parity_ok": False,
            "memory_gate_ok": False,
            "metrics_before": before,
            "metrics_after_error": after_error,
            "metrics_delta": _delta(before, after_error),
            "promote": False,
            "decision": "reject_compile_error",
            "decision_reason": "FFN-only mx.compile candidate raised before completing parity/timing, so it remains disabled and rejected",
        }
    finally:
        restore_original_state()

    if error_result is not None:
        return error_result

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    branch_parity_ok = branch_parity["max_abs"] <= args.parity_atol and branch_parity["rel_l2"] <= args.parity_rel_l2
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    all_parity_ok = bool(branch_parity_ok and parity_ok and first_parity_ok)
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    rss_delta = metrics_delta.get("current_rss_kib")
    mlx_cache_delta = metrics_delta.get("mlx_cache_bytes")
    memory_ok = (
        pageouts_delta in (None, 0)
        and swapouts_delta in (None, 0)
        and (rss_delta is None or rss_delta <= 0)
        and (mlx_cache_delta is None or mlx_cache_delta <= 0)
    )
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"

    if not all_parity_ok:
        decision = "reject_parity"
        reason = "strict parity failed for the FFN-only mx.compile subgraph candidate"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the FFN-only mx.compile subgraph candidate is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory/RSS/MLX-cache/pageout/swap observation also regressed." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the FFN-only mx.compile subgraph candidate "
            "from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "FFN-only mx.compile subgraph candidate is faster than noise, but RSS/MLX-cache/pageout/swap observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "FFN-only mx.compile subgraph candidate is strictly equivalent, disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    return {
        "name": "ffn_subgraph_compile",
        "target_segment": "fc1_swiglu_fc2",
        "target_boundary": "FeedForward-only fc1 -> native SwiGLU -> fc2 branch scheduling",
        "selection_rationale": (
            f"Current segmented median for fc1_swiglu_fc2 is {ffn_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior evidence rejected broad full-block mx.compile and "
            "several FFN layout/dense-dequant routes; this single-variable candidate compiles only the reference "
            "mlp.fc1 -> native SwiGLU -> mlp.fc2 branch while leaving attention, AdaLN, residuals, weights, "
            "quantization, and default generation behavior unchanged."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "default_enabled_after_task": False,
        "production_integrated": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.FeedForward",
            "enabled_flag_for_this_run": "use_ffn_subgraph_compile_candidate",
            "helper": "FeedForward._ffn_subgraph_compiled wrapping FeedForward._ffn_baseline_subgraph",
            "single_variable_guard": "all known block, Attention, and FFN candidates are forced off except use_ffn_subgraph_compile_candidate",
            "lora_path": "falls back to the existing uncompiled LoRA path whenever lora is not None",
            "compiled_scope": "only FeedForward fc1 -> native nn.silu(gate) * value -> fc2",
        },
        "branch_shape_dtype_contract": branch_shape_dtype,
        "branch_parity": branch_parity,
        "branch_parity_ok": branch_parity_ok,
        "compile_first_call_seconds": first_call_seconds,
        "compile_cache_info_after_interleaved": cache_info_after,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "memory_gate_components": {
            "pageouts_delta": pageouts_delta,
            "swapouts_delta": swapouts_delta,
            "rss_kib_delta": rss_delta,
            "mlx_cache_bytes_delta": mlx_cache_delta,
            "requires_no_positive_rss_or_mlx_cache_delta": True,
        },
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default FeedForward flag plus a cached mx.compile wrapper; no weight, quantization, layout, sigma/NFE, or default generation path changes",
            "compile_cost": "first candidate call records FFN subgraph compile/cache setup; warm interleaved samples measure the cached compiled branch",
            "memory": "candidate is rejected unless pageout, swapout, RSS, and MLX cache deltas do not regress",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other sequence shapes may trigger different MLX compile/cache behavior",
            "maintainability": "narrower than the rejected full-block compile route but still requires explicit cache lifecycle if ever retained beyond opt-in experiments",
            "strict_equivalence": "FFN branch output, first compiled full block, and last interleaved full block are checked against uncompiled baselines",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, strict parity, memory/RSS/cache deltas, Amdahl contribution, compile cost, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _attention_projection_input(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    target: str,
) -> mx.array:
    """Return the exact rank-3 input tensor consumed by an Attention projection."""
    shift_msa, scale_msa, _gate_msa, _shift_mlp, _scale_mlp, _gate_mlp = modulation
    h = block.norm1(x) * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
    if target == "qkv_proj":
        return h

    B, S, _ = h.shape
    qkv = block.attn._qkv_project(h).reshape(B, S, block.attn.heads, 3, block.attn.head_dim)
    q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
    q = block.attn.q_norm(q).transpose(0, 2, 1, 3)
    k = block.attn.k_norm(k).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    q = apply_rotary(q, *rotary)
    k = apply_rotary(k, *rotary)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=block.attn.scale, mask=None)
    return out.transpose(0, 2, 1, 3).reshape(B, S, block.attn.heads * block.attn.head_dim).astype(h.dtype)


def _candidate_attention_pre_qkv_contiguous(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe explicit ``mx.contiguous`` materialization of Attention input before ``qkv_proj``."""

    qkv_median = segment_stats.get("qkv_quantized_matmul", {}).get("median_seconds")
    original_block_flags = {
        "use_packed_adaln_gather_candidate": bool(getattr(block, "use_packed_adaln_gather_candidate", False)),
        "use_indexed_adaln_affine_metal_candidate": bool(
            getattr(block, "use_indexed_adaln_affine_metal_candidate", False)
        ),
        "use_indexed_gated_residual_metal_candidate": bool(
            getattr(block, "use_indexed_gated_residual_metal_candidate", False)
        ),
    }
    mlp_candidate_flags = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    original_mlp_flags = {flag: bool(getattr(block.mlp, flag, False)) for flag in mlp_candidate_flags}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    attention_candidate_flags = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_out_layout_metal_candidate",
    )
    original_attn_flags = {flag: bool(getattr(block.attn, flag, False)) for flag in attention_candidate_flags}

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine with prior block, Attention, or FFN routes.
        for flag in original_block_flags:
            setattr(block, flag, False)
        for flag in mlp_candidate_flags:
            setattr(block.mlp, flag, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        for flag in attention_candidate_flags:
            setattr(block.attn, flag, False)
        block.attn.use_pre_qkv_contiguous_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        attn_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        baseline_qkv = block.attn._qkv_project(attn_input)
        hidden_size = int(getattr(block.attn, "_hidden", attn_input.shape[-1]))
        set_candidate(True)
        candidate_input = block.attn._pre_qkv_input(attn_input)
        direct_input = materialize_attention_input_contiguous(attn_input, hidden_size)
        candidate_qkv = block.attn._qkv_project(attn_input)
        mx.eval(attn_input, baseline_qkv, candidate_input, direct_input, candidate_qkv)
        mx.synchronize()
        input_materialization_parity = {
            "method": _diff_stats(attn_input, candidate_input),
            "direct_helper": _diff_stats(attn_input, direct_input),
        }
        qkv_parity = _diff_stats(baseline_qkv, candidate_qkv)
        input_shape_dtype = {
            "attention_input_shape": list(attn_input.shape),
            "attention_input_dtype": str(attn_input.dtype),
            "candidate_input_shape": list(candidate_input.shape),
            "candidate_input_dtype": str(candidate_input.dtype),
            "direct_input_shape": list(direct_input.shape),
            "direct_input_dtype": str(direct_input.dtype),
            "baseline_qkv_shape": list(baseline_qkv.shape),
            "candidate_qkv_shape": list(candidate_qkv.shape),
            "baseline_qkv_dtype": str(baseline_qkv.dtype),
            "candidate_qkv_dtype": str(candidate_qkv.dtype),
            "input_shape_matches_baseline": bool(attn_input.shape == candidate_input.shape == direct_input.shape),
            "input_dtype_matches_baseline": bool(attn_input.dtype == candidate_input.dtype == direct_input.dtype),
            "qkv_shape_matches_baseline": bool(baseline_qkv.shape == candidate_qkv.shape),
            "qkv_dtype_matches_baseline": bool(baseline_qkv.dtype == candidate_qkv.dtype),
            "materialization_api": "mx.contiguous(attention_input) immediately before Attention.qkv_proj",
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 6043,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        for flag, value in original_block_flags.items():
            setattr(block, flag, value)
        for flag, value in original_mlp_flags.items():
            setattr(block.mlp, flag, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        for flag, value in original_attn_flags.items():
            setattr(block.attn, flag, value)

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    input_parity_ok = all(
        stats_["max_abs"] <= args.parity_atol and stats_["rel_l2"] <= args.parity_rel_l2
        for stats_ in input_materialization_parity.values()
    )
    qkv_parity_ok = qkv_parity["max_abs"] <= args.parity_atol and qkv_parity["rel_l2"] <= args.parity_rel_l2
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(input_parity_ok and qkv_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "pre-QKV contiguous materialization changed input, qkv projection, or full-block outputs beyond configured bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says pre-QKV Attention input contiguous materialization is slower than baseline"
        promoted = False
    elif not stable_faster:
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        decision = "reject_unproven_noise"
        reason = (
            "interleaved timing and bootstrap CI do not separate pre-QKV Attention input contiguous materialization "
            "from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "pre-QKV Attention input contiguous materialization is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "pre-QKV Attention input contiguous materialization is strictly equivalent, disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_input_zero = all(
        stats_["max_abs"] == 0.0 and stats_["rel_l2"] == 0.0 for stats_ in input_materialization_parity.values()
    )
    strict_qkv_zero = qkv_parity["max_abs"] == 0.0 and qkv_parity["rel_l2"] == 0.0
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    return {
        "name": "attention_pre_qkv_contiguous",
        "target_segment": "qkv_quantized_matmul",
        "target_boundary": "AdaLN-normalized Attention input materialized with mx.contiguous immediately before Attention.qkv_proj quantized projection",
        "selection_rationale": (
            f"Current segmented median for qkv_quantized_matmul is {qkv_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior Attention probes changed projection rank, QKV "
            "pretranspose scheduling, q/k RMSNorm/RoPE fusion, SDPA input layout, or post-SDPA layout; this "
            "single-variable probe leaves QMM rank, q/k RMSNorm, RoPE, SDPA, out projection, weights, and row "
            "order unchanged while testing whether explicitly materializing the AdaLN-normalized qkv input helps "
            "the dominant QKV QMM scheduler."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {
                "use_pre_qkv_contiguous_candidate": False,
                "use_qkv_2d_projection_candidate": False,
                "use_out_2d_projection_candidate": False,
                "use_qkv_pretranspose_layout_candidate": False,
                "use_qkv_rmsnorm_sdpa_metal_candidate": False,
                "use_qkv_rmsnorm_rotary_sdpa_metal_candidate": False,
                "use_rotary_qk_metal_candidate": False,
                "use_pre_sdpa_contiguous_candidate": False,
                "use_sdpa_out_layout_metal_candidate": False,
            },
            "enabled_flag_for_this_run": "use_pre_qkv_contiguous_candidate",
            "helper": "Attention._pre_qkv_input -> materialize_attention_input_contiguous",
            "single_variable_guard": (
                "prior block, Attention, and FFN candidate flags are forced off during this probe; "
                "only use_pre_qkv_contiguous_candidate is toggled"
            ),
            "lora_path": "falls back to the existing non-contiguous LoRA path whenever lora is not None",
        },
        "input_shape_dtype_contract": input_shape_dtype,
        "input_materialization_parity": input_materialization_parity,
        "input_materialization_parity_ok": input_parity_ok,
        "qkv_projection_parity": qkv_parity,
        "qkv_projection_parity_ok": qkv_parity_ok,
        "strict_input_parity_zero": strict_input_zero,
        "strict_qkv_parity_zero": strict_qkv_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one pure mx.contiguous helper plus one disabled-by-default Attention flag; no weights, quantization, sigma/NFE, cache, projection rank, or default generation path changes",
            "compile_cost": "no mx.compile, custom Metal kernel, or persistent compiler cache is introduced; first call records ordinary MLX lazy/kernel setup cost",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas; candidate may allocate one explicit qkv input buffer",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other shapes and dtypes must be remeasured before promotion",
            "maintainability": "local pre-QKV materialization switch with explicit LoRA fallback; other candidate flags are forced off during this probe",
            "strict_equivalence": "input materialization, qkv projection output, and full-block output are checked against the baseline before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_attention_projection_2d_qmm(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe one Attention projection by flattening its rank-3 input before quantized QMM."""

    qkv_median = segment_stats.get("qkv_quantized_matmul", {}).get("median_seconds")
    out_median = segment_stats.get("out_projection", {}).get("median_seconds")
    if args.attention_2d_target == "auto":
        selected = "qkv_proj" if float(qkv_median or 0.0) >= float(out_median or 0.0) else "out_proj"
        selection_mode = "auto_larger_attention_projection_segment_median"
    else:
        selected = args.attention_2d_target
        selection_mode = "operator_arg"
    target_segment = "qkv_quantized_matmul" if selected == "qkv_proj" else "out_projection"

    original_qkv_flag = bool(getattr(block.attn, "use_qkv_2d_projection_candidate", False))
    original_out_flag = bool(getattr(block.attn, "use_out_2d_projection_candidate", False))

    def set_candidate(enabled: bool) -> None:
        block.attn.use_qkv_2d_projection_candidate = bool(enabled and selected == "qkv_proj")
        block.attn.use_out_2d_projection_candidate = bool(enabled and selected == "out_proj")

    def project_input(inp: mx.array) -> mx.array:
        return block.attn._qkv_project(inp) if selected == "qkv_proj" else block.attn._out_project(inp)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        projection_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, selected)
        baseline_projection = project_input(projection_input)
        set_candidate(True)
        candidate_projection = project_input(projection_input)
        mx.eval(projection_input, baseline_projection, candidate_projection)
        mx.synchronize()
        projection_parity = _diff_stats(baseline_projection, candidate_projection)
        projection_shape_dtype = {
            "input_shape": list(projection_input.shape),
            "input_dtype": str(projection_input.dtype),
            "flattened_shape": [int(projection_input.shape[0]) * int(projection_input.shape[1]), int(projection_input.shape[2])],
            "baseline_output_shape": list(baseline_projection.shape),
            "candidate_output_shape": list(candidate_projection.shape),
            "baseline_output_dtype": str(baseline_projection.dtype),
            "candidate_output_dtype": str(candidate_projection.dtype),
            "restores_original_leading_shape": bool(
                tuple(baseline_projection.shape[:-1])
                == tuple(projection_input.shape[:-1])
                == tuple(candidate_projection.shape[:-1])
            ),
            "dtype_matches_baseline": bool(baseline_projection.dtype == candidate_projection.dtype),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 4219,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        block.attn.use_qkv_2d_projection_candidate = original_qkv_flag
        block.attn.use_out_2d_projection_candidate = original_out_flag

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = (
        projection_parity["max_abs"] <= args.parity_atol
        and projection_parity["rel_l2"] <= args.parity_rel_l2
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "strict-equivalence check failed for the Attention 2D projection QMM candidate"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the Attention 2D projection candidate is slower than rank-3 baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the Attention 2D projection candidate "
            "from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "Attention 2D projection candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "Attention 2D projection candidate is strictly equivalent, disabled by default, memory-clean, "
            "and faster than interleaved rank-3 baseline outside measured noise"
        )
        promoted = True

    return {
        "name": f"attention_{selected}_2d_qmm",
        "target_projection": selected,
        "target_segment": target_segment,
        "target_boundary": "rank-3 [B,S,H] activation before quantized nn.Linear, flattened to [B*S,H] and reshaped back after projection",
        "selection_rationale": (
            f"Selection mode {selection_mode}: current segmented medians are qkv_quantized_matmul={qkv_median} s "
            f"and out_projection={out_median} s, so {selected} is the larger Attention projection boundary. "
            "The artifact's Metal trace summary records MLX 4-bit affine_qmm kernels and materialized "
            "projection buffers; this candidate tests only rank layout at that selected QMM boundary."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {
                "use_qkv_2d_projection_candidate": False,
                "use_out_2d_projection_candidate": False,
            },
            "enabled_flag_for_this_run": (
                "use_qkv_2d_projection_candidate" if selected == "qkv_proj" else "use_out_2d_projection_candidate"
            ),
            "helper": "linear_rank3_input_as_rank2",
            "lora_path": "base projection only; LoRA deltas stay on the existing rank-3 path and default behavior is unchanged",
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": projection_parity["max_abs"] == 0.0 and projection_parity["rel_l2"] == 0.0,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one helper plus disabled-by-default Attention flags; no weights, quantization, sigma/NFE, cache, or default generation path changes",
            "compile_cost": "no mx.compile or persistent compiler cache is introduced by this candidate; first candidate call still records ordinary MLX lazy/kernel setup cost",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; larger S/M/N must be remeasured before release-profile promotion",
            "maintainability": "simple, local rank-layout switch; both qkv and out projection switches remain opt-in and disabled by default",
            "strict_equivalence": "direct projection output and full block output are checked against rank-3 baselines",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, Amdahl contribution, complexity, memory, resolution scaling, and strict parity; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_attention_out_dense_dequant(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe resident dense reconstruction of only ``Attention.out_proj`` quantized weights."""

    out_median = segment_stats.get("out_projection", {}).get("median_seconds")
    qkv_median = segment_stats.get("qkv_quantized_matmul", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    source_is_quantized = getattr(block.attn.out_proj, "scales", None) is not None
    dequantize_available = getattr(mx, "dequantize", None) is not None
    if not source_is_quantized and not args.tiny:
        return {
            "name": "attention_out_dense_dequant",
            "target_segment": "out_projection",
            "target_boundary": "Attention.out_proj quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_out_proj_reconstruction",
            "decision_reason": (
                "real block Attention.out_proj is not an MLX quantized linear with public scales/biases; "
                "dense reconstruction was not attempted"
            ),
            "out_proj_cache_info": block.attn.out_dense_dequant_cache_info(),
        }
    if source_is_quantized and not dequantize_available:
        return {
            "name": "attention_out_dense_dequant",
            "target_segment": "out_projection",
            "target_boundary": "Attention.out_proj quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_out_proj_reconstruction",
            "decision_reason": "mx.dequantize is unavailable, so quantized Attention.out_proj weights cannot be reconstructed safely",
            "out_proj_cache_info": block.attn.out_dense_dequant_cache_info(),
        }

    mlp_candidate_flags = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    block_flag_names = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    attention_candidate_flags = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_out_layout_metal_candidate",
    )
    original_mlp_flags = {name: bool(getattr(block.mlp, name, False)) for name in mlp_candidate_flags}
    original_block_flags = {name: bool(getattr(block, name, False)) for name in block_flag_names}
    original_attention_flags = {name: bool(getattr(block.attn, name, False)) for name in attention_candidate_flags}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: all prior block/MLP/Attention candidate routes are off;
        # only Attention.out_proj swaps from quantized QMM to the resident dense dequantized weight.
        for name in mlp_candidate_flags:
            setattr(block.mlp, name, False)
        for name in block_flag_names:
            setattr(block, name, False)
        for name in attention_candidate_flags:
            setattr(block.attn, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_tile_size
        block.attn.use_out_dense_dequant_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        block.attn.clear_out_dense_dequant_cache()
        set_candidate(False)
        projection_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "out_proj")
        baseline_projection = block.attn._out_project(projection_input)

        set_candidate(True)
        cache_before_projection = block.attn.out_dense_dequant_cache_info()
        dense_projection_started = time.perf_counter()
        candidate_projection = block.attn._out_project(projection_input)
        mx.eval(projection_input, baseline_projection, candidate_projection)
        mx.synchronize()
        first_dense_out_projection_seconds = time.perf_counter() - dense_projection_started
        cache_after_projection = block.attn.out_dense_dequant_cache_info()
        projection_parity = _diff_stats(baseline_projection, candidate_projection)
        projection_shape_dtype = {
            "input_shape": list(projection_input.shape),
            "input_dtype": str(projection_input.dtype),
            "baseline_output_shape": list(baseline_projection.shape),
            "candidate_output_shape": list(candidate_projection.shape),
            "baseline_output_dtype": str(baseline_projection.dtype),
            "candidate_output_dtype": str(candidate_projection.dtype),
            "output_shape_matches_baseline": bool(baseline_projection.shape == candidate_projection.shape),
            "output_dtype_matches_baseline": bool(baseline_projection.dtype == candidate_projection.dtype),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_candidate_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 14203,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        cache_after_interleaved = block.attn.out_dense_dequant_cache_info()
    finally:
        for name, value in original_mlp_flags.items():
            setattr(block.mlp, name, value)
        for name, value in original_block_flags.items():
            setattr(block, name, value)
        for name, value in original_attention_flags.items():
            setattr(block.attn, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_tile_size
        if not original_attention_flags.get("use_out_dense_dequant_candidate", False):
            block.attn.clear_out_dense_dequant_cache()

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = (
        projection_parity["max_abs"] <= args.parity_atol
        and projection_parity["rel_l2"] <= args.parity_rel_l2
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "resident dense-dequantized Attention.out_proj changed projection or full-block outputs beyond configured parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says resident dense-dequantized Attention.out_proj is slower than baseline quantized out_proj"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate resident dense-dequantized Attention.out_proj "
            "from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "resident dense-dequantized Attention.out_proj is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "resident dense-dequantized Attention.out_proj is within parity bounds, disabled by default, "
            "memory-clean, and faster than baseline quantized out_proj outside measured noise"
        )
        promoted = True

    strict_projection_zero = projection_parity["max_abs"] == 0.0 and projection_parity["rel_l2"] == 0.0
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    dense_resident_nbytes = int(cache_after_interleaved.get("dense_nbytes") or 0)
    packed_source_nbytes = int(cache_after_interleaved.get("source_weight_nbytes") or 0) + int(
        cache_after_interleaved.get("source_scales_nbytes") or 0
    ) + int(cache_after_interleaved.get("source_biases_nbytes") or 0)
    return {
        "name": "attention_out_dense_dequant",
        "target_segment": "out_projection",
        "target_boundary": "Attention.out_proj uses a resident dense weight reconstructed from the quantized out_proj pack/scales/biases",
        "selection_rationale": (
            f"Current segmented medians are qkv_quantized_matmul={qkv_median} s, "
            f"mlx_fast_sdpa={sdpa_median} s, and out_projection={out_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior Attention probes changed projection rank, QKV layout, "
            "q/k normalization/RoPE scheduling, SDPA input contiguity, or the post-SDPA layout copy. This probe "
            "changes only the out projection weight representation after the real 4-bit block is resident, replacing "
            "repeated QMM dequant/GEMM dispatch with dense MLX matmul against a cached dequantized weight."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {name: False for name in attention_candidate_flags},
            "enabled_flag_for_this_run": "use_out_dense_dequant_candidate",
            "helper": "Attention._out_dense_dequant_weight + dense_linear_projection",
            "single_variable_guard": "all prior block, MLP, Attention projection-rank, QKV-layout, SDPA-contiguity, and Metal candidates are forced off during this probe",
            "lora_path": "falls back to the existing quantized out_proj base projection whenever lora is not None; LoRA out deltas keep their existing path",
            "cache_lifecycle": "cache key follows the loaded out_proj weight/scales/biases arrays; QuantizedBlockProvider clears it whenever the reusable resident block slot is rebound",
        },
        "out_proj_source_reconstruction": {
            "api": "mx.dequantize(weight, scales, biases, group_size, bits, mode, dtype=scales.dtype)",
            "source_is_quantized": source_is_quantized,
            "dequantize_available": dequantize_available,
            "cache_before_projection": cache_before_projection,
            "cache_after_projection": cache_after_projection,
            "cache_after_interleaved": cache_after_interleaved,
            "first_dense_out_projection_seconds_includes_materialization": first_dense_out_projection_seconds,
            "resident_dense_nbytes": dense_resident_nbytes,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "extra_resident_nbytes_vs_packed_source": dense_resident_nbytes - packed_source_nbytes,
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_candidate_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention flag plus resident dense out_proj cache; no default generation path, sigma/NFE, or non-out_proj weight changes",
            "compile_cost": "no mx.compile or custom Metal kernel is introduced; first dense out projection records dequantization/materialization cost separately from warm interleaved timing",
            "memory": "candidate holds an extra dense out_proj weight while the quantized block is resident; artifact records dense bytes, packed-source bytes, RSS/MLX deltas, pageouts, and swapouts",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence and one resident block; larger shapes and all-block lifecycle need remeasurement before promotion",
            "maintainability": "local Attention.out_proj-only switch with explicit LoRA fallback and provider cache invalidation on block-slot rebinding",
            "strict_equivalence": "direct out projection and full block output are checked against the quantized baseline before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_attention_out_tiled_dense_dequant(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe transient output-channel tiled dense reconstruction of ``Attention.out_proj``."""

    out_median = segment_stats.get("out_projection", {}).get("median_seconds")
    qkv_median = segment_stats.get("qkv_quantized_matmul", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    tile_size = int(args.attention_out_tile_size)
    if tile_size <= 0:
        raise ValueError(f"--attention-out-tile-size must be positive, got {tile_size}")

    source_is_quantized = getattr(block.attn.out_proj, "scales", None) is not None
    dequantize_available = getattr(mx, "dequantize", None) is not None
    if not source_is_quantized and not args.tiny:
        return {
            "name": "attention_out_tiled_dense_dequant",
            "target_segment": "out_projection",
            "target_boundary": "Attention.out_proj output-channel tiled quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_out_proj_reconstruction",
            "decision_reason": (
                "real block Attention.out_proj is not an MLX quantized linear with public scales/biases; "
                "tiled dense reconstruction was not attempted"
            ),
            "out_tiling": block.attn.out_tiled_dense_dequant_info(tile_size),
        }
    if source_is_quantized and not dequantize_available:
        return {
            "name": "attention_out_tiled_dense_dequant",
            "target_segment": "out_projection",
            "target_boundary": "Attention.out_proj output-channel tiled quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_out_proj_reconstruction",
            "decision_reason": "mx.dequantize is unavailable, so quantized Attention.out_proj weight tiles cannot be reconstructed safely",
            "out_tiling": block.attn.out_tiled_dense_dequant_info(tile_size),
        }

    mlp_candidate_flags = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    block_flag_names = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    attention_candidate_flags = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_out_tiled_dense_dequant_candidate",
        "use_qkv_tiled_dense_dequant_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_head_batch_rank3_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    original_mlp_flags = {name: bool(getattr(block.mlp, name, False)) for name in mlp_candidate_flags}
    original_block_flags = {name: bool(getattr(block, name, False)) for name in block_flag_names}
    original_attention_flags = {name: bool(getattr(block.attn, name, False)) for name in attention_candidate_flags}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_fc1_tile_size = int(getattr(block.mlp, "ffn_fc1_tiled_output_channels", 2048))
    original_fc2_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))
    original_qkv_tile_size = int(getattr(block.attn, "qkv_tiled_output_channels", 2048))
    original_out_tile_size = int(getattr(block.attn, "out_tiled_output_channels", 2048))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: all prior block/MLP/Attention routes are off;
        # only Attention.out_proj swaps from quantized QMM to transient dense output tiles.
        for name in mlp_candidate_flags:
            setattr(block.mlp, name, False)
        for name in block_flag_names:
            setattr(block, name, False)
        for name in attention_candidate_flags:
            setattr(block.attn, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = original_fc1_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.attn.qkv_tiled_output_channels = original_qkv_tile_size
        block.attn.out_tiled_output_channels = tile_size
        block.attn.use_out_tiled_dense_dequant_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        projection_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "out_proj")
        baseline_projection = block.attn._out_project(projection_input)

        set_candidate(True)
        tiling_before_projection = block.attn.out_tiled_dense_dequant_info(tile_size)
        tiled_projection_started = time.perf_counter()
        candidate_projection = block.attn._out_project(projection_input)
        fallback_projection = block.attn._out_project(projection_input, lora=object())
        mx.eval(projection_input, baseline_projection, candidate_projection, fallback_projection)
        mx.synchronize()
        first_tiled_out_projection_seconds = time.perf_counter() - tiled_projection_started
        tiling_after_projection = block.attn.out_tiled_dense_dequant_info(tile_size)
        projection_parity = {
            "out_tiled_dense_dequant": _diff_stats(baseline_projection, candidate_projection),
            "lora_fallback_quantized_projection": _diff_stats(baseline_projection, fallback_projection),
        }
        projection_shape_dtype = {
            "input_shape": list(projection_input.shape),
            "input_dtype": str(projection_input.dtype),
            "baseline_projection_shape": list(baseline_projection.shape),
            "candidate_projection_shape": list(candidate_projection.shape),
            "fallback_projection_shape": list(fallback_projection.shape),
            "baseline_projection_dtype": str(baseline_projection.dtype),
            "candidate_projection_dtype": str(candidate_projection.dtype),
            "fallback_projection_dtype": str(fallback_projection.dtype),
            "projection_shape_matches_baseline": bool(baseline_projection.shape == candidate_projection.shape),
            "projection_dtype_matches_baseline": bool(baseline_projection.dtype == candidate_projection.dtype),
            "fallback_shape_dtype_matches_baseline": bool(
                baseline_projection.shape == fallback_projection.shape
                and baseline_projection.dtype == fallback_projection.dtype
            ),
            "out_output_tile_channels": tile_size,
            "row_order_contract": "baseline and candidate both emit Attention.out_proj rows in unchanged output-channel order",
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_candidate_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 17021,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        tiling_after_interleaved = block.attn.out_tiled_dense_dequant_info(tile_size)
    finally:
        for name, value in original_mlp_flags.items():
            setattr(block.mlp, name, value)
        for name, value in original_block_flags.items():
            setattr(block, name, value)
        for name, value in original_attention_flags.items():
            setattr(block.attn, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc1_tiled_output_channels = original_fc1_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.attn.qkv_tiled_output_channels = original_qkv_tile_size
        block.attn.out_tiled_output_channels = original_out_tile_size
        if not original_attention_flags.get("use_out_dense_dequant_candidate", False):
            block.attn.clear_out_dense_dequant_cache()

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "transient tiled dense-dequantized Attention.out_proj changed projection or full-block outputs beyond configured parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says transient tiled dense-dequantized Attention.out_proj is slower than baseline quantized out_proj"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate transient tiled dense-dequantized Attention.out_proj "
            "from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "transient tiled dense-dequantized Attention.out_proj is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "transient tiled dense-dequantized Attention.out_proj is within parity bounds, disabled by default, "
            "does not keep a resident full dense out_proj weight, is memory-clean, and is faster than baseline outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    max_tile_nbytes = int(tiling_after_interleaved.get("max_dense_tile_nbytes") or 0)
    full_dense_nbytes = int(tiling_after_interleaved.get("full_dense_nbytes_if_resident") or 0)
    packed_source_nbytes = int(tiling_after_interleaved.get("packed_quantized_source_nbytes") or 0)
    return {
        "name": "attention_out_tiled_dense_dequant",
        "target_segment": "out_projection",
        "target_boundary": "Attention.out_proj transiently dequantizes output-channel tiles, runs dense matmul per tile, and concatenates in original row order",
        "selection_rationale": (
            f"Current segmented medians are qkv_quantized_matmul={qkv_median} s, "
            f"mlx_fast_sdpa={sdpa_median} s, and out_projection={out_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior out_proj dense-dequant kept a full dense out_proj "
            "matrix resident; this single-variable probe keeps QKV, q/k normalization/RoPE, SDPA, post-SDPA layout, "
            "projection rank, weights, and LoRA behavior unchanged while changing only out_proj weight materialization "
            "to transient output-channel tiles."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {name: False for name in attention_candidate_flags},
            "enabled_flag_for_this_run": "use_out_tiled_dense_dequant_candidate",
            "helper": "tiled_dense_linear_projection + Attention.out_tiled_dense_dequant_info via Attention._out_project",
            "single_variable_guard": "all prior block, MLP, Attention projection-rank, dense-out, QKV-layout, SDPA-contiguity, and Metal candidates are forced off during this probe",
            "lora_path": "falls back to the existing quantized out_proj base projection whenever lora is not None; LoRA out deltas keep their existing path",
            "resident_full_dense_cache": False,
        },
        "out_tiling": {
            "api": "for each output row tile: mx.dequantize(weight[start:stop], scales[start:stop], biases[start:stop], group_size, bits, mode, dtype=scales.dtype); input @ tile.T; concatenate outputs",
            "source_is_quantized": source_is_quantized,
            "dequantize_available": dequantize_available,
            "tile_size_argument": tile_size,
            "tiling_before_projection": tiling_before_projection,
            "tiling_after_projection": tiling_after_projection,
            "tiling_after_interleaved": tiling_after_interleaved,
            "first_tiled_out_projection_seconds_includes_tile_materialization": first_tiled_out_projection_seconds,
            "max_transient_dense_tile_nbytes": max_tile_nbytes,
            "full_dense_nbytes_if_resident": full_dense_nbytes,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "max_transient_tile_nbytes_vs_full_dense": (max_tile_nbytes / full_dense_nbytes) if full_dense_nbytes else None,
            "extra_persistent_dense_nbytes_vs_packed_source": 0,
            "row_order_contract": "original out_proj output rows are sliced into tiles and concatenated in ascending row order",
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_candidate_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "memory_gate_components": {
            "pageouts_delta": pageouts_delta,
            "swapouts_delta": swapouts_delta,
            "mlx_cache_bytes_delta": metrics_delta.get("mlx_cache_bytes"),
            "mlx_peak_bytes_delta": metrics_delta.get("mlx_peak_bytes"),
            "rss_kib_delta": metrics_delta.get("current_rss_kib"),
        },
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention flag plus one tiled dense projection helper already shared with prior probes; no default generation path, sigma/NFE, cache, or projection-rank change",
            "compile_cost": "no mx.compile or custom Metal kernel is introduced; timing includes per-tile dequantization/materialization and synchronization needed to keep tiles transient",
            "memory": "candidate never stores a resident full dense out_proj; artifact records max tile bytes, full-dense equivalent bytes, RSS/MLX deltas, pageouts, and swapouts",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence and chosen output tile size; tile size must be remeasured for other shapes before promotion",
            "maintainability": "local out_proj-only switch with explicit LoRA fallback; prior Attention candidates are forced off during this probe",
            "strict_equivalence": "direct out projection, LoRA fallback base projection, and full block output are checked against the quantized baseline before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_attention_qkv_input_chunked_qmm(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe ``Attention.qkv_proj`` input-feature quantization-group chunked QMM."""

    qkv_median = segment_stats.get("qkv_quantized_matmul", {}).get("median_seconds")
    layout_median = segment_stats.get("qk_rmsnorm_rope_layout", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    chunk_groups = int(args.attention_qkv_input_chunk_groups)
    if chunk_groups <= 0:
        raise ValueError(f"--attention-qkv-input-chunk-groups must be positive, got {chunk_groups}")

    source_is_quantized = getattr(block.attn.qkv_proj, "scales", None) is not None
    quantized_matmul_available = getattr(mx, "quantized_matmul", None) is not None
    initial_info = block.attn.qkv_input_chunked_qmm_info(chunk_groups)
    if not initial_info.get("output_features_match_qkv_contract"):
        return {
            "name": "attention_qkv_input_chunked_qmm",
            "target_segment": "qkv_quantized_matmul",
            "target_boundary": "Attention.qkv_proj input-feature/group chunked quantized QMM",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_qkv_row_contract_mismatch",
            "decision_reason": "Attention.qkv_proj output rows do not match heads * 3 * head_dim, so QKV row-order parity could not be checked safely",
            "qkv_input_chunked_qmm": initial_info,
        }
    if not source_is_quantized and not args.tiny:
        return {
            "name": "attention_qkv_input_chunked_qmm",
            "target_segment": "qkv_quantized_matmul",
            "target_boundary": "Attention.qkv_proj input-feature/group chunked quantized QMM",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_qkv_not_quantized",
            "decision_reason": "real block Attention.qkv_proj is not an MLX quantized linear with public scales; input-group chunked QMM was not attempted",
            "qkv_input_chunked_qmm": initial_info,
        }
    if source_is_quantized and not quantized_matmul_available:
        return {
            "name": "attention_qkv_input_chunked_qmm",
            "target_segment": "qkv_quantized_matmul",
            "target_boundary": "Attention.qkv_proj input-feature/group chunked quantized QMM",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_quantized_matmul_unavailable",
            "decision_reason": "mx.quantized_matmul is unavailable, so quantized Attention.qkv_proj input chunks cannot be run faithfully",
            "qkv_input_chunked_qmm": initial_info,
        }

    mlp_candidate_flags = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc1_split_gate_value_quantized_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_fc2_input_chunked_qmm_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
        "use_ffn_subgraph_compile_candidate",
    )
    block_flag_names = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    attention_candidate_flags = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_out_tiled_dense_dequant_candidate",
        "use_qkv_tiled_dense_dequant_candidate",
        "use_qkv_headgroup_row_sliced_qmm_candidate",
        "use_qkv_input_chunked_qmm_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_head_batch_rank3_candidate",
        "use_sdpa_headgroup_split_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    original_mlp_flags = {name: bool(getattr(block.mlp, name, False)) for name in mlp_candidate_flags}
    original_block_flags = {name: bool(getattr(block, name, False)) for name in block_flag_names}
    original_attention_flags = {name: bool(getattr(block.attn, name, False)) for name in attention_candidate_flags}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_fc2_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))
    original_qkv_tile_size = int(getattr(block.attn, "qkv_tiled_output_channels", 2048))
    original_headgroup_size = int(getattr(block.attn, "qkv_headgroup_heads_per_slice", 8))
    original_qkv_input_chunk_groups = int(getattr(block.attn, "qkv_input_chunk_groups", 42))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: all prior block/MLP/Attention routes are off;
        # only qkv_proj input quantization groups are split across partial QMM launches.
        for name in mlp_candidate_flags:
            setattr(block.mlp, name, False)
        for name in block_flag_names:
            setattr(block, name, False)
        for name in attention_candidate_flags:
            setattr(block.attn, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.attn.qkv_tiled_output_channels = original_qkv_tile_size
        block.attn.qkv_headgroup_heads_per_slice = original_headgroup_size
        block.attn.qkv_input_chunk_groups = chunk_groups
        block.attn.use_qkv_input_chunked_qmm_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        projection_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        baseline_projection = block.attn._qkv_project(projection_input)
        baseline_raw_qkv = baseline_projection.reshape(
            projection_input.shape[0], projection_input.shape[1], block.attn.heads, 3, block.attn.head_dim
        )
        baseline_q, baseline_k, baseline_v = block.attn._qkv_sdpa_tensors(projection_input)

        set_candidate(True)
        qmm_before_projection = block.attn.qkv_input_chunked_qmm_info(chunk_groups)
        projection_started = time.perf_counter()
        candidate_projection = block.attn._qkv_project(projection_input)
        fallback_projection = block.attn._qkv_project(projection_input, lora=object())
        mx.eval(projection_input, baseline_projection, candidate_projection, fallback_projection)
        mx.synchronize()
        first_chunked_qkv_projection_seconds = time.perf_counter() - projection_started
        candidate_raw_qkv = candidate_projection.reshape(
            projection_input.shape[0], projection_input.shape[1], block.attn.heads, 3, block.attn.head_dim
        )
        candidate_q, candidate_k, candidate_v = block.attn._qkv_sdpa_tensors(projection_input)
        mx.eval(
            baseline_raw_qkv,
            candidate_raw_qkv,
            baseline_q,
            baseline_k,
            baseline_v,
            candidate_q,
            candidate_k,
            candidate_v,
        )
        mx.synchronize()
        qmm_after_projection = block.attn.qkv_input_chunked_qmm_info(chunk_groups)
        projection_parity = {
            "qkv_input_chunked_qmm": _diff_stats(baseline_projection, candidate_projection),
            "lora_fallback_fused_qkv": _diff_stats(baseline_projection, fallback_projection),
            "raw_q_rows": _diff_stats(baseline_raw_qkv[:, :, :, 0], candidate_raw_qkv[:, :, :, 0]),
            "raw_k_rows": _diff_stats(baseline_raw_qkv[:, :, :, 1], candidate_raw_qkv[:, :, :, 1]),
            "raw_v_rows": _diff_stats(baseline_raw_qkv[:, :, :, 2], candidate_raw_qkv[:, :, :, 2]),
            "sdpa_q_after_q_norm": _diff_stats(baseline_q, candidate_q),
            "sdpa_k_after_k_norm": _diff_stats(baseline_k, candidate_k),
            "sdpa_v_layout": _diff_stats(baseline_v, candidate_v),
        }
        projection_shape_dtype = {
            "input_shape": list(projection_input.shape),
            "input_dtype": str(projection_input.dtype),
            "baseline_projection_shape": list(baseline_projection.shape),
            "candidate_projection_shape": list(candidate_projection.shape),
            "fallback_projection_shape": list(fallback_projection.shape),
            "baseline_projection_dtype": str(baseline_projection.dtype),
            "candidate_projection_dtype": str(candidate_projection.dtype),
            "fallback_projection_dtype": str(fallback_projection.dtype),
            "raw_qkv_shape": list(baseline_raw_qkv.shape),
            "candidate_raw_qkv_shape": list(candidate_raw_qkv.shape),
            "raw_qkv_dtype": str(baseline_raw_qkv.dtype),
            "candidate_raw_qkv_dtype": str(candidate_raw_qkv.dtype),
            "baseline_q_shape": list(baseline_q.shape),
            "candidate_q_shape": list(candidate_q.shape),
            "baseline_k_shape": list(baseline_k.shape),
            "candidate_k_shape": list(candidate_k.shape),
            "baseline_v_shape": list(baseline_v.shape),
            "candidate_v_shape": list(candidate_v.shape),
            "baseline_q_dtype": str(baseline_q.dtype),
            "candidate_q_dtype": str(candidate_q.dtype),
            "baseline_k_dtype": str(baseline_k.dtype),
            "candidate_k_dtype": str(candidate_k.dtype),
            "baseline_v_dtype": str(baseline_v.dtype),
            "candidate_v_dtype": str(candidate_v.dtype),
            "projection_shape_matches_baseline": bool(baseline_projection.shape == candidate_projection.shape),
            "projection_dtype_matches_baseline": bool(baseline_projection.dtype == candidate_projection.dtype),
            "fallback_shape_dtype_matches_baseline": bool(
                baseline_projection.shape == fallback_projection.shape and baseline_projection.dtype == fallback_projection.dtype
            ),
            "raw_qkv_shape_matches_baseline": bool(baseline_raw_qkv.shape == candidate_raw_qkv.shape),
            "raw_qkv_dtype_matches_baseline": bool(baseline_raw_qkv.dtype == candidate_raw_qkv.dtype),
            "q_shape_dtype_matches_baseline": bool(baseline_q.shape == candidate_q.shape and baseline_q.dtype == candidate_q.dtype),
            "k_shape_dtype_matches_baseline": bool(baseline_k.shape == candidate_k.shape and baseline_k.dtype == candidate_k.dtype),
            "v_shape_dtype_matches_baseline": bool(baseline_v.shape == candidate_v.shape and baseline_v.dtype == candidate_v.dtype),
            "qkv_input_chunk_groups": chunk_groups,
            "qkv_input_chunk_count": qmm_after_projection.get("chunk_count"),
            "row_order_contract": "baseline and candidate both reshape to [B,S,heads,3,head_dim] from unchanged per-head-interleaved output rows",
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_candidate_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 27191,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        qmm_after_interleaved = block.attn.qkv_input_chunked_qmm_info(chunk_groups)
    finally:
        for name, value in original_mlp_flags.items():
            setattr(block.mlp, name, value)
        for name, value in original_block_flags.items():
            setattr(block, name, value)
        for name, value in original_attention_flags.items():
            setattr(block.attn, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.attn.qkv_tiled_output_channels = original_qkv_tile_size
        block.attn.qkv_headgroup_heads_per_slice = original_headgroup_size
        block.attn.qkv_input_chunk_groups = original_qkv_input_chunk_groups

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "input/group chunked quantized Attention.qkv_proj changed QKV or full-block outputs beyond configured parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says input/group chunked quantized Attention.qkv_proj is slower than baseline fused qkv_proj"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate input/group chunked quantized Attention.qkv_proj "
            "from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "input/group chunked quantized Attention.qkv_proj is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "input/group chunked quantized Attention.qkv_proj is within parity bounds, disabled by default, "
            "preserves original QKV row order without dense dequantization, is memory-clean, and is faster than baseline outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    chunk_count = int(qmm_after_interleaved.get("chunk_count") or 0)
    partial_output_nbytes = int(getattr(candidate_projection, "nbytes", 0)) if "candidate_projection" in locals() else None
    return {
        "name": "attention_qkv_input_chunked_qmm",
        "target_segment": "qkv_quantized_matmul",
        "target_boundary": "Attention.qkv_proj input quantization groups are split into complete group chunks, projected by partial QMMs, and accumulated without changing output row order",
        "selection_rationale": (
            f"Current segmented medians are qkv_quantized_matmul={qkv_median} s, "
            f"qk_rmsnorm_rope_layout={layout_median} s, and mlx_fast_sdpa={sdpa_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior Attention probes changed projection rank, input contiguity, "
            "output-row/headgroup slicing, dense dequantization, QKV layout, q/k normalization/RoPE scheduling, or SDPA layout. "
            "This probe changes only qkv_proj QMM scheduling/cache shape by slicing the input feature dimension on complete "
            "quantization-group boundaries and accumulating partial outputs without changing weights, output rows, or QKV order."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {name: False for name in attention_candidate_flags},
            "enabled_flag_for_this_run": "use_qkv_input_chunked_qmm_candidate",
            "helper": "quantized_matmul_input_chunked_projection + Attention.qkv_input_chunked_qmm_info via Attention._qkv_project",
            "single_variable_guard": "all prior block, MLP, Attention projection-rank, dense-dequant, QKV-layout, SDPA-contiguity, and Metal candidates are forced off during this probe",
            "lora_path": "falls back to the existing fused qkv_proj base projection whenever lora is not None; LoRA q/k/v deltas keep their existing path",
            "dense_dequantization": False,
            "resident_full_dense_cache": False,
        },
        "qkv_input_chunked_qmm": {
            "api": "for each complete input group chunk: mx.quantized_matmul(qkv_input[..., feature_start:feature_stop], weight[:, packed_start:packed_stop], sliced scales/biases, transpose=True); sum partial outputs; add learned bias once if present",
            "source_is_quantized": source_is_quantized,
            "quantized_matmul_available": quantized_matmul_available,
            "chunk_groups_argument": chunk_groups,
            "qmm_before_projection": qmm_before_projection,
            "qmm_after_projection": qmm_after_projection,
            "qmm_after_interleaved": qmm_after_interleaved,
            "first_chunked_qkv_projection_seconds_includes_partial_qmm_launches": first_chunked_qkv_projection_seconds,
            "partial_qmm_launches_per_qkv": chunk_count,
            "partial_output_nbytes_each": partial_output_nbytes,
            "learned_bias_added_once": True,
            "dense_weight_materialization": False,
            "row_order_contract": "input chunks accumulate into the full qkv output tensor, so original per-head-interleaved rows [h0:q,k,v][h1:q,k,v]... are not reindexed",
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_candidate_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
            "known_risk": "partial QMM accumulation can change floating-point reduction order versus the monolithic qkv_proj QMM",
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "memory_gate_components": {
            "pageouts_delta": pageouts_delta,
            "swapouts_delta": swapouts_delta,
            "mlx_cache_bytes_delta": metrics_delta.get("mlx_cache_bytes"),
            "mlx_peak_bytes_delta": metrics_delta.get("mlx_peak_bytes"),
            "rss_kib_delta": metrics_delta.get("current_rss_kib"),
        },
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention flag plus a shared packed-weight input-group slicing helper; no default generation path, sigma/NFE, cache, dense dequantization, or projection-rank change",
            "compile_cost": "no mx.compile or custom Metal kernel is introduced; timing includes multiple qkv quantized_matmul launches and partial-output additions",
            "memory": "candidate does not materialize a dense qkv_proj weight; artifact records RSS/MLX deltas, pageouts, swapouts, chunk metadata, and partial-output size",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence, one real 4-bit resident block, and chosen input chunk groups; other shapes must be remeasured before promotion",
            "maintainability": "local qkv_proj-only switch with explicit LoRA fallback; prior Attention candidates are forced off and not combined",
            "strict_equivalence": "raw QKV projection, LoRA fallback fused projection, reshaped q/k/v row order, post-qk-norm SDPA tensors, and full block output are checked before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_attention_qkv_headgroup_row_sliced_qmm(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe ``Attention.qkv_proj`` as whole-head output-row-sliced quantized QMMs."""

    qkv_median = segment_stats.get("qkv_quantized_matmul", {}).get("median_seconds")
    layout_median = segment_stats.get("qk_rmsnorm_rope_layout", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    headgroup_size = int(args.attention_qkv_headgroup_size)
    if headgroup_size <= 0:
        raise ValueError(f"--attention-qkv-headgroup-size must be positive, got {headgroup_size}")

    source_is_quantized = getattr(block.attn.qkv_proj, "scales", None) is not None
    quantized_matmul_available = getattr(mx, "quantized_matmul", None) is not None
    initial_info = block.attn.qkv_headgroup_row_sliced_qmm_info(headgroup_size)
    if not initial_info.get("output_features_match_qkv_contract"):
        return {
            "name": "attention_qkv_headgroup_row_sliced_qmm",
            "target_segment": "qkv_quantized_matmul",
            "target_boundary": "Attention.qkv_proj whole-head output-row-sliced quantized QMM",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_qkv_row_contract_mismatch",
            "decision_reason": "Attention.qkv_proj output rows do not match heads * 3 * head_dim, so whole-head row slicing would not preserve the QKV contract",
            "qkv_headgroup_row_sliced_qmm": initial_info,
        }
    if not source_is_quantized and not args.tiny:
        return {
            "name": "attention_qkv_headgroup_row_sliced_qmm",
            "target_segment": "qkv_quantized_matmul",
            "target_boundary": "Attention.qkv_proj whole-head output-row-sliced quantized QMM",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_qkv_not_quantized",
            "decision_reason": "real block Attention.qkv_proj is not an MLX quantized linear with public scales; quantized row-sliced QMM was not attempted",
            "qkv_headgroup_row_sliced_qmm": initial_info,
        }
    if source_is_quantized and not quantized_matmul_available:
        return {
            "name": "attention_qkv_headgroup_row_sliced_qmm",
            "target_segment": "qkv_quantized_matmul",
            "target_boundary": "Attention.qkv_proj whole-head output-row-sliced quantized QMM",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_quantized_matmul_unavailable",
            "decision_reason": "mx.quantized_matmul is unavailable, so quantized Attention.qkv_proj output-row slices cannot be projected safely",
            "qkv_headgroup_row_sliced_qmm": initial_info,
        }

    mlp_candidate_flags = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc1_split_gate_value_quantized_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_fc2_input_chunked_qmm_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
        "use_ffn_subgraph_compile_candidate",
    )
    block_flag_names = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    attention_candidate_flags = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_out_tiled_dense_dequant_candidate",
        "use_qkv_tiled_dense_dequant_candidate",
        "use_qkv_headgroup_row_sliced_qmm_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_head_batch_rank3_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    original_mlp_flags = {name: bool(getattr(block.mlp, name, False)) for name in mlp_candidate_flags}
    original_block_flags = {name: bool(getattr(block, name, False)) for name in block_flag_names}
    original_attention_flags = {name: bool(getattr(block.attn, name, False)) for name in attention_candidate_flags}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_fc2_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))
    original_qkv_tile_size = int(getattr(block.attn, "qkv_tiled_output_channels", 2048))
    original_headgroup_size = int(getattr(block.attn, "qkv_headgroup_heads_per_slice", 8))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: all prior block/MLP/Attention routes are off;
        # only qkv_proj output rows are split into complete-head groups and projected by QMM.
        for name in mlp_candidate_flags:
            setattr(block.mlp, name, False)
        for name in block_flag_names:
            setattr(block, name, False)
        for name in attention_candidate_flags:
            setattr(block.attn, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.attn.qkv_tiled_output_channels = original_qkv_tile_size
        block.attn.qkv_headgroup_heads_per_slice = headgroup_size
        block.attn.use_qkv_headgroup_row_sliced_qmm_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        projection_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        baseline_projection = block.attn._qkv_project(projection_input)
        baseline_raw_qkv = baseline_projection.reshape(
            projection_input.shape[0], projection_input.shape[1], block.attn.heads, 3, block.attn.head_dim
        )
        baseline_q, baseline_k, baseline_v = block.attn._qkv_sdpa_tensors(projection_input)

        set_candidate(True)
        headgroup_before_projection = block.attn.qkv_headgroup_row_sliced_qmm_info(headgroup_size)
        projection_started = time.perf_counter()
        candidate_projection = block.attn._qkv_project(projection_input)
        fallback_projection = block.attn._qkv_project(projection_input, lora=object())
        mx.eval(projection_input, baseline_projection, candidate_projection, fallback_projection)
        mx.synchronize()
        first_headgroup_qkv_projection_seconds = time.perf_counter() - projection_started
        candidate_raw_qkv = candidate_projection.reshape(
            projection_input.shape[0], projection_input.shape[1], block.attn.heads, 3, block.attn.head_dim
        )
        candidate_q, candidate_k, candidate_v = block.attn._qkv_sdpa_tensors(projection_input)
        mx.eval(
            baseline_raw_qkv,
            candidate_raw_qkv,
            baseline_q,
            baseline_k,
            baseline_v,
            candidate_q,
            candidate_k,
            candidate_v,
        )
        mx.synchronize()
        headgroup_after_projection = block.attn.qkv_headgroup_row_sliced_qmm_info(headgroup_size)
        projection_parity = {
            "qkv_headgroup_row_sliced": _diff_stats(baseline_projection, candidate_projection),
            "lora_fallback_fused_qkv": _diff_stats(baseline_projection, fallback_projection),
            "raw_q_rows": _diff_stats(baseline_raw_qkv[:, :, :, 0], candidate_raw_qkv[:, :, :, 0]),
            "raw_k_rows": _diff_stats(baseline_raw_qkv[:, :, :, 1], candidate_raw_qkv[:, :, :, 1]),
            "raw_v_rows": _diff_stats(baseline_raw_qkv[:, :, :, 2], candidate_raw_qkv[:, :, :, 2]),
            "sdpa_q_after_q_norm": _diff_stats(baseline_q, candidate_q),
            "sdpa_k_after_k_norm": _diff_stats(baseline_k, candidate_k),
            "sdpa_v_layout": _diff_stats(baseline_v, candidate_v),
        }
        projection_shape_dtype = {
            "input_shape": list(projection_input.shape),
            "input_dtype": str(projection_input.dtype),
            "baseline_projection_shape": list(baseline_projection.shape),
            "candidate_projection_shape": list(candidate_projection.shape),
            "fallback_projection_shape": list(fallback_projection.shape),
            "baseline_projection_dtype": str(baseline_projection.dtype),
            "candidate_projection_dtype": str(candidate_projection.dtype),
            "fallback_projection_dtype": str(fallback_projection.dtype),
            "raw_qkv_shape": list(baseline_raw_qkv.shape),
            "candidate_raw_qkv_shape": list(candidate_raw_qkv.shape),
            "raw_qkv_dtype": str(baseline_raw_qkv.dtype),
            "candidate_raw_qkv_dtype": str(candidate_raw_qkv.dtype),
            "baseline_q_shape": list(baseline_q.shape),
            "candidate_q_shape": list(candidate_q.shape),
            "baseline_k_shape": list(baseline_k.shape),
            "candidate_k_shape": list(candidate_k.shape),
            "baseline_v_shape": list(baseline_v.shape),
            "candidate_v_shape": list(candidate_v.shape),
            "baseline_q_dtype": str(baseline_q.dtype),
            "candidate_q_dtype": str(candidate_q.dtype),
            "baseline_k_dtype": str(baseline_k.dtype),
            "candidate_k_dtype": str(candidate_k.dtype),
            "baseline_v_dtype": str(baseline_v.dtype),
            "candidate_v_dtype": str(candidate_v.dtype),
            "projection_shape_matches_baseline": bool(baseline_projection.shape == candidate_projection.shape),
            "projection_dtype_matches_baseline": bool(baseline_projection.dtype == candidate_projection.dtype),
            "fallback_shape_dtype_matches_baseline": bool(
                baseline_projection.shape == fallback_projection.shape and baseline_projection.dtype == fallback_projection.dtype
            ),
            "raw_qkv_shape_matches_baseline": bool(baseline_raw_qkv.shape == candidate_raw_qkv.shape),
            "raw_qkv_dtype_matches_baseline": bool(baseline_raw_qkv.dtype == candidate_raw_qkv.dtype),
            "q_shape_dtype_matches_baseline": bool(baseline_q.shape == candidate_q.shape and baseline_q.dtype == candidate_q.dtype),
            "k_shape_dtype_matches_baseline": bool(baseline_k.shape == candidate_k.shape and baseline_k.dtype == candidate_k.dtype),
            "v_shape_dtype_matches_baseline": bool(baseline_v.shape == candidate_v.shape and baseline_v.dtype == candidate_v.dtype),
            "qkv_headgroup_heads_per_slice": headgroup_size,
            "row_order_contract": "baseline and candidate both reshape to [B,S,heads,3,head_dim] from unchanged per-head-interleaved output rows",
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_candidate_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 23117,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        headgroup_after_interleaved = block.attn.qkv_headgroup_row_sliced_qmm_info(headgroup_size)
    finally:
        for name, value in original_mlp_flags.items():
            setattr(block.mlp, name, value)
        for name, value in original_block_flags.items():
            setattr(block, name, value)
        for name, value in original_attention_flags.items():
            setattr(block.attn, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.attn.qkv_tiled_output_channels = original_qkv_tile_size
        block.attn.qkv_headgroup_heads_per_slice = original_headgroup_size

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "whole-head row-sliced quantized Attention.qkv_proj changed QKV or full-block outputs beyond configured parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says whole-head row-sliced quantized Attention.qkv_proj is slower than baseline fused qkv_proj"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate whole-head row-sliced quantized Attention.qkv_proj "
            "from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "whole-head row-sliced quantized Attention.qkv_proj is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "whole-head row-sliced quantized Attention.qkv_proj is within parity bounds, disabled by default, "
            "preserves original QKV row order without dense dequantization, is memory-clean, and is faster than baseline outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    projection_count = int(headgroup_after_interleaved.get("separate_projection_count") or 0)
    return {
        "name": "attention_qkv_headgroup_row_sliced_qmm",
        "target_segment": "qkv_quantized_matmul",
        "target_boundary": "Attention.qkv_proj output rows are split into contiguous complete-head groups, projected by row-sliced QMM, and concatenated in original row order",
        "selection_rationale": (
            f"Current segmented medians are qkv_quantized_matmul={qkv_median} s, "
            f"qk_rmsnorm_rope_layout={layout_median} s, and mlx_fast_sdpa={sdpa_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior Attention probes changed projection rank, input contiguity, "
            "dense dequantization, QKV layout, q/k normalization/RoPE scheduling, or SDPA layout. This probe changes only "
            "qkv_proj QMM scheduling/cache shape by slicing output rows at whole-head boundaries and concatenating without changing row order or math."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {name: False for name in attention_candidate_flags},
            "enabled_flag_for_this_run": "use_qkv_headgroup_row_sliced_qmm_candidate",
            "helper": "qkv_headgroup_row_sliced_projection + linear_output_row_slice_projection via Attention._qkv_project",
            "single_variable_guard": "all prior block, MLP, Attention projection-rank, dense-dequant, QKV-layout, SDPA-contiguity, and Metal candidates are forced off during this probe",
            "lora_path": "falls back to the existing fused qkv_proj base projection whenever lora is not None; LoRA q/k/v deltas keep their existing path",
            "dense_dequantization": False,
        },
        "qkv_headgroup_row_sliced_qmm": {
            "api": "for each complete-head output row group [head_start*3D:head_stop*3D], call mx.quantized_matmul on sliced packed qkv rows/scales/biases, then concatenate groups on axis -1",
            "source_is_quantized": source_is_quantized,
            "quantized_matmul_available": quantized_matmul_available,
            "headgroup_size_argument": headgroup_size,
            "headgroup_before_projection": headgroup_before_projection,
            "headgroup_after_projection": headgroup_after_projection,
            "headgroup_after_interleaved": headgroup_after_interleaved,
            "first_headgroup_qkv_projection_seconds_includes_projection_launches_and_concat": first_headgroup_qkv_projection_seconds,
            "separate_projection_count": projection_count,
            "row_order_contract": "original qkv_proj output rows are per-head interleaved [h0:q,k,v][h1:q,k,v]...; each slice covers complete heads and concatenation is ascending row order",
            "dense_dequantization": False,
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_candidate_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "memory_gate_components": {
            "pageouts_delta": pageouts_delta,
            "swapouts_delta": swapouts_delta,
            "mlx_cache_bytes_delta": metrics_delta.get("mlx_cache_bytes"),
            "mlx_peak_bytes_delta": metrics_delta.get("mlx_peak_bytes"),
            "rss_kib_delta": metrics_delta.get("current_rss_kib"),
        },
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention flag plus a small whole-head qkv row-slice helper; no default generation path, sigma/NFE, cache, dense dequantization, or projection-rank change",
            "compile_cost": "no mx.compile or custom Metal kernel is introduced; timing includes multiple qkv quantized_matmul launches and final concatenation",
            "memory": "candidate does not materialize a dense qkv_proj weight; artifact records RSS/MLX deltas, pageouts, and swapouts",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence, one real 4-bit resident block, and chosen heads-per-slice; other shapes must be remeasured before promotion",
            "maintainability": "local qkv_proj-only switch with explicit LoRA fallback; prior Attention candidates are forced off and not combined",
            "strict_equivalence": "raw QKV projection, LoRA fallback fused projection, reshaped q/k/v row order, post-qk-norm SDPA tensors, and full block output are checked before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_attention_qkv_tiled_dense_dequant(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe transient output-channel tiled dense reconstruction of ``Attention.qkv_proj``."""

    qkv_median = segment_stats.get("qkv_quantized_matmul", {}).get("median_seconds")
    layout_median = segment_stats.get("qk_rmsnorm_rope_layout", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    tile_size = int(args.attention_qkv_tile_size)
    if tile_size <= 0:
        raise ValueError(f"--attention-qkv-tile-size must be positive, got {tile_size}")

    source_is_quantized = getattr(block.attn.qkv_proj, "scales", None) is not None
    dequantize_available = getattr(mx, "dequantize", None) is not None
    if not source_is_quantized and not args.tiny:
        return {
            "name": "attention_qkv_tiled_dense_dequant",
            "target_segment": "qkv_quantized_matmul",
            "target_boundary": "Attention.qkv_proj output-channel tiled quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_qkv_reconstruction",
            "decision_reason": (
                "real block Attention.qkv_proj is not an MLX quantized linear with public scales/biases; "
                "tiled dense reconstruction was not attempted"
            ),
            "qkv_tiling": block.attn.qkv_tiled_dense_dequant_info(tile_size),
        }
    if source_is_quantized and not dequantize_available:
        return {
            "name": "attention_qkv_tiled_dense_dequant",
            "target_segment": "qkv_quantized_matmul",
            "target_boundary": "Attention.qkv_proj output-channel tiled quantized weight reconstruction",
            "candidate_available": False,
            "promote": False,
            "decision": "blocked_qkv_reconstruction",
            "decision_reason": "mx.dequantize is unavailable, so quantized Attention.qkv_proj weight tiles cannot be reconstructed safely",
            "qkv_tiling": block.attn.qkv_tiled_dense_dequant_info(tile_size),
        }

    mlp_candidate_flags = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
        "use_ffn_subgraph_compile_candidate",
    )
    block_flag_names = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    attention_candidate_flags = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_out_tiled_dense_dequant_candidate",
        "use_qkv_tiled_dense_dequant_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_head_batch_rank3_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    original_mlp_flags = {name: bool(getattr(block.mlp, name, False)) for name in mlp_candidate_flags}
    original_block_flags = {name: bool(getattr(block, name, False)) for name in block_flag_names}
    original_attention_flags = {name: bool(getattr(block.attn, name, False)) for name in attention_candidate_flags}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_fc2_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))
    original_qkv_tile_size = int(getattr(block.attn, "qkv_tiled_output_channels", 2048))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: all prior block/MLP/Attention routes are off;
        # only Attention.qkv_proj swaps from quantized QMM to transient dense output tiles.
        for name in mlp_candidate_flags:
            setattr(block.mlp, name, False)
        for name in block_flag_names:
            setattr(block, name, False)
        for name in attention_candidate_flags:
            setattr(block.attn, name, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.attn.qkv_tiled_output_channels = tile_size
        block.attn.use_qkv_tiled_dense_dequant_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        projection_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        baseline_projection = block.attn._qkv_project(projection_input)
        baseline_raw_qkv = baseline_projection.reshape(
            projection_input.shape[0], projection_input.shape[1], block.attn.heads, 3, block.attn.head_dim
        )
        baseline_q, baseline_k, baseline_v = block.attn._qkv_sdpa_tensors(projection_input)

        set_candidate(True)
        tiling_before_projection = block.attn.qkv_tiled_dense_dequant_info(tile_size)
        tiled_projection_started = time.perf_counter()
        candidate_projection = block.attn._qkv_project(projection_input)
        mx.eval(projection_input, baseline_projection, candidate_projection)
        mx.synchronize()
        first_tiled_qkv_projection_seconds = time.perf_counter() - tiled_projection_started
        candidate_raw_qkv = candidate_projection.reshape(
            projection_input.shape[0], projection_input.shape[1], block.attn.heads, 3, block.attn.head_dim
        )
        candidate_q, candidate_k, candidate_v = block.attn._qkv_sdpa_tensors(projection_input)
        mx.eval(
            baseline_raw_qkv,
            candidate_raw_qkv,
            baseline_q,
            baseline_k,
            baseline_v,
            candidate_q,
            candidate_k,
            candidate_v,
        )
        mx.synchronize()
        tiling_after_projection = block.attn.qkv_tiled_dense_dequant_info(tile_size)
        projection_parity = {
            "qkv_tiled_dense_dequant": _diff_stats(baseline_projection, candidate_projection),
            "raw_q_rows": _diff_stats(baseline_raw_qkv[:, :, :, 0], candidate_raw_qkv[:, :, :, 0]),
            "raw_k_rows": _diff_stats(baseline_raw_qkv[:, :, :, 1], candidate_raw_qkv[:, :, :, 1]),
            "raw_v_rows": _diff_stats(baseline_raw_qkv[:, :, :, 2], candidate_raw_qkv[:, :, :, 2]),
            "sdpa_q_after_q_norm": _diff_stats(baseline_q, candidate_q),
            "sdpa_k_after_k_norm": _diff_stats(baseline_k, candidate_k),
            "sdpa_v_layout": _diff_stats(baseline_v, candidate_v),
        }
        projection_shape_dtype = {
            "input_shape": list(projection_input.shape),
            "input_dtype": str(projection_input.dtype),
            "baseline_projection_shape": list(baseline_projection.shape),
            "candidate_projection_shape": list(candidate_projection.shape),
            "baseline_projection_dtype": str(baseline_projection.dtype),
            "candidate_projection_dtype": str(candidate_projection.dtype),
            "raw_qkv_shape": list(baseline_raw_qkv.shape),
            "candidate_raw_qkv_shape": list(candidate_raw_qkv.shape),
            "raw_qkv_dtype": str(baseline_raw_qkv.dtype),
            "candidate_raw_qkv_dtype": str(candidate_raw_qkv.dtype),
            "baseline_q_shape": list(baseline_q.shape),
            "candidate_q_shape": list(candidate_q.shape),
            "baseline_k_shape": list(baseline_k.shape),
            "candidate_k_shape": list(candidate_k.shape),
            "baseline_v_shape": list(baseline_v.shape),
            "candidate_v_shape": list(candidate_v.shape),
            "baseline_q_dtype": str(baseline_q.dtype),
            "candidate_q_dtype": str(candidate_q.dtype),
            "baseline_k_dtype": str(baseline_k.dtype),
            "candidate_k_dtype": str(candidate_k.dtype),
            "baseline_v_dtype": str(baseline_v.dtype),
            "candidate_v_dtype": str(candidate_v.dtype),
            "projection_shape_matches_baseline": bool(baseline_projection.shape == candidate_projection.shape),
            "projection_dtype_matches_baseline": bool(baseline_projection.dtype == candidate_projection.dtype),
            "raw_qkv_shape_matches_baseline": bool(baseline_raw_qkv.shape == candidate_raw_qkv.shape),
            "raw_qkv_dtype_matches_baseline": bool(baseline_raw_qkv.dtype == candidate_raw_qkv.dtype),
            "q_shape_dtype_matches_baseline": bool(baseline_q.shape == candidate_q.shape and baseline_q.dtype == candidate_q.dtype),
            "k_shape_dtype_matches_baseline": bool(baseline_k.shape == candidate_k.shape and baseline_k.dtype == candidate_k.dtype),
            "v_shape_dtype_matches_baseline": bool(baseline_v.shape == candidate_v.shape and baseline_v.dtype == candidate_v.dtype),
            "qkv_output_tile_channels": tile_size,
            "row_order_contract": "baseline and candidate both reshape to [B,S,heads,3,head_dim] from the unchanged output-row order",
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_candidate_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 16057,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
        tiling_after_interleaved = block.attn.qkv_tiled_dense_dequant_info(tile_size)
    finally:
        for name, value in original_mlp_flags.items():
            setattr(block.mlp, name, value)
        for name, value in original_block_flags.items():
            setattr(block, name, value)
        for name, value in original_attention_flags.items():
            setattr(block.attn, name, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size
        block.attn.qkv_tiled_output_channels = original_qkv_tile_size

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    projection_parity_ok = all(
        projection_stats["max_abs"] <= args.parity_atol
        and projection_stats["rel_l2"] <= args.parity_rel_l2
        for projection_stats in projection_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "transient tiled dense-dequantized Attention.qkv_proj changed QKV or full-block outputs beyond configured parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says transient tiled dense-dequantized Attention.qkv_proj is slower than baseline quantized qkv_proj"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate transient tiled dense-dequantized Attention.qkv_proj "
            "from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "transient tiled dense-dequantized Attention.qkv_proj is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "transient tiled dense-dequantized Attention.qkv_proj is within parity bounds, disabled by default, "
            "preserves original QKV row order without a resident full dense QKV weight, is memory-clean, "
            "and is faster than baseline outside measured noise"
        )
        promoted = True

    strict_projection_zero = all(
        projection_stats["max_abs"] == 0.0 and projection_stats["rel_l2"] == 0.0
        for projection_stats in projection_parity.values()
    )
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    max_tile_nbytes = int(tiling_after_interleaved.get("max_dense_tile_nbytes") or 0)
    full_dense_nbytes = int(tiling_after_interleaved.get("full_dense_nbytes_if_resident") or 0)
    packed_source_nbytes = int(tiling_after_interleaved.get("packed_quantized_source_nbytes") or 0)
    return {
        "name": "attention_qkv_tiled_dense_dequant",
        "target_segment": "qkv_quantized_matmul",
        "target_boundary": "Attention.qkv_proj transiently dequantizes output-channel tiles, runs dense matmul per tile, and concatenates in original row order",
        "selection_rationale": (
            f"Current segmented medians are qkv_quantized_matmul={qkv_median} s, "
            f"qk_rmsnorm_rope_layout={layout_median} s, and mlx_fast_sdpa={sdpa_median} s at sequence length "
            f"{sequence_meta.get('sequence_length')}. Prior Attention probes changed projection rank, input contiguity, "
            "QKV layout, q/k normalization/RoPE scheduling, SDPA input/output layout, or out_proj representation. "
            "This probe changes only qkv_proj weight materialization at the dominant Attention QMM boundary: output rows "
            "are tiled, dequantized transiently, projected densely, then concatenated without changing the per-head interleaved row contract."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {name: False for name in attention_candidate_flags},
            "enabled_flag_for_this_run": "use_qkv_tiled_dense_dequant_candidate",
            "helper": "tiled_dense_linear_projection + Attention.qkv_tiled_dense_dequant_info via Attention._qkv_project",
            "single_variable_guard": "all prior block, MLP, Attention projection-rank, dense-out, QKV-layout, SDPA-contiguity, and Metal candidates are forced off during this probe",
            "lora_path": "falls back to the existing quantized qkv_proj base projection whenever lora is not None; LoRA q/k/v deltas keep their existing path",
            "resident_full_dense_cache": False,
        },
        "qkv_tiling": {
            "api": "for each output row tile: mx.dequantize(weight[start:stop], scales[start:stop], biases[start:stop], group_size, bits, mode, dtype=scales.dtype); input @ tile.T; concatenate outputs",
            "source_is_quantized": source_is_quantized,
            "dequantize_available": dequantize_available,
            "tile_size_argument": tile_size,
            "tiling_before_projection": tiling_before_projection,
            "tiling_after_projection": tiling_after_projection,
            "tiling_after_interleaved": tiling_after_interleaved,
            "first_tiled_qkv_projection_seconds_includes_tile_materialization": first_tiled_qkv_projection_seconds,
            "max_transient_dense_tile_nbytes": max_tile_nbytes,
            "full_dense_nbytes_if_resident": full_dense_nbytes,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "max_transient_tile_nbytes_vs_full_dense": (max_tile_nbytes / full_dense_nbytes) if full_dense_nbytes else None,
            "extra_persistent_dense_nbytes_vs_packed_source": 0,
            "row_order_contract": "original qkv_proj output rows are per-head interleaved [h0:q,k,v][h1:q,k,v]...; tiling slices rows and concatenates in ascending row order",
        },
        "projection_shape_dtype_contract": projection_shape_dtype,
        "projection_parity": projection_parity,
        "projection_parity_ok": projection_parity_ok,
        "strict_projection_parity_zero": strict_projection_zero,
        "first_candidate_call_seconds": first_candidate_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "quality_boundary": {
            "parity_atol": args.parity_atol,
            "parity_rel_l2": args.parity_rel_l2,
            "retain_only_if_within_bounds": True,
        },
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention flag plus one tiled dense projection helper already shared with FFN probes; no default generation path, sigma/NFE, cache, or projection-rank change",
            "compile_cost": "no mx.compile or custom Metal kernel is introduced; timing includes per-tile dequantization/materialization and synchronization needed to keep tiles transient",
            "memory": "candidate never stores a resident full dense qkv_proj; artifact records max tile bytes, full-dense equivalent bytes, RSS/MLX deltas, pageouts, and swapouts",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence and chosen output tile size; tile size must be remeasured for other shapes before promotion",
            "maintainability": "local qkv_proj-only switch with explicit LoRA fallback; prior Attention candidates are forced off during this probe",
            "strict_equivalence": "raw QKV projection, reshaped q/k/v row order, post-qk-norm SDPA tensors, and full block output are checked before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_attention_qkv_pretranspose_layout(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe pretransposing packed QKV to SDPA layout before q/k/v slicing."""

    qkv_median = segment_stats.get("qkv_quantized_matmul", {}).get("median_seconds")
    layout_median = segment_stats.get("qk_rmsnorm_rope_layout", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    original_pretranspose_flag = bool(getattr(block.attn, "use_qkv_pretranspose_layout_candidate", False))
    original_qkv_2d_flag = bool(getattr(block.attn, "use_qkv_2d_projection_candidate", False))
    original_out_2d_flag = bool(getattr(block.attn, "use_out_2d_projection_candidate", False))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine it with prior projection-rank probes.
        block.attn.use_qkv_2d_projection_candidate = False
        block.attn.use_out_2d_projection_candidate = False
        block.attn.use_qkv_pretranspose_layout_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        qkv_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        baseline_q, baseline_k, baseline_v = block.attn._qkv_sdpa_tensors(qkv_input)
        set_candidate(True)
        candidate_q, candidate_k, candidate_v = block.attn._qkv_sdpa_tensors(qkv_input)
        mx.eval(qkv_input, baseline_q, baseline_k, baseline_v, candidate_q, candidate_k, candidate_v)
        mx.synchronize()
        qkv_layout_parity = {
            "q": _diff_stats(baseline_q, candidate_q),
            "k": _diff_stats(baseline_k, candidate_k),
            "v": _diff_stats(baseline_v, candidate_v),
        }
        qkv_layout_shape_dtype = {
            "input_shape": list(qkv_input.shape),
            "input_dtype": str(qkv_input.dtype),
            "baseline_q_shape": list(baseline_q.shape),
            "candidate_q_shape": list(candidate_q.shape),
            "baseline_k_shape": list(baseline_k.shape),
            "candidate_k_shape": list(candidate_k.shape),
            "baseline_v_shape": list(baseline_v.shape),
            "candidate_v_shape": list(candidate_v.shape),
            "baseline_q_dtype": str(baseline_q.dtype),
            "candidate_q_dtype": str(candidate_q.dtype),
            "baseline_k_dtype": str(baseline_k.dtype),
            "candidate_k_dtype": str(candidate_k.dtype),
            "baseline_v_dtype": str(baseline_v.dtype),
            "candidate_v_dtype": str(candidate_v.dtype),
            "sdpa_layout": "[B,H,S,D]",
            "pretranspose_intermediate_layout": "[B,H,3,S,D]",
            "shapes_match": bool(
                baseline_q.shape == candidate_q.shape
                and baseline_k.shape == candidate_k.shape
                and baseline_v.shape == candidate_v.shape
            ),
            "dtypes_match": bool(
                baseline_q.dtype == candidate_q.dtype
                and baseline_k.dtype == candidate_k.dtype
                and baseline_v.dtype == candidate_v.dtype
            ),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 5309,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        block.attn.use_qkv_pretranspose_layout_candidate = original_pretranspose_flag
        block.attn.use_qkv_2d_projection_candidate = original_qkv_2d_flag
        block.attn.use_out_2d_projection_candidate = original_out_2d_flag

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    layout_parity_ok = all(
        stats_for_tensor["max_abs"] <= args.parity_atol
        and stats_for_tensor["rel_l2"] <= args.parity_rel_l2
        for stats_for_tensor in qkv_layout_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(layout_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "strict-equivalence check failed for the Attention QKV pretranspose layout candidate"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the Attention QKV pretranspose layout candidate is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the Attention QKV pretranspose layout "
            "candidate from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "Attention QKV pretranspose layout candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "Attention QKV pretranspose layout candidate is strictly equivalent, disabled by default, "
            "memory-clean, and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_qkv_zero = all(
        tensor_stats["max_abs"] == 0.0 and tensor_stats["rel_l2"] == 0.0
        for tensor_stats in qkv_layout_parity.values()
    )
    return {
        "name": "attention_qkv_pretranspose_layout",
        "target_segment": "qkv_to_sdpa_layout",
        "target_boundary": "projected QKV reshaped as [B,S,H,3,D] then pretransposed to [B,H,3,S,D] before q/k/v slicing",
        "selection_rationale": (
            f"Current segmented medians are qkv_quantized_matmul={qkv_median} s, "
            f"qk_rmsnorm_rope_layout={layout_median} s, and mlx_fast_sdpa={sdpa_median} s. "
            "The referenced trace records a materialized qkv_projected_interleaved boundary before "
            "SDPA. This probe keeps QKV row semantics and QMM dispatch unchanged while changing only "
            "when the projected packed tensor is transposed into SDPA-ready [B,H,S,D] order."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {
                "use_qkv_2d_projection_candidate": False,
                "use_out_2d_projection_candidate": False,
                "use_qkv_pretranspose_layout_candidate": False,
            },
            "enabled_flag_for_this_run": "use_qkv_pretranspose_layout_candidate",
            "helper": "Attention._qkv_sdpa_tensors",
            "lora_path": "falls back to the existing baseline q/k/v layout path whenever lora is not None",
        },
        "qkv_layout_shape_dtype_contract": qkv_layout_shape_dtype,
        "qkv_layout_parity": qkv_layout_parity,
        "qkv_layout_parity_ok": layout_parity_ok,
        "strict_qkv_layout_parity_zero": strict_qkv_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention layout branch; no weights, quantization, sigma/NFE, cache, or default generation path changes",
            "compile_cost": "no mx.compile or persistent compiler cache is introduced by this candidate",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; larger S/M/N must be remeasured before release-profile promotion",
            "maintainability": "local layout scheduling switch with an explicit LoRA fallback; prior projection-rank candidate flags are forced off during this probe",
            "strict_equivalence": "q/k/v SDPA-layout tensors and full block output are checked against baseline layout",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, Amdahl contribution, memory, maintainability, resolution scaling, and strict parity; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_adaln_packed_gather(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe packing the six repeated per-token AdaLN gathers into one gather."""

    names = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
    target_labels = (
        "adaln_msa_indexed_norm_affine",
        "msa_gated_residual",
        "adaln_mlp_indexed_norm_affine",
        "mlp_gated_residual",
    )
    target_segment_medians = {
        label: segment_stats.get(label, {}).get("median_seconds") for label in target_labels
    }
    target_segment_median_sum = sum(
        float(value) for value in target_segment_medians.values() if value is not None
    )
    original_flag = bool(getattr(block, "use_packed_adaln_gather_candidate", False))
    unique_rows = sorted({int(value) for value in adaln_indices.tolist()})
    sequence_rows = int(adaln_indices.shape[0])
    modulation_rows = int(modulation[0].shape[0]) if modulation else 0

    def set_candidate(enabled: bool) -> None:
        block.use_packed_adaln_gather_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        baseline_rows = tuple(tensor[adaln_indices] for tensor in modulation)
        candidate_rows = gather_packed_modulation_rows(modulation, adaln_indices)
        mx.eval(*(baseline_rows + candidate_rows))
        mx.synchronize()
        row_parity = {
            name: _diff_stats(reference, candidate)
            for name, reference, candidate in zip(names, baseline_rows, candidate_rows)
        }
        row_shape_dtype = {
            "baseline_shapes": {name: list(tensor.shape) for name, tensor in zip(names, baseline_rows)},
            "candidate_shapes": {name: list(tensor.shape) for name, tensor in zip(names, candidate_rows)},
            "baseline_dtypes": {name: str(tensor.dtype) for name, tensor in zip(names, baseline_rows)},
            "candidate_dtypes": {name: str(tensor.dtype) for name, tensor in zip(names, candidate_rows)},
            "packed_gather_layout": "stack six [rows, hidden] tables as [6, rows, hidden], then gather [:, adaln_indices, :]",
            "shapes_match": all(
                reference.shape == candidate.shape for reference, candidate in zip(baseline_rows, candidate_rows)
            ),
            "dtypes_match": all(
                reference.dtype == candidate.dtype for reference, candidate in zip(baseline_rows, candidate_rows)
            ),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 7219,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        block.use_packed_adaln_gather_candidate = original_flag

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    row_parity_ok = all(
        tensor_stats["max_abs"] <= args.parity_atol and tensor_stats["rel_l2"] <= args.parity_rel_l2
        for tensor_stats in row_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(row_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "packed AdaLN gather changed gathered rows or full-block output beyond configured bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the packed AdaLN gather candidate is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the packed AdaLN gather candidate "
            "from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "packed AdaLN gather candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "packed AdaLN gather candidate is strictly equivalent, disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_row_zero = all(
        tensor_stats["max_abs"] == 0.0 and tensor_stats["rel_l2"] == 0.0
        for tensor_stats in row_parity.values()
    )
    return {
        "name": "adaln_packed_gather",
        "target_segment": "adaln_modulation_gather_materialization",
        "target_boundary": "six repeated per-token gathers from the unique (timestep, modality) AdaLN modulation table",
        "selection_rationale": (
            "Inspection found the block AdaLN projection is already unique-row at the projection input: "
            "the warm block harness projects one row per distinct timestep and reshapes to a small "
            "(timestep, modality) table, while the hot block repeatedly gathers that table over the "
            f"{sequence_rows} packed tokens. This candidate therefore leaves projection math unchanged "
            "and tests only whether packing the six shift/scale/gate gathers into one gather improves the "
            "modulation memory boundary."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.TransformerBlock",
            "default_flags": {"use_packed_adaln_gather_candidate": False},
            "enabled_flag_for_this_run": "use_packed_adaln_gather_candidate",
            "helper": "gather_packed_modulation_rows",
            "single_variable_guard": "attention, FFN, compile, quantization, timestep table, and projection paths are unchanged",
        },
        "projection_path_inspection": {
            "per_token_adaln_projection_present": False,
            "modulation_projection_in_timed_hotpath": sequence_meta.get("modulation_projection_in_timed_hotpath"),
            "block_adaln_projection_input_rows_for_step": sequence_meta.get("distinct_timestep_count_for_step"),
            "modulation_table_rows": modulation_rows,
            "unique_modulation_rows_used": len(unique_rows),
            "unique_modulation_row_indices_used": unique_rows,
            "sequence_rows_gathered": sequence_rows,
            "repeated_gather_factor_vs_unique_rows_used": (
                sequence_rows / len(unique_rows) if unique_rows else None
            ),
        },
        "target_segment_medians_seconds": target_segment_medians,
        "target_segment_median_sum_seconds": target_segment_median_sum,
        "row_shape_dtype_contract": row_shape_dtype,
        "row_parity": row_parity,
        "row_parity_ok": row_parity_ok,
        "strict_row_parity_zero": strict_row_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default TransformerBlock branch plus a pure MLX stack/gather helper; no weights, quantization, sigma/NFE, cache, or default generation path changes",
            "compile_cost": "no mx.compile, custom Metal kernel, or persistent compiler cache is introduced by this candidate",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas; candidate may materialize all six gathered modulation tensors at once",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; larger S/H must be remeasured before promotion",
            "strict_equivalence": "six gathered modulation row tensors and full block output are checked against baseline gathers",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, Amdahl contribution, memory, maintainability, resolution scaling, and strict parity; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_indexed_adaln_affine_metal(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe fusing each DiT block ``normed * (1 + scale[index]) + shift[index]`` modulation."""

    target_labels = ("adaln_msa_indexed_norm_affine", "adaln_mlp_indexed_norm_affine")
    target_segment_medians = {label: segment_stats.get(label, {}).get("median_seconds") for label in target_labels}
    target_segment_median_sum = sum(
        float(value) for value in target_segment_medians.values() if value is not None
    )
    block_flag_names = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    attn_flag_names = (
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_out_layout_metal_candidate",
    )
    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    original_block_flags = {name: bool(getattr(block, name, False)) for name in block_flag_names}
    original_attn_flags = {name: bool(getattr(block.attn, name, False)) for name in attn_flag_names}
    original_ffn_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    sequence_rows = int(adaln_indices.shape[0])
    hidden_size = int(x.shape[-1])
    modulation_rows = int(modulation[0].shape[0]) if modulation else 0
    unique_rows = sorted({int(value) for value in adaln_indices.tolist()})

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: force off every prior TransformerBlock/Attention/FFN candidate.
        for name in block_flag_names:
            setattr(block, name, False)
        for name in attn_flag_names:
            setattr(block.attn, name, False)
        for name in ffn_flag_names:
            setattr(block.mlp, name, False)
        block.use_indexed_adaln_affine_metal_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, _gate_mlp = modulation
        norm1_out = block.norm1(x)
        baseline_msa_affine = norm1_out * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
        candidate_msa_affine = indexed_adaln_affine_metal(norm1_out, scale_msa, shift_msa, adaln_indices)
        attn_out = block.attn(baseline_msa_affine, rotary)
        after_msa = x + gate_msa[adaln_indices] * attn_out
        norm2_out = block.norm2(after_msa)
        baseline_mlp_affine = norm2_out * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]
        candidate_mlp_affine = indexed_adaln_affine_metal(norm2_out, scale_mlp, shift_mlp, adaln_indices)
        mx.eval(
            norm1_out,
            baseline_msa_affine,
            candidate_msa_affine,
            attn_out,
            after_msa,
            norm2_out,
            baseline_mlp_affine,
            candidate_mlp_affine,
        )
        mx.synchronize()
        affine_parity = {
            "msa_indexed_norm_affine": _diff_stats(baseline_msa_affine, candidate_msa_affine),
            "mlp_indexed_norm_affine": _diff_stats(baseline_mlp_affine, candidate_mlp_affine),
        }
        affine_shape_dtype = {
            "norm1_shape": list(norm1_out.shape),
            "norm2_shape": list(norm2_out.shape),
            "scale_msa_shape": list(scale_msa.shape),
            "shift_msa_shape": list(shift_msa.shape),
            "scale_mlp_shape": list(scale_mlp.shape),
            "shift_mlp_shape": list(shift_mlp.shape),
            "adaln_indices_shape": list(adaln_indices.shape),
            "baseline_msa_affine_shape": list(baseline_msa_affine.shape),
            "candidate_msa_affine_shape": list(candidate_msa_affine.shape),
            "baseline_mlp_affine_shape": list(baseline_mlp_affine.shape),
            "candidate_mlp_affine_shape": list(candidate_mlp_affine.shape),
            "norm1_dtype": str(norm1_out.dtype),
            "norm2_dtype": str(norm2_out.dtype),
            "scale_msa_dtype": str(scale_msa.dtype),
            "shift_msa_dtype": str(shift_msa.dtype),
            "scale_mlp_dtype": str(scale_mlp.dtype),
            "shift_mlp_dtype": str(shift_mlp.dtype),
            "adaln_indices_dtype": str(adaln_indices.dtype),
            "baseline_msa_affine_dtype": str(baseline_msa_affine.dtype),
            "candidate_msa_affine_dtype": str(candidate_msa_affine.dtype),
            "baseline_mlp_affine_dtype": str(baseline_mlp_affine.dtype),
            "candidate_mlp_affine_dtype": str(candidate_mlp_affine.dtype),
            "kernel_inputs": "normed [B,S,H], scale table [rows,H], shift table [rows,H], adaln_indices [S]",
            "kernel_outputs": "one modulated norm tensor [B,S,H] per MSA/MLP boundary",
            "shapes_match": bool(
                baseline_msa_affine.shape == candidate_msa_affine.shape
                and baseline_mlp_affine.shape == candidate_mlp_affine.shape
            ),
            "dtypes_match": bool(
                baseline_msa_affine.dtype == candidate_msa_affine.dtype
                and baseline_mlp_affine.dtype == candidate_mlp_affine.dtype
            ),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 10037,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    except Exception as exc:
        after = _metrics()
        return {
            "name": "indexed_adaln_affine_metal",
            "target_segment": "adaln_msa_indexed_norm_affine+adaln_mlp_indexed_norm_affine",
            "target_boundary": "DiT block AdaLN modulation: normed * (1 + scale[adaln_indices]) + shift[adaln_indices]",
            "opt_in_only": True,
            "disabled_by_default": True,
            "default_behavior_unchanged": True,
            "production_integrated": True,
            "strict_exact_semantics": False,
            "parity_bounded_not_assumed_exact": True,
            "candidate_available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics_before": before,
            "metrics_after": after,
            "metrics_delta": _delta(before, after),
            "promote": False,
            "decision": "blocked_unsupported",
            "decision_reason": "custom Metal indexed AdaLN affine kernel could not be constructed or executed locally",
        }
    finally:
        for name, value in original_block_flags.items():
            setattr(block, name, value)
        for name, value in original_attn_flags.items():
            setattr(block.attn, name, value)
        for name, value in original_ffn_flags.items():
            setattr(block.mlp, name, value)

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    affine_parity_ok = all(
        tensor_stats["max_abs"] <= args.parity_atol and tensor_stats["rel_l2"] <= args.parity_rel_l2
        for tensor_stats in affine_parity.values()
    )
    shape_dtype_ok = bool(affine_shape_dtype["shapes_match"] and affine_shape_dtype["dtypes_match"])
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(affine_parity_ok and shape_dtype_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "custom Metal indexed AdaLN affine path changed modulation or full-block output beyond configured bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the indexed AdaLN affine Metal candidate is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the indexed AdaLN affine Metal "
            "candidate from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "indexed AdaLN affine Metal candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "indexed AdaLN affine Metal candidate is within parity bounds, disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_affine_zero = all(
        tensor_stats["max_abs"] == 0.0 and tensor_stats["rel_l2"] == 0.0
        for tensor_stats in affine_parity.values()
    )
    return {
        "name": "indexed_adaln_affine_metal",
        "target_segment": "adaln_msa_indexed_norm_affine+adaln_mlp_indexed_norm_affine",
        "target_boundary": "two DiT modulation boundaries: normed * (1 + scale[adaln_indices]) + shift[adaln_indices]",
        "selection_rationale": (
            "The standing warm block profile separates the two AdaLN indexed norm-affine regions from "
            "the larger QMM/attention/FFN work, and prior packed-gather/residual routes left this exact "
            "scale/shift gather plus affine expression in the hot path. This single-variable candidate "
            "leaves QMM, attention, FFN, residual gating, quantization, and timestep projection unchanged "
            "while replacing only each MSA/MLP modulation gather + multiply + add with one indexed custom "
            "Metal kernel."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": False,
        "parity_bounded_not_assumed_exact": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.TransformerBlock",
            "default_flags": {
                "use_packed_adaln_gather_candidate": False,
                "use_indexed_adaln_affine_metal_candidate": False,
                "use_indexed_gated_residual_metal_candidate": False,
                "Attention.*_candidate": False,
                "FeedForward.*_candidate": False,
            },
            "enabled_flag_for_this_run": "use_indexed_adaln_affine_metal_candidate",
            "single_variable_guard": "all other TransformerBlock, Attention, and FeedForward candidate flags are forced off during this probe",
            "lora_path": "falls back to the existing MLX gather/affine path whenever lora is not None",
            "kernel": "mx.fast.metal_kernel reads normed, scale table, shift table, and adaln_indices and writes one modulated norm tensor per boundary",
        },
        "modulation_table_inspection": {
            "sequence_rows": sequence_rows,
            "hidden_size": hidden_size,
            "modulation_table_rows": modulation_rows,
            "unique_modulation_rows_used": len(unique_rows),
            "unique_modulation_row_indices_used": unique_rows,
            "repeated_modulation_factor_vs_unique_rows_used": sequence_rows / len(unique_rows) if unique_rows else None,
            "modulation_projection_in_timed_hotpath": sequence_meta.get("modulation_projection_in_timed_hotpath"),
        },
        "target_segment_medians_seconds": target_segment_medians,
        "target_segment_median_sum_seconds": target_segment_median_sum,
        "affine_shape_dtype_contract": affine_shape_dtype,
        "affine_parity": affine_parity,
        "affine_parity_ok": affine_parity_ok,
        "strict_affine_parity_zero": strict_affine_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default TransformerBlock flag plus a cached indexed Metal pointwise kernel; no weights, quantization, sigma/NFE, cache, residual, or projection-rank change",
            "compile_cost": "first candidate call records custom Metal JIT/setup cost; warm interleaved samples measure the cached kernel path",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other sequence lengths and dtypes must be remeasured",
            "maintainability": "local modulation-boundary scheduling switch with explicit LoRA fallback; all other candidate flags are forced off during this probe",
            "strict_equivalence": "BF16 arithmetic-order drift is not assumed away; direct modulation parity and full-block parity are checked before timing promotion",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, Amdahl contribution, memory, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_indexed_gated_residual_metal(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe fusing each DiT block ``base + gate[indices] * branch`` residual boundary."""

    target_labels = ("msa_gated_residual", "mlp_gated_residual")
    target_segment_medians = {label: segment_stats.get(label, {}).get("median_seconds") for label in target_labels}
    target_segment_median_sum = sum(
        float(value) for value in target_segment_medians.values() if value is not None
    )
    original_residual_flag = bool(getattr(block, "use_indexed_gated_residual_metal_candidate", False))
    original_packed_flag = bool(getattr(block, "use_packed_adaln_gather_candidate", False))
    sequence_rows = int(adaln_indices.shape[0])
    hidden_size = int(x.shape[-1])
    gate_rows = int(modulation[2].shape[0]) if len(modulation) >= 3 else 0
    unique_rows = sorted({int(value) for value in adaln_indices.tolist()})

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine it with the prior packed-gather route.
        block.use_packed_adaln_gather_candidate = False
        block.use_indexed_gated_residual_metal_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        h = block.norm1(x) * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
        attn_out = block.attn(h, rotary)
        baseline_msa = x + gate_msa[adaln_indices] * attn_out
        candidate_msa = indexed_gated_residual_metal(x, gate_msa, adaln_indices, attn_out)
        h2 = block.norm2(baseline_msa) * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]
        mlp_out = block.mlp(h2)
        baseline_mlp = baseline_msa + gate_mlp[adaln_indices] * mlp_out
        candidate_mlp = indexed_gated_residual_metal(baseline_msa, gate_mlp, adaln_indices, mlp_out)
        mx.eval(h, attn_out, baseline_msa, candidate_msa, h2, mlp_out, baseline_mlp, candidate_mlp)
        mx.synchronize()
        residual_parity = {
            "msa_gated_residual": _diff_stats(baseline_msa, candidate_msa),
            "mlp_gated_residual": _diff_stats(baseline_mlp, candidate_mlp),
        }
        residual_shape_dtype = {
            "base_shape": list(x.shape),
            "attn_branch_shape": list(attn_out.shape),
            "mlp_branch_shape": list(mlp_out.shape),
            "gate_msa_shape": list(gate_msa.shape),
            "gate_mlp_shape": list(gate_mlp.shape),
            "adaln_indices_shape": list(adaln_indices.shape),
            "base_dtype": str(x.dtype),
            "attn_branch_dtype": str(attn_out.dtype),
            "mlp_branch_dtype": str(mlp_out.dtype),
            "gate_msa_dtype": str(gate_msa.dtype),
            "gate_mlp_dtype": str(gate_mlp.dtype),
            "adaln_indices_dtype": str(adaln_indices.dtype),
            "kernel_inputs": "base [B,S,H], gate table [rows,H], adaln_indices [S], branch_out [B,S,H]",
            "kernel_outputs": "one residual tensor [B,S,H] per boundary",
            "shapes_match": bool(
                baseline_msa.shape == candidate_msa.shape and baseline_mlp.shape == candidate_mlp.shape
            ),
            "dtypes_match": bool(
                baseline_msa.dtype == candidate_msa.dtype and baseline_mlp.dtype == candidate_mlp.dtype
            ),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 9679,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    except Exception as exc:
        after = _metrics()
        return {
            "name": "indexed_gated_residual_metal",
            "target_segment": "msa_gated_residual+mlp_gated_residual",
            "target_boundary": "DiT block residual add with per-row AdaLN gate gather",
            "opt_in_only": True,
            "disabled_by_default": True,
            "default_behavior_unchanged": True,
            "production_integrated": True,
            "strict_exact_semantics": False,
            "parity_bounded_not_assumed_exact": True,
            "candidate_available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics_before": before,
            "metrics_after": after,
            "metrics_delta": _delta(before, after),
            "promote": False,
            "decision": "blocked_unsupported",
            "decision_reason": "custom Metal indexed gated-residual kernel could not be constructed or executed locally",
        }
    finally:
        block.use_indexed_gated_residual_metal_candidate = original_residual_flag
        block.use_packed_adaln_gather_candidate = original_packed_flag

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    residual_parity_ok = all(
        tensor_stats["max_abs"] <= args.parity_atol and tensor_stats["rel_l2"] <= args.parity_rel_l2
        for tensor_stats in residual_parity.values()
    )
    shape_dtype_ok = bool(residual_shape_dtype["shapes_match"] and residual_shape_dtype["dtypes_match"])
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(residual_parity_ok and shape_dtype_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "custom Metal indexed gated-residual path changed residual or full-block output beyond configured bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the indexed gated-residual Metal candidate is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the indexed gated-residual Metal "
            "candidate from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "indexed gated-residual Metal candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "indexed gated-residual Metal candidate is within parity bounds, disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_residual_zero = all(
        tensor_stats["max_abs"] == 0.0 and tensor_stats["rel_l2"] == 0.0
        for tensor_stats in residual_parity.values()
    )
    return {
        "name": "indexed_gated_residual_metal",
        "target_segment": "msa_gated_residual+mlp_gated_residual",
        "target_boundary": "two DiT residual boundaries: base + gate[adaln_indices] * branch_out",
        "selection_rationale": (
            "The existing warm block trace identifies gather/pointwise materialization around the matmul-heavy path, "
            f"and the current segmented medians for the two gated residual segments are {target_segment_medians}. "
            "Prior AdaLN work packed six gathers but left the residual arithmetic as separate gather/multiply/add. "
            "This single-variable candidate leaves attention, FFN, q/k/v layout, projections, quantization, and the "
            "AdaLN norm-affine boundaries unchanged while replacing only each residual gate gather + multiply + add "
            "with one indexed custom Metal kernel."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": False,
        "parity_bounded_not_assumed_exact": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.TransformerBlock",
            "default_flags": {
                "use_packed_adaln_gather_candidate": False,
                "use_indexed_gated_residual_metal_candidate": False,
            },
            "enabled_flag_for_this_run": "use_indexed_gated_residual_metal_candidate",
            "single_variable_guard": "the prior packed AdaLN gather candidate is forced off during this probe; Attention/FFN candidate flags are untouched/default",
            "lora_path": "falls back to the existing gather/multiply/add residual path whenever lora is not None",
            "kernel": "mx.fast.metal_kernel reads base, gate table, adaln_indices, and branch_out and writes one residual tensor per boundary",
        },
        "residual_table_inspection": {
            "sequence_rows": sequence_rows,
            "hidden_size": hidden_size,
            "gate_table_rows": gate_rows,
            "unique_gate_rows_used": len(unique_rows),
            "unique_gate_row_indices_used": unique_rows,
            "repeated_gate_factor_vs_unique_rows_used": sequence_rows / len(unique_rows) if unique_rows else None,
            "modulation_projection_in_timed_hotpath": sequence_meta.get("modulation_projection_in_timed_hotpath"),
        },
        "target_segment_medians_seconds": target_segment_medians,
        "target_segment_median_sum_seconds": target_segment_median_sum,
        "residual_shape_dtype_contract": residual_shape_dtype,
        "residual_parity": residual_parity,
        "residual_parity_ok": residual_parity_ok,
        "strict_residual_parity_zero": strict_residual_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default TransformerBlock flag plus a cached indexed Metal pointwise kernel; no weights, quantization, sigma/NFE, cache, or projection-rank change",
            "compile_cost": "first candidate call records custom Metal JIT/setup cost; warm interleaved samples measure the cached kernel path",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other sequence lengths and dtypes must be remeasured",
            "maintainability": "local residual-boundary scheduling switch with explicit LoRA fallback; prior packed-gather candidate is forced off during this probe",
            "strict_equivalence": "BF16 arithmetic-order drift is not assumed away; direct residual parity and full-block parity are checked before timing promotion",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, Amdahl contribution, memory, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_attention_pre_sdpa_contiguous(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe explicit q/k/v ``mx.contiguous`` materialization immediately before SDPA."""

    layout_median = segment_stats.get("qk_rmsnorm_rope_layout", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    original_contiguous_flag = bool(getattr(block.attn, "use_pre_sdpa_contiguous_candidate", False))
    original_norm_layout_flag = bool(getattr(block.attn, "use_qkv_rmsnorm_sdpa_metal_candidate", False))
    original_rotary_flag = bool(getattr(block.attn, "use_rotary_qk_metal_candidate", False))
    original_pretranspose_flag = bool(getattr(block.attn, "use_qkv_pretranspose_layout_candidate", False))
    original_sdpa_out_flag = bool(getattr(block.attn, "use_sdpa_out_layout_metal_candidate", False))
    original_qkv_2d_flag = bool(getattr(block.attn, "use_qkv_2d_projection_candidate", False))
    original_out_2d_flag = bool(getattr(block.attn, "use_out_2d_projection_candidate", False))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine it with prior Attention layout/Metal/QMM probes.
        block.attn.use_qkv_2d_projection_candidate = False
        block.attn.use_out_2d_projection_candidate = False
        block.attn.use_qkv_pretranspose_layout_candidate = False
        block.attn.use_qkv_rmsnorm_sdpa_metal_candidate = False
        block.attn.use_rotary_qk_metal_candidate = False
        block.attn.use_sdpa_out_layout_metal_candidate = False
        block.attn.use_pre_sdpa_contiguous_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        attn_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        q, k, v = block.attn._qkv_sdpa_tensors(attn_input)
        q = apply_rotary(q, *rotary)
        k = apply_rotary(k, *rotary)
        baseline_q, baseline_k, baseline_v = block.attn._pre_sdpa_inputs(q, k, v)
        baseline_sdpa = mx.fast.scaled_dot_product_attention(
            baseline_q,
            baseline_k,
            baseline_v,
            scale=block.attn.scale,
            mask=None,
        )
        set_candidate(True)
        candidate_q, candidate_k, candidate_v = block.attn._pre_sdpa_inputs(q, k, v)
        candidate_sdpa = mx.fast.scaled_dot_product_attention(
            candidate_q,
            candidate_k,
            candidate_v,
            scale=block.attn.scale,
            mask=None,
        )
        mx.eval(
            attn_input,
            baseline_q,
            baseline_k,
            baseline_v,
            baseline_sdpa,
            candidate_q,
            candidate_k,
            candidate_v,
            candidate_sdpa,
        )
        mx.synchronize()
        qkv_pre_sdpa_parity = {
            "q": _diff_stats(baseline_q, candidate_q),
            "k": _diff_stats(baseline_k, candidate_k),
            "v": _diff_stats(baseline_v, candidate_v),
        }
        sdpa_output_parity = _diff_stats(baseline_sdpa, candidate_sdpa)
        pre_sdpa_shape_dtype = {
            "input_shape": list(attn_input.shape),
            "input_dtype": str(attn_input.dtype),
            "baseline_q_shape": list(baseline_q.shape),
            "candidate_q_shape": list(candidate_q.shape),
            "baseline_k_shape": list(baseline_k.shape),
            "candidate_k_shape": list(candidate_k.shape),
            "baseline_v_shape": list(baseline_v.shape),
            "candidate_v_shape": list(candidate_v.shape),
            "baseline_q_dtype": str(baseline_q.dtype),
            "candidate_q_dtype": str(candidate_q.dtype),
            "baseline_k_dtype": str(baseline_k.dtype),
            "candidate_k_dtype": str(candidate_k.dtype),
            "baseline_v_dtype": str(baseline_v.dtype),
            "candidate_v_dtype": str(candidate_v.dtype),
            "baseline_sdpa_shape": list(baseline_sdpa.shape),
            "candidate_sdpa_shape": list(candidate_sdpa.shape),
            "baseline_sdpa_dtype": str(baseline_sdpa.dtype),
            "candidate_sdpa_dtype": str(candidate_sdpa.dtype),
            "sdpa_layout": "[B,H,S,D]",
            "materialization_api": "mx.contiguous(q), mx.contiguous(k), mx.contiguous(v)",
            "shapes_match": bool(
                baseline_q.shape == candidate_q.shape
                and baseline_k.shape == candidate_k.shape
                and baseline_v.shape == candidate_v.shape
                and baseline_sdpa.shape == candidate_sdpa.shape
            ),
            "dtypes_match": bool(
                baseline_q.dtype == candidate_q.dtype
                and baseline_k.dtype == candidate_k.dtype
                and baseline_v.dtype == candidate_v.dtype
                and baseline_sdpa.dtype == candidate_sdpa.dtype
            ),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 5923,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        block.attn.use_pre_sdpa_contiguous_candidate = original_contiguous_flag
        block.attn.use_qkv_rmsnorm_sdpa_metal_candidate = original_norm_layout_flag
        block.attn.use_rotary_qk_metal_candidate = original_rotary_flag
        block.attn.use_qkv_pretranspose_layout_candidate = original_pretranspose_flag
        block.attn.use_sdpa_out_layout_metal_candidate = original_sdpa_out_flag
        block.attn.use_qkv_2d_projection_candidate = original_qkv_2d_flag
        block.attn.use_out_2d_projection_candidate = original_out_2d_flag

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    input_parity_ok = all(
        stats_for_tensor["max_abs"] <= args.parity_atol
        and stats_for_tensor["rel_l2"] <= args.parity_rel_l2
        for stats_for_tensor in qkv_pre_sdpa_parity.values()
    )
    sdpa_parity_ok = (
        sdpa_output_parity["max_abs"] <= args.parity_atol
        and sdpa_output_parity["rel_l2"] <= args.parity_rel_l2
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(input_parity_ok and sdpa_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "pre-SDPA contiguous materialization changed q/k/v, SDPA, or full-block outputs beyond configured bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says pre-SDPA q/k/v contiguous materialization is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate pre-SDPA q/k/v contiguous materialization "
            "from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "pre-SDPA q/k/v contiguous materialization is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "pre-SDPA q/k/v contiguous materialization satisfies parity bounds, is disabled by default, "
            "memory-clean, and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_input_zero = all(
        tensor_stats["max_abs"] == 0.0 and tensor_stats["rel_l2"] == 0.0
        for tensor_stats in qkv_pre_sdpa_parity.values()
    )
    strict_sdpa_zero = sdpa_output_parity["max_abs"] == 0.0 and sdpa_output_parity["rel_l2"] == 0.0
    return {
        "name": "attention_pre_sdpa_contiguous",
        "target_segment": "mlx_fast_sdpa",
        "target_boundary": "q/k/v [B,H,S,D] tensors after q/k RMSNorm and RoPE, immediately before scaled_dot_product_attention",
        "selection_rationale": (
            f"Current segmented medians are qk_rmsnorm_rope_layout={layout_median} s and "
            f"mlx_fast_sdpa={sdpa_median} s. Prior Attention probes changed projection rank, "
            "QKV pretranspose scheduling, q/k RMSNorm-layout fusion, RoPE, or post-SDPA output layout. "
            "This probe changes only whether the final q/k/v SDPA inputs are explicitly materialized "
            "with mx.contiguous immediately before MLX SDPA, testing for hidden stride/layout costs."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {
                "use_qkv_2d_projection_candidate": False,
                "use_out_2d_projection_candidate": False,
                "use_qkv_pretranspose_layout_candidate": False,
                "use_qkv_rmsnorm_sdpa_metal_candidate": False,
                "use_rotary_qk_metal_candidate": False,
                "use_pre_sdpa_contiguous_candidate": False,
                "use_sdpa_out_layout_metal_candidate": False,
            },
            "enabled_flag_for_this_run": "use_pre_sdpa_contiguous_candidate",
            "helper": "Attention._pre_sdpa_inputs -> materialize_sdpa_inputs_contiguous",
            "single_variable_guard": "prior Attention projection-rank, pretranspose, q/k RMSNorm-layout, RoPE, and post-SDPA layout candidates are forced off during this probe",
            "lora_path": "falls back to existing baseline q/k/v tensors whenever lora is not None",
        },
        "pre_sdpa_shape_dtype_contract": pre_sdpa_shape_dtype,
        "qkv_pre_sdpa_parity": qkv_pre_sdpa_parity,
        "qkv_pre_sdpa_parity_ok": input_parity_ok,
        "strict_qkv_pre_sdpa_parity_zero": strict_input_zero,
        "sdpa_output_parity": sdpa_output_parity,
        "sdpa_output_parity_ok": sdpa_parity_ok,
        "strict_sdpa_output_parity_zero": strict_sdpa_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention branch plus a pure mx.contiguous helper; no weights, quantization, sigma/NFE, cache, or default generation path changes",
            "compile_cost": "no mx.compile, custom Metal kernel, or persistent compiler cache is introduced by this candidate",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas; candidate may allocate up to three explicit q/k/v buffers",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; larger S/M/N must be remeasured before release-profile promotion",
            "maintainability": "local pre-SDPA materialization switch with an explicit LoRA fallback; prior Attention candidates are forced off during this probe",
            "strict_equivalence": "q/k/v inputs, direct SDPA output, and full block output are checked against baseline layout",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, Amdahl contribution, memory, maintainability, resolution scaling, and strict parity; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_attention_sdpa_headgroup_split_rank4(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe calling rank-4 MLX SDPA on contiguous head groups instead of all heads at once."""

    layout_median = segment_stats.get("qk_rmsnorm_rope_layout", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    requested_heads_per_group = int(args.attention_sdpa_headgroup_size)
    attention_flag_names = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_out_tiled_dense_dequant_candidate",
        "use_qkv_tiled_dense_dequant_candidate",
        "use_qkv_headgroup_row_sliced_qmm_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_head_batch_rank3_candidate",
        "use_sdpa_headgroup_split_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    original_flags = {name: bool(getattr(block.attn, name, False)) for name in attention_flag_names}
    original_heads_per_group = int(getattr(block.attn, "sdpa_headgroup_heads_per_group", 8))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine it with prior Attention materialization,
        # QMM, dense-dequant, RoPE, RMSNorm-layout, rank-3 SDPA, or post-SDPA layout probes.
        for name in attention_flag_names:
            setattr(block.attn, name, False)
        block.attn.sdpa_headgroup_heads_per_group = requested_heads_per_group
        block.attn.use_sdpa_headgroup_split_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        attn_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        q, k, v = block.attn._qkv_sdpa_tensors(attn_input)
        q = apply_rotary(q, *rotary)
        k = apply_rotary(k, *rotary)
        baseline_sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=block.attn.scale, mask=None)
        set_candidate(True)
        candidate_sdpa = sdpa_headgroup_split_rank4(
            q,
            k,
            v,
            scale=block.attn.scale,
            mask=None,
            heads_per_group=requested_heads_per_group,
        )
        mx.eval(attn_input, q, k, v, baseline_sdpa, candidate_sdpa)
        mx.synchronize()
        batch, heads, seq, head_dim = q.shape
        total_heads = int(heads)
        effective_heads_per_group = min(requested_heads_per_group, total_heads)
        head_group_ranges = []
        for head_start in range(0, total_heads, effective_heads_per_group):
            head_stop = min(head_start + effective_heads_per_group, total_heads)
            head_group_ranges.append([head_start, head_stop])
        sdpa_output_parity = _diff_stats(baseline_sdpa, candidate_sdpa)
        sdpa_shape_dtype = {
            "input_shape": list(attn_input.shape),
            "input_dtype": str(attn_input.dtype),
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "q_dtype": str(q.dtype),
            "k_dtype": str(k.dtype),
            "v_dtype": str(v.dtype),
            "rank4_sdpa_input_layout": "[B,H,S,D]",
            "baseline_sdpa_heads_per_call": total_heads,
            "requested_heads_per_group": requested_heads_per_group,
            "effective_heads_per_group": effective_heads_per_group,
            "head_group_count": len(head_group_ranges),
            "head_group_ranges": head_group_ranges,
            "baseline_sdpa_shape": list(baseline_sdpa.shape),
            "candidate_sdpa_shape": list(candidate_sdpa.shape),
            "baseline_sdpa_dtype": str(baseline_sdpa.dtype),
            "candidate_sdpa_dtype": str(candidate_sdpa.dtype),
            "mask_supported_by_candidate": True,
            "mask_used_in_probe": None,
            "mask_semantics": "mask=None in this real block probe; helper reuses broadcast masks and slices explicit per-head masks along the same contiguous head range",
            "shapes_match": bool(baseline_sdpa.shape == candidate_sdpa.shape == q.shape),
            "dtypes_match": bool(baseline_sdpa.dtype == candidate_sdpa.dtype),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 6421,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    except Exception as exc:
        after = _metrics()
        return {
            "name": "attention_sdpa_headgroup_split_rank4",
            "target_segment": "mlx_fast_sdpa",
            "target_boundary": "q/k/v [B,H,S,D] tensors immediately around mx.fast.scaled_dot_product_attention",
            "selection_rationale": (
                f"Current segmented medians are qk_rmsnorm_rope_layout={layout_median} s and "
                f"mlx_fast_sdpa={sdpa_median} s. This single-variable probe leaves q/k/v preparation, "
                "RoPE, output layout, projections, quantization, and cache behavior unchanged, and tests only "
                "whether splitting the supported rank-4 SDPA call into contiguous head groups improves MLX scheduling."
            ),
            "opt_in_only": True,
            "strict_exact_semantics": True,
            "disabled_by_default": True,
            "default_behavior_unchanged": True,
            "production_integrated": True,
            "candidate_available": False,
            "requested_heads_per_group": requested_heads_per_group,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics_before": before,
            "metrics_after": after,
            "metrics_delta": _delta(before, after),
            "promote": False,
            "decision": "blocked_unsupported",
            "decision_reason": "rank-4 SDPA head-group split candidate could not be constructed or executed locally",
        }
    finally:
        for name, value in original_flags.items():
            setattr(block.attn, name, value)
        block.attn.sdpa_headgroup_heads_per_group = original_heads_per_group

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    sdpa_parity_ok = (
        sdpa_output_parity["max_abs"] <= args.parity_atol
        and sdpa_output_parity["rel_l2"] <= args.parity_rel_l2
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(sdpa_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "rank-4 head-group SDPA changed direct SDPA or full-block outputs beyond configured bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says rank-4 head-group SDPA is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate rank-4 head-group SDPA from measurement "
            "noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "rank-4 head-group SDPA is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "rank-4 head-group SDPA satisfies parity bounds, is disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_sdpa_zero = sdpa_output_parity["max_abs"] == 0.0 and sdpa_output_parity["rel_l2"] == 0.0
    return {
        "name": "attention_sdpa_headgroup_split_rank4",
        "target_segment": "mlx_fast_sdpa",
        "target_boundary": "q/k/v [B,H,S,D] tensors immediately around mx.fast.scaled_dot_product_attention",
        "selection_rationale": (
            f"Current segmented medians are qk_rmsnorm_rope_layout={layout_median} s and "
            f"mlx_fast_sdpa={sdpa_median} s. Prior Attention probes changed projection rank, QKV "
            "pretranspose scheduling, q/k RMSNorm-layout fusion, RoPE, pre-SDPA contiguity, rank-3 SDPA, "
            "or post-SDPA output layout. This probe keeps q/k/v as supported rank-4 [B,H,S,D] tensors and "
            "changes only SDPA scheduling by calling mx.fast.scaled_dot_product_attention on contiguous head groups."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {name: False for name in attention_flag_names},
            "enabled_flag_for_this_run": "use_sdpa_headgroup_split_candidate",
            "helper": "sdpa_headgroup_split_rank4",
            "requested_heads_per_group": requested_heads_per_group,
            "effective_heads_per_group": effective_heads_per_group,
            "single_variable_guard": "all prior Attention materialization, projection-rank, dense-dequant, RoPE, RMSNorm-layout, rank-3 SDPA, and post-SDPA layout candidates are forced off during this probe",
            "lora_path": "falls back to existing baseline single-call rank-4 SDPA whenever lora is not None",
            "mask_path": "passes the same broadcast mask to each group and slices masks with an explicit head axis; the measured MiniMax-H3 DiT block uses mask=None",
        },
        "sdpa_shape_dtype_contract": sdpa_shape_dtype,
        "sdpa_output_parity": sdpa_output_parity,
        "sdpa_output_parity_ok": sdpa_parity_ok,
        "strict_sdpa_output_parity_zero": strict_sdpa_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention branch plus a small rank-4 SDPA helper; no weights, quantization, sigma/NFE, cache, output layout, or default generation path changes",
            "compile_cost": "no mx.compile, custom Metal kernel, or persistent compiler cache is introduced by this candidate",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas; candidate may allocate one SDPA output per head group before concatenation",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; MLX may choose different SDPA kernels for other sequence/head/head_dim shapes",
            "maintainability": "local SDPA dispatch switch with explicit LoRA fallback and mask handling; prior Attention candidates are forced off during this probe",
            "strict_equivalence": "direct SDPA output and full block output are checked against the baseline single rank-4 SDPA call",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, Amdahl contribution, memory, maintainability, resolution scaling, and strict parity; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_attention_sdpa_head_batch_rank3(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe folding attention heads into the batch dimension only for MLX SDPA."""

    layout_median = segment_stats.get("qk_rmsnorm_rope_layout", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    attention_flag_names = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_qkv_tiled_dense_dequant_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_head_batch_rank3_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    original_flags = {name: bool(getattr(block.attn, name, False)) for name in attention_flag_names}

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine it with prior Attention materialization,
        # QMM-rank, dense-dequant, RoPE, RMSNorm-layout, or post-SDPA layout probes.
        for name in attention_flag_names:
            setattr(block.attn, name, False)
        block.attn.use_sdpa_head_batch_rank3_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        attn_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        q, k, v = block.attn._qkv_sdpa_tensors(attn_input)
        q = apply_rotary(q, *rotary)
        k = apply_rotary(k, *rotary)
        baseline_sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=block.attn.scale, mask=None)
        set_candidate(True)
        candidate_sdpa = sdpa_head_batch_rank3(q, k, v, scale=block.attn.scale, mask=None)
        mx.eval(attn_input, q, k, v, baseline_sdpa, candidate_sdpa)
        mx.synchronize()
        batch, heads, seq, head_dim = q.shape
        rank3_shape = (int(batch) * int(heads), int(seq), int(head_dim))
        sdpa_output_parity = _diff_stats(baseline_sdpa, candidate_sdpa)
        sdpa_shape_dtype = {
            "input_shape": list(attn_input.shape),
            "input_dtype": str(attn_input.dtype),
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "q_dtype": str(q.dtype),
            "k_dtype": str(k.dtype),
            "v_dtype": str(v.dtype),
            "rank4_sdpa_input_layout": "[B,H,S,D]",
            "rank3_sdpa_input_shape": list(rank3_shape),
            "rank3_sdpa_input_layout": "[B*H,S,D] with original head order folded into the batch axis",
            "baseline_sdpa_shape": list(baseline_sdpa.shape),
            "candidate_sdpa_shape": list(candidate_sdpa.shape),
            "baseline_sdpa_dtype": str(baseline_sdpa.dtype),
            "candidate_sdpa_dtype": str(candidate_sdpa.dtype),
            "mask_supported_by_candidate": False,
            "mask_used_in_probe": None,
            "shapes_match": bool(baseline_sdpa.shape == candidate_sdpa.shape == q.shape),
            "dtypes_match": bool(baseline_sdpa.dtype == candidate_sdpa.dtype),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 6197,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    except Exception as exc:
        after = _metrics()
        return {
            "name": "attention_sdpa_head_batch_rank3",
            "target_segment": "mlx_fast_sdpa",
            "target_boundary": "q/k/v [B,H,S,D] tensors immediately around mx.fast.scaled_dot_product_attention",
            "selection_rationale": (
                f"Current segmented medians are qk_rmsnorm_rope_layout={layout_median} s and "
                f"mlx_fast_sdpa={sdpa_median} s. This single-variable probe leaves q/k/v preparation, "
                "RoPE, output layout, projections, quantization, and cache behavior unchanged, and tests only "
                "whether MLX dispatches SDPA more efficiently when heads are folded into rank-3 [B*H,S,D]."
            ),
            "opt_in_only": True,
            "strict_exact_semantics": True,
            "disabled_by_default": True,
            "default_behavior_unchanged": True,
            "production_integrated": True,
            "candidate_available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics_before": before,
            "metrics_after": after,
            "metrics_delta": _delta(before, after),
            "promote": False,
            "decision": "blocked_unsupported",
            "decision_reason": "MLX rank-3 scaled_dot_product_attention candidate could not be constructed or executed locally",
        }
    finally:
        for name, value in original_flags.items():
            setattr(block.attn, name, value)

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    sdpa_parity_ok = (
        sdpa_output_parity["max_abs"] <= args.parity_atol
        and sdpa_output_parity["rel_l2"] <= args.parity_rel_l2
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(sdpa_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "rank-3 head-batched SDPA changed direct SDPA or full-block outputs beyond configured bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says rank-3 head-batched SDPA is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate rank-3 head-batched SDPA from measurement "
            "noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "rank-3 head-batched SDPA is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "rank-3 head-batched SDPA satisfies parity bounds, is disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_sdpa_zero = sdpa_output_parity["max_abs"] == 0.0 and sdpa_output_parity["rel_l2"] == 0.0
    return {
        "name": "attention_sdpa_head_batch_rank3",
        "target_segment": "mlx_fast_sdpa",
        "target_boundary": "q/k/v [B,H,S,D] tensors immediately around mx.fast.scaled_dot_product_attention",
        "selection_rationale": (
            f"Current segmented medians are qk_rmsnorm_rope_layout={layout_median} s and "
            f"mlx_fast_sdpa={sdpa_median} s. Prior Attention probes changed projection rank, QKV "
            "pretranspose scheduling, q/k RMSNorm-layout fusion, RoPE, pre-SDPA contiguity, or post-SDPA "
            "output layout. This probe changes only the SDPA call rank by reshaping prepared q/k/v from "
            "[B,H,S,D] to [B*H,S,D] for mx.fast.scaled_dot_product_attention and reshaping its output back."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {name: False for name in attention_flag_names},
            "enabled_flag_for_this_run": "use_sdpa_head_batch_rank3_candidate",
            "helper": "sdpa_head_batch_rank3",
            "single_variable_guard": "all prior Attention materialization, projection-rank, dense-dequant, RoPE, RMSNorm-layout, and post-SDPA layout candidates are forced off during this probe",
            "lora_path": "falls back to existing baseline rank-4 SDPA whenever lora is not None",
            "mask_path": "falls back to existing baseline rank-4 SDPA whenever mask is not None",
        },
        "sdpa_shape_dtype_contract": sdpa_shape_dtype,
        "sdpa_output_parity": sdpa_output_parity,
        "sdpa_output_parity_ok": sdpa_parity_ok,
        "strict_sdpa_output_parity_zero": strict_sdpa_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention branch plus a pure reshape helper; no weights, quantization, sigma/NFE, cache, output layout, or default generation path changes",
            "compile_cost": "no mx.compile, custom Metal kernel, or persistent compiler cache is introduced by this candidate",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas; candidate may allocate reshape/materialization buffers depending on MLX layout handling",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; MLX may choose different SDPA kernels for other S/H/D shapes",
            "maintainability": "local SDPA dispatch switch with explicit LoRA and mask fallback; prior Attention candidates are forced off during this probe",
            "strict_equivalence": "direct SDPA output and full block output are checked against baseline rank-4 SDPA",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, Amdahl contribution, memory, maintainability, resolution scaling, and strict parity; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_attention_qkv_rmsnorm_rotary_sdpa_metal(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe fusing q/k RMSNorm, RoPE, and SDPA-layout materialization in one Metal kernel."""

    qkv_median = segment_stats.get("qkv_quantized_matmul", {}).get("median_seconds")
    layout_median = segment_stats.get("qk_rmsnorm_rope_layout", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    original_flags = {
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate": bool(
            getattr(block.attn, "use_qkv_rmsnorm_rotary_sdpa_metal_candidate", False)
        ),
        "use_qkv_rmsnorm_sdpa_metal_candidate": bool(getattr(block.attn, "use_qkv_rmsnorm_sdpa_metal_candidate", False)),
        "use_rotary_qk_metal_candidate": bool(getattr(block.attn, "use_rotary_qk_metal_candidate", False)),
        "use_qkv_pretranspose_layout_candidate": bool(getattr(block.attn, "use_qkv_pretranspose_layout_candidate", False)),
        "use_pre_sdpa_contiguous_candidate": bool(getattr(block.attn, "use_pre_sdpa_contiguous_candidate", False)),
        "use_sdpa_out_layout_metal_candidate": bool(getattr(block.attn, "use_sdpa_out_layout_metal_candidate", False)),
        "use_qkv_2d_projection_candidate": bool(getattr(block.attn, "use_qkv_2d_projection_candidate", False)),
        "use_out_2d_projection_candidate": bool(getattr(block.attn, "use_out_2d_projection_candidate", False)),
    }

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine it with prior Attention candidates.
        block.attn.use_qkv_2d_projection_candidate = False
        block.attn.use_out_2d_projection_candidate = False
        block.attn.use_qkv_pretranspose_layout_candidate = False
        block.attn.use_qkv_rmsnorm_sdpa_metal_candidate = False
        block.attn.use_rotary_qk_metal_candidate = False
        block.attn.use_pre_sdpa_contiguous_candidate = False
        block.attn.use_sdpa_out_layout_metal_candidate = False
        block.attn.use_qkv_rmsnorm_rotary_sdpa_metal_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        qkv_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        baseline_q, baseline_k, baseline_v = block.attn._qkv_sdpa_tensors(qkv_input)
        baseline_q = apply_rotary(baseline_q, *rotary)
        baseline_k = apply_rotary(baseline_k, *rotary)
        baseline_sdpa = mx.fast.scaled_dot_product_attention(
            baseline_q,
            baseline_k,
            baseline_v,
            scale=block.attn.scale,
            mask=None,
        )
        set_candidate(True)
        candidate_q, candidate_k, candidate_v = block.attn._qkv_rmsnorm_rotary_sdpa_tensors(qkv_input, rotary)
        candidate_sdpa = mx.fast.scaled_dot_product_attention(
            candidate_q,
            candidate_k,
            candidate_v,
            scale=block.attn.scale,
            mask=None,
        )
        mx.eval(
            qkv_input,
            baseline_q,
            baseline_k,
            baseline_v,
            baseline_sdpa,
            candidate_q,
            candidate_k,
            candidate_v,
            candidate_sdpa,
        )
        mx.synchronize()
        qkv_norm_rotary_layout_parity = {
            "q": _diff_stats(baseline_q, candidate_q),
            "k": _diff_stats(baseline_k, candidate_k),
            "v": _diff_stats(baseline_v, candidate_v),
            "sdpa": _diff_stats(baseline_sdpa, candidate_sdpa),
        }
        qkv_norm_rotary_layout_shape_dtype = {
            "input_shape": list(qkv_input.shape),
            "input_dtype": str(qkv_input.dtype),
            "baseline_q_shape": list(baseline_q.shape),
            "candidate_q_shape": list(candidate_q.shape),
            "baseline_k_shape": list(baseline_k.shape),
            "candidate_k_shape": list(candidate_k.shape),
            "baseline_v_shape": list(baseline_v.shape),
            "candidate_v_shape": list(candidate_v.shape),
            "baseline_sdpa_shape": list(baseline_sdpa.shape),
            "candidate_sdpa_shape": list(candidate_sdpa.shape),
            "baseline_q_dtype": str(baseline_q.dtype),
            "candidate_q_dtype": str(candidate_q.dtype),
            "baseline_k_dtype": str(baseline_k.dtype),
            "candidate_k_dtype": str(candidate_k.dtype),
            "baseline_v_dtype": str(baseline_v.dtype),
            "candidate_v_dtype": str(candidate_v.dtype),
            "baseline_sdpa_dtype": str(baseline_sdpa.dtype),
            "candidate_sdpa_dtype": str(candidate_sdpa.dtype),
            "cos_shape": list(rotary[0].shape),
            "sin_shape": list(rotary[1].shape),
            "cos_dtype": str(rotary[0].dtype),
            "sin_dtype": str(rotary[1].dtype),
            "sdpa_layout": "[B,H,S,D]",
            "kernel_input_layout": "projected qkv reshaped as [B,S,H,3,D] plus cos/sin [S,R]",
            "kernel_outputs": "one RoPE-applied q_out, one RoPE-applied k_out, and one v_out tensor from a single mx.fast.metal_kernel launch",
            "shapes_match": bool(
                baseline_q.shape == candidate_q.shape
                and baseline_k.shape == candidate_k.shape
                and baseline_v.shape == candidate_v.shape
                and baseline_sdpa.shape == candidate_sdpa.shape
            ),
            "dtypes_match": bool(
                baseline_q.dtype == candidate_q.dtype
                and baseline_k.dtype == candidate_k.dtype
                and baseline_v.dtype == candidate_v.dtype
                and baseline_sdpa.dtype == candidate_sdpa.dtype
            ),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 9371,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    except Exception as exc:
        after = _metrics()
        return {
            "name": "attention_qkv_rmsnorm_rotary_sdpa_metal",
            "target_segment": "qk_rmsnorm_rope_layout",
            "target_boundary": "q/k RMSNorm, q/k RoPE, and q/k/v materialization into [B,H,S,D] immediately after qkv_proj",
            "opt_in_only": True,
            "disabled_by_default": True,
            "default_behavior_unchanged": True,
            "production_integrated": True,
            "strict_exact_semantics": False,
            "candidate_available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics_before": before,
            "metrics_after": after,
            "metrics_delta": _delta(before, after),
            "promote": False,
            "decision": "blocked_unsupported",
            "decision_reason": "custom Metal q/k RMSNorm+RoPE SDPA-layout kernel could not be constructed or executed locally",
        }
    finally:
        for flag, value in original_flags.items():
            setattr(block.attn, flag, value)

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    qkv_norm_rotary_layout_parity_ok = all(
        tensor_stats["max_abs"] <= args.parity_atol and tensor_stats["rel_l2"] <= args.parity_rel_l2
        for tensor_stats in qkv_norm_rotary_layout_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(qkv_norm_rotary_layout_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "custom Metal q/k RMSNorm+RoPE SDPA-layout path is not within the configured DiT parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the custom Metal q/k RMSNorm+RoPE SDPA-layout candidate is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the custom Metal q/k RMSNorm+RoPE SDPA-layout "
            "candidate from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "custom Metal q/k RMSNorm+RoPE SDPA-layout candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "custom Metal q/k RMSNorm+RoPE SDPA-layout candidate satisfies parity bounds, is disabled by default, "
            "memory-clean, and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_qkv_norm_rotary_layout_zero = all(
        tensor_stats["max_abs"] == 0.0 and tensor_stats["rel_l2"] == 0.0
        for tensor_stats in qkv_norm_rotary_layout_parity.values()
    )
    return {
        "name": "attention_qkv_rmsnorm_rotary_sdpa_metal",
        "target_segment": "qk_rmsnorm_rope_layout",
        "target_boundary": "q/k RMSNorm, q/k RoPE, and q/k/v materialization into [B,H,S,D] immediately after qkv_proj",
        "selection_rationale": (
            f"Current segmented medians are qkv_quantized_matmul={qkv_median} s, "
            f"qk_rmsnorm_rope_layout={layout_median} s, and mlx_fast_sdpa={sdpa_median} s. "
            "Prior Attention probes split projection rank, q/k RMSNorm-layout, RoPE-only, pre-SDPA contiguity, or post-SDPA layout. "
            "This single-variable candidate leaves QMM and SDPA arithmetic unchanged while replacing native q/k RMSNorm, q/k RoPE, "
            "and q/k/v transpose materialization with one custom Metal kernel that writes SDPA-ready tensors after qkv_proj."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": False,
        "parity_bounded_not_assumed_exact": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {
                "use_qkv_2d_projection_candidate": False,
                "use_out_2d_projection_candidate": False,
                "use_qkv_pretranspose_layout_candidate": False,
                "use_qkv_rmsnorm_sdpa_metal_candidate": False,
                "use_qkv_rmsnorm_rotary_sdpa_metal_candidate": False,
                "use_rotary_qk_metal_candidate": False,
                "use_pre_sdpa_contiguous_candidate": False,
                "use_sdpa_out_layout_metal_candidate": False,
            },
            "enabled_flag_for_this_run": "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
            "single_variable_guard": "prior Attention projection-rank, pretranspose, q/k RMSNorm-layout, RoPE-only, pre-SDPA, and post-SDPA flags are forced off during this probe",
            "lora_path": "falls back to the existing baseline q/k/v plus apply_rotary path whenever lora is not None",
            "kernel": "mx.fast.metal_kernel reads [B,S,H,3,D] qkv, q/k RMSNorm weights, and [S,R] cos/sin, then writes RoPE-applied q/k/v [B,H,S,D] tensors",
        },
        "qkv_norm_rotary_layout_shape_dtype_contract": qkv_norm_rotary_layout_shape_dtype,
        "qkv_norm_rotary_layout_parity": qkv_norm_rotary_layout_parity,
        "qkv_norm_rotary_layout_parity_ok": qkv_norm_rotary_layout_parity_ok,
        "strict_qkv_norm_rotary_layout_parity_zero": strict_qkv_norm_rotary_layout_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention flag plus a cached custom Metal kernel helper; no weights, quantization, sigma/NFE, cache, or projection-rank change",
            "compile_cost": "first candidate call records custom Metal JIT/setup cost; warm interleaved samples measure the cached kernel path",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other sequence lengths and dtypes must be remeasured",
            "maintainability": "local q/k RMSNorm+RoPE plus SDPA-layout scheduling switch with explicit LoRA fallback; prior Attention candidates are forced off during this probe",
            "strict_equivalence": "custom Metal RMSNorm reduction and fused BF16 arithmetic order are not assumed exact; direct q/k/v/SDPA parity and full block parity are checked before timing promotion",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, Amdahl contribution, memory, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_attention_qkv_rmsnorm_sdpa_metal(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe fusing q/k RMSNorm with SDPA-layout materialization in one Metal kernel."""

    qkv_median = segment_stats.get("qkv_quantized_matmul", {}).get("median_seconds")
    layout_median = segment_stats.get("qk_rmsnorm_rope_layout", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    original_norm_layout_flag = bool(getattr(block.attn, "use_qkv_rmsnorm_sdpa_metal_candidate", False))
    original_rotary_flag = bool(getattr(block.attn, "use_rotary_qk_metal_candidate", False))
    original_pretranspose_flag = bool(getattr(block.attn, "use_qkv_pretranspose_layout_candidate", False))
    original_qkv_2d_flag = bool(getattr(block.attn, "use_qkv_2d_projection_candidate", False))
    original_out_2d_flag = bool(getattr(block.attn, "use_out_2d_projection_candidate", False))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine it with prior Attention candidates.
        block.attn.use_qkv_2d_projection_candidate = False
        block.attn.use_out_2d_projection_candidate = False
        block.attn.use_qkv_pretranspose_layout_candidate = False
        block.attn.use_rotary_qk_metal_candidate = False
        block.attn.use_qkv_rmsnorm_sdpa_metal_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        qkv_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        baseline_q, baseline_k, baseline_v = block.attn._qkv_sdpa_tensors(qkv_input)
        set_candidate(True)
        candidate_q, candidate_k, candidate_v = block.attn._qkv_sdpa_tensors(qkv_input)
        mx.eval(qkv_input, baseline_q, baseline_k, baseline_v, candidate_q, candidate_k, candidate_v)
        mx.synchronize()
        qkv_norm_layout_parity = {
            "q": _diff_stats(baseline_q, candidate_q),
            "k": _diff_stats(baseline_k, candidate_k),
            "v": _diff_stats(baseline_v, candidate_v),
        }
        qkv_norm_layout_shape_dtype = {
            "input_shape": list(qkv_input.shape),
            "input_dtype": str(qkv_input.dtype),
            "baseline_q_shape": list(baseline_q.shape),
            "candidate_q_shape": list(candidate_q.shape),
            "baseline_k_shape": list(baseline_k.shape),
            "candidate_k_shape": list(candidate_k.shape),
            "baseline_v_shape": list(baseline_v.shape),
            "candidate_v_shape": list(candidate_v.shape),
            "baseline_q_dtype": str(baseline_q.dtype),
            "candidate_q_dtype": str(candidate_q.dtype),
            "baseline_k_dtype": str(baseline_k.dtype),
            "candidate_k_dtype": str(candidate_k.dtype),
            "baseline_v_dtype": str(baseline_v.dtype),
            "candidate_v_dtype": str(candidate_v.dtype),
            "sdpa_layout": "[B,H,S,D]",
            "kernel_input_layout": "projected qkv reshaped as [B,S,H,3,D]",
            "kernel_outputs": "one q_out, one k_out, and one v_out tensor from a single mx.fast.metal_kernel launch",
            "shapes_match": bool(
                baseline_q.shape == candidate_q.shape
                and baseline_k.shape == candidate_k.shape
                and baseline_v.shape == candidate_v.shape
            ),
            "dtypes_match": bool(
                baseline_q.dtype == candidate_q.dtype
                and baseline_k.dtype == candidate_k.dtype
                and baseline_v.dtype == candidate_v.dtype
            ),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 7817,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    except Exception as exc:
        after = _metrics()
        return {
            "name": "attention_qkv_rmsnorm_sdpa_metal",
            "target_segment": "qk_rmsnorm_rope_layout",
            "target_boundary": "q/k RMSNorm plus q/k/v materialization into [B,H,S,D] immediately after qkv_proj",
            "opt_in_only": True,
            "disabled_by_default": True,
            "default_behavior_unchanged": True,
            "production_integrated": True,
            "strict_exact_semantics": False,
            "candidate_available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics_before": before,
            "metrics_after": after,
            "metrics_delta": _delta(before, after),
            "promote": False,
            "decision": "blocked_unsupported",
            "decision_reason": "custom Metal q/k RMSNorm SDPA-layout kernel could not be constructed or executed locally",
        }
    finally:
        block.attn.use_qkv_rmsnorm_sdpa_metal_candidate = original_norm_layout_flag
        block.attn.use_rotary_qk_metal_candidate = original_rotary_flag
        block.attn.use_qkv_pretranspose_layout_candidate = original_pretranspose_flag
        block.attn.use_qkv_2d_projection_candidate = original_qkv_2d_flag
        block.attn.use_out_2d_projection_candidate = original_out_2d_flag

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    qkv_norm_layout_parity_ok = all(
        tensor_stats["max_abs"] <= args.parity_atol and tensor_stats["rel_l2"] <= args.parity_rel_l2
        for tensor_stats in qkv_norm_layout_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(qkv_norm_layout_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "custom Metal q/k RMSNorm SDPA-layout path is not within the configured DiT parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the custom Metal q/k RMSNorm SDPA-layout candidate is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the custom Metal q/k RMSNorm SDPA-layout "
            "candidate from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "custom Metal q/k RMSNorm SDPA-layout candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "custom Metal q/k RMSNorm SDPA-layout candidate satisfies parity bounds, is disabled by default, "
            "memory-clean, and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_qkv_norm_layout_zero = all(
        tensor_stats["max_abs"] == 0.0 and tensor_stats["rel_l2"] == 0.0
        for tensor_stats in qkv_norm_layout_parity.values()
    )
    return {
        "name": "attention_qkv_rmsnorm_sdpa_metal",
        "target_segment": "qk_rmsnorm_rope_layout",
        "target_boundary": "q/k RMSNorm plus q/k/v materialization into [B,H,S,D] immediately after qkv_proj",
        "selection_rationale": (
            f"Current segmented medians are qkv_quantized_matmul={qkv_median} s, "
            f"qk_rmsnorm_rope_layout={layout_median} s, and mlx_fast_sdpa={sdpa_median} s. "
            "Prior Attention probes changed projection rank, transpose scheduling, or q/k RoPE. This single-variable "
            "candidate leaves QMM, RoPE, and SDPA unchanged while replacing native q/k RMSNorm plus q/k/v transpose "
            "materialization with one custom Metal kernel that writes SDPA-ready tensors after qkv_proj."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": False,
        "parity_bounded_not_assumed_exact": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {
                "use_qkv_2d_projection_candidate": False,
                "use_out_2d_projection_candidate": False,
                "use_qkv_pretranspose_layout_candidate": False,
                "use_qkv_rmsnorm_sdpa_metal_candidate": False,
                "use_rotary_qk_metal_candidate": False,
            },
            "enabled_flag_for_this_run": "use_qkv_rmsnorm_sdpa_metal_candidate",
            "single_variable_guard": "prior Attention projection-rank, pretranspose, and RoPE Metal candidate flags are forced off during this probe",
            "lora_path": "falls back to the existing baseline q/k/v layout path whenever lora is not None",
            "kernel": "mx.fast.metal_kernel reads [B,S,H,3,D] qkv and q/k RMSNorm weights, then writes q/k/v [B,H,S,D] tensors",
        },
        "qkv_norm_layout_shape_dtype_contract": qkv_norm_layout_shape_dtype,
        "qkv_norm_layout_parity": qkv_norm_layout_parity,
        "qkv_norm_layout_parity_ok": qkv_norm_layout_parity_ok,
        "strict_qkv_norm_layout_parity_zero": strict_qkv_norm_layout_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention flag plus a cached custom Metal kernel helper; no weights, quantization, sigma/NFE, cache, or projection-rank change",
            "compile_cost": "first candidate call records custom Metal JIT/setup cost; warm interleaved samples measure the cached kernel path",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other sequence lengths and dtypes must be remeasured",
            "maintainability": "local q/k RMSNorm plus SDPA-layout scheduling switch with explicit LoRA fallback; prior Attention candidates are forced off during this probe",
            "strict_equivalence": "custom Metal RMSNorm reduction is not assumed exact; direct q/k/v layout parity and full block parity are checked before timing promotion",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, Amdahl contribution, memory, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_attention_pre_out_proj_contiguous(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe explicit ``mx.contiguous`` materialization immediately before ``Attention.out_proj``."""

    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    out_proj_median = segment_stats.get("out_projection", {}).get("median_seconds")
    original_block_flags = {
        "use_packed_adaln_gather_candidate": bool(getattr(block, "use_packed_adaln_gather_candidate", False)),
        "use_indexed_adaln_affine_metal_candidate": bool(
            getattr(block, "use_indexed_adaln_affine_metal_candidate", False)
        ),
        "use_indexed_gated_residual_metal_candidate": bool(
            getattr(block, "use_indexed_gated_residual_metal_candidate", False)
        ),
    }
    mlp_candidate_flags = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
    )
    original_mlp_flags = {flag: bool(getattr(block.mlp, flag, False)) for flag in mlp_candidate_flags}
    original_chunk_size = int(getattr(block.mlp, "ffn_sequence_chunk_size", 512))
    original_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 512))
    attention_candidate_flags = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    original_attn_flags = {flag: bool(getattr(block.attn, flag, False)) for flag in attention_candidate_flags}

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: all prior block, FFN, and Attention candidates are off.
        for flag in original_block_flags:
            setattr(block, flag, False)
        for flag in mlp_candidate_flags:
            setattr(block.mlp, flag, False)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_tile_size
        for flag in attention_candidate_flags:
            setattr(block.attn, flag, False)
        block.attn.use_pre_out_proj_contiguous_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        out_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "out_proj")
        baseline_input = block.attn._pre_out_project_input(out_input)
        baseline_projection = block.attn._out_project(out_input)
        inner_dim = int(getattr(block.attn, "_inner", block.attn.heads * block.attn.head_dim))
        direct_input = materialize_attention_output_contiguous(out_input, inner_dim)
        set_candidate(True)
        candidate_input = block.attn._pre_out_project_input(out_input)
        candidate_projection = block.attn._out_project(out_input)
        mx.eval(out_input, baseline_input, direct_input, candidate_input, baseline_projection, candidate_projection)
        mx.synchronize()
        output_materialization_parity = {
            "method": _diff_stats(baseline_input, candidate_input),
            "direct_helper": _diff_stats(out_input, direct_input),
        }
        out_projection_parity = _diff_stats(baseline_projection, candidate_projection)
        output_shape_dtype = {
            "attention_out_proj_input_shape": list(out_input.shape),
            "attention_out_proj_input_dtype": str(out_input.dtype),
            "candidate_input_shape": list(candidate_input.shape),
            "candidate_input_dtype": str(candidate_input.dtype),
            "direct_input_shape": list(direct_input.shape),
            "direct_input_dtype": str(direct_input.dtype),
            "baseline_projection_shape": list(baseline_projection.shape),
            "candidate_projection_shape": list(candidate_projection.shape),
            "baseline_projection_dtype": str(baseline_projection.dtype),
            "candidate_projection_dtype": str(candidate_projection.dtype),
            "input_shape_matches_baseline": bool(out_input.shape == candidate_input.shape == direct_input.shape),
            "input_dtype_matches_baseline": bool(out_input.dtype == candidate_input.dtype == direct_input.dtype),
            "projection_shape_matches_baseline": bool(baseline_projection.shape == candidate_projection.shape),
            "projection_dtype_matches_baseline": bool(baseline_projection.dtype == candidate_projection.dtype),
            "input_layout": "[B,S,H*D] merged heads from SDPA output transpose/reshape",
            "materialization_api": "mx.contiguous(attention_output) immediately before Attention.out_proj",
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 9059,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        for flag, value in original_block_flags.items():
            setattr(block, flag, value)
        for flag, value in original_mlp_flags.items():
            setattr(block.mlp, flag, value)
        block.mlp.ffn_sequence_chunk_size = original_chunk_size
        block.mlp.ffn_fc2_tiled_output_channels = original_tile_size
        for flag, value in original_attn_flags.items():
            setattr(block.attn, flag, value)

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    output_parity_ok = all(
        stats_["max_abs"] <= args.parity_atol and stats_["rel_l2"] <= args.parity_rel_l2
        for stats_ in output_materialization_parity.values()
    )
    out_projection_parity_ok = (
        out_projection_parity["max_abs"] <= args.parity_atol
        and out_projection_parity["rel_l2"] <= args.parity_rel_l2
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(output_parity_ok and out_projection_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "pre-out_proj contiguous materialization changed input, out projection, or full-block outputs beyond configured bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says pre-out_proj Attention output contiguous materialization is slower than baseline"
        promoted = False
    elif not stable_faster:
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        decision = "reject_unproven_noise"
        reason = (
            "interleaved timing and bootstrap CI do not separate pre-out_proj Attention output contiguous "
            "materialization from baseline; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "pre-out_proj Attention output contiguous materialization is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "pre-out_proj Attention output contiguous materialization is strictly equivalent, disabled by default, "
            "memory-clean, and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_output_zero = all(
        stats_["max_abs"] == 0.0 and stats_["rel_l2"] == 0.0 for stats_ in output_materialization_parity.values()
    )
    strict_out_projection_zero = out_projection_parity["max_abs"] == 0.0 and out_projection_parity["rel_l2"] == 0.0
    strict_full_zero = (
        parity["max_abs"] == 0.0
        and parity["rel_l2"] == 0.0
        and first_parity["max_abs"] == 0.0
        and first_parity["rel_l2"] == 0.0
    )
    return {
        "name": "attention_pre_out_proj_contiguous",
        "target_segment": "out_projection",
        "target_boundary": "merged SDPA Attention output materialized with mx.contiguous immediately before Attention.out_proj quantized projection",
        "selection_rationale": (
            f"Current segmented medians are mlx_fast_sdpa={sdpa_median} s and out_projection={out_proj_median} s "
            f"at sequence length {sequence_meta.get('sequence_length')}. The out_projection segment includes the "
            "post-SDPA transpose/reshape and quantized out_proj QMM. Prior Attention probes changed projection rank, "
            "QKV layout, q/k RMSNorm/RoPE fusion, SDPA input layout, or replaced the post-SDPA layout copy with Metal. "
            "This single-variable probe leaves q/k/v, RoPE, SDPA arithmetic, merged-head element order, QMM rank, "
            "out_proj weights, and default generation unchanged while testing whether an explicit contiguous buffer "
            "right before out_proj reduces hidden materialization/dispatch cost."
        ),
        "opt_in_only": True,
        "profiler_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {
                "use_pre_qkv_contiguous_candidate": False,
                "use_qkv_2d_projection_candidate": False,
                "use_out_2d_projection_candidate": False,
                "use_out_dense_dequant_candidate": False,
                "use_qkv_pretranspose_layout_candidate": False,
                "use_qkv_rmsnorm_sdpa_metal_candidate": False,
                "use_qkv_rmsnorm_rotary_sdpa_metal_candidate": False,
                "use_rotary_qk_metal_candidate": False,
                "use_pre_sdpa_contiguous_candidate": False,
                "use_sdpa_out_layout_metal_candidate": False,
                "use_pre_out_proj_contiguous_candidate": False,
            },
            "enabled_flag_for_this_run": "use_pre_out_proj_contiguous_candidate",
            "helper": "Attention._pre_out_project_input -> materialize_attention_output_contiguous",
            "single_variable_guard": (
                "prior block, Attention, and FFN candidate flags are forced off during this probe; "
                "only use_pre_out_proj_contiguous_candidate is toggled"
            ),
            "lora_path": "falls back to the existing non-contiguous LoRA path whenever lora is not None",
        },
        "output_shape_dtype_contract": output_shape_dtype,
        "output_materialization_parity": output_materialization_parity,
        "output_materialization_parity_ok": output_parity_ok,
        "out_projection_parity": out_projection_parity,
        "out_projection_parity_ok": out_projection_parity_ok,
        "strict_output_materialization_parity_zero": strict_output_zero,
        "strict_out_projection_parity_zero": strict_out_projection_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": strict_full_zero,
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one pure mx.contiguous helper plus one disabled-by-default Attention flag; no weights, quantization, sigma/NFE, cache, projection rank, custom Metal, or default generation path changes",
            "compile_cost": "no mx.compile, custom Metal kernel, or persistent compiler cache is introduced; first call records ordinary MLX lazy/kernel setup cost",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas; candidate may allocate one explicit merged-head Attention output buffer before out_proj",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other shapes and dtypes must be remeasured before promotion",
            "maintainability": "local pre-out_proj materialization switch with explicit LoRA fallback; other candidate flags are forced off during this probe",
            "strict_equivalence": "output materialization, out projection output, and full-block output are checked against the baseline before any timing decision",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, memory, Amdahl contribution, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_attention_sdpa_out_layout_metal(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe a pure Metal copy from SDPA ``[B,H,S,D]`` output to merged-head ``[B,S,H*D]``."""

    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    out_proj_median = segment_stats.get("out_projection", {}).get("median_seconds")
    original_sdpa_out_flag = bool(getattr(block.attn, "use_sdpa_out_layout_metal_candidate", False))
    original_norm_layout_flag = bool(getattr(block.attn, "use_qkv_rmsnorm_sdpa_metal_candidate", False))
    original_rotary_flag = bool(getattr(block.attn, "use_rotary_qk_metal_candidate", False))
    original_pretranspose_flag = bool(getattr(block.attn, "use_qkv_pretranspose_layout_candidate", False))
    original_qkv_2d_flag = bool(getattr(block.attn, "use_qkv_2d_projection_candidate", False))
    original_out_2d_flag = bool(getattr(block.attn, "use_out_2d_projection_candidate", False))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine it with prior Attention candidates.
        block.attn.use_qkv_2d_projection_candidate = False
        block.attn.use_out_2d_projection_candidate = False
        block.attn.use_qkv_pretranspose_layout_candidate = False
        block.attn.use_qkv_rmsnorm_sdpa_metal_candidate = False
        block.attn.use_rotary_qk_metal_candidate = False
        block.attn.use_sdpa_out_layout_metal_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        attn_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        B, S, _ = attn_input.shape
        q, k, v = block.attn._qkv_sdpa_tensors(attn_input)
        q = apply_rotary(q, *rotary)
        k = apply_rotary(k, *rotary)
        sdpa_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=block.attn.scale, mask=None)
        baseline_layout = sdpa_out.transpose(0, 2, 1, 3).reshape(B, S, block.attn.heads * block.attn.head_dim)
        candidate_layout = sdpa_out_to_bshd_metal(sdpa_out)
        mx.eval(attn_input, sdpa_out, baseline_layout, candidate_layout)
        mx.synchronize()
        layout_parity = _diff_stats(baseline_layout, candidate_layout)
        layout_shape_dtype = {
            "attention_input_shape": list(attn_input.shape),
            "attention_input_dtype": str(attn_input.dtype),
            "sdpa_output_shape": list(sdpa_out.shape),
            "sdpa_output_dtype": str(sdpa_out.dtype),
            "baseline_layout_shape": list(baseline_layout.shape),
            "candidate_layout_shape": list(candidate_layout.shape),
            "baseline_layout_dtype": str(baseline_layout.dtype),
            "candidate_layout_dtype": str(candidate_layout.dtype),
            "input_layout": "[B,H,S,D] SDPA output",
            "output_layout": "[B,S,H*D] merged heads before out_proj",
            "kernel_outputs": "one merged-head tensor from a single mx.fast.metal_kernel launch",
            "shapes_match": bool(baseline_layout.shape == candidate_layout.shape),
            "dtypes_match": bool(baseline_layout.dtype == candidate_layout.dtype),
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 8423,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    except Exception as exc:
        after = _metrics()
        return {
            "name": "attention_sdpa_out_layout_metal",
            "target_segment": "out_projection",
            "target_boundary": "post-SDPA layout copy from [B,H,S,D] to [B,S,H*D] immediately before attn.out_proj",
            "opt_in_only": True,
            "disabled_by_default": True,
            "default_behavior_unchanged": True,
            "production_integrated": True,
            "strict_exact_semantics": True,
            "candidate_available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics_before": before,
            "metrics_after": after,
            "metrics_delta": _delta(before, after),
            "promote": False,
            "decision": "blocked_unsupported",
            "decision_reason": "custom Metal SDPA-output layout kernel could not be constructed or executed locally",
        }
    finally:
        block.attn.use_sdpa_out_layout_metal_candidate = original_sdpa_out_flag
        block.attn.use_qkv_rmsnorm_sdpa_metal_candidate = original_norm_layout_flag
        block.attn.use_rotary_qk_metal_candidate = original_rotary_flag
        block.attn.use_qkv_pretranspose_layout_candidate = original_pretranspose_flag
        block.attn.use_qkv_2d_projection_candidate = original_qkv_2d_flag
        block.attn.use_out_2d_projection_candidate = original_out_2d_flag

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    layout_exact = layout_parity["max_abs"] == 0.0 and layout_parity["rel_l2"] == 0.0
    layout_parity_ok = layout_exact and layout_shape_dtype["shapes_match"] and layout_shape_dtype["dtypes_match"]
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(layout_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "custom Metal SDPA-output layout path is not exact or is outside the configured DiT parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the custom Metal SDPA-output layout candidate is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the custom Metal SDPA-output layout "
            "candidate from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "custom Metal SDPA-output layout candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "custom Metal SDPA-output layout candidate is exact, disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    return {
        "name": "attention_sdpa_out_layout_metal",
        "target_segment": "out_projection",
        "target_boundary": "post-SDPA layout copy from [B,H,S,D] to [B,S,H*D] immediately before attn.out_proj",
        "selection_rationale": (
            f"Current segmented medians are mlx_fast_sdpa={sdpa_median} s and out_projection={out_proj_median} s. "
            "The out_projection segment includes the baseline post-SDPA transpose/reshape materialization before "
            "the quantized output projection. Prior Attention probes changed projection rank, QKV layout, q/k RMSNorm, "
            "or RoPE. This single-variable candidate leaves QMM, q/k/v, RoPE, SDPA arithmetic, and out_proj unchanged "
            "while replacing only the post-SDPA merged-head layout copy with one custom Metal kernel."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {
                "use_qkv_2d_projection_candidate": False,
                "use_out_2d_projection_candidate": False,
                "use_qkv_pretranspose_layout_candidate": False,
                "use_qkv_rmsnorm_sdpa_metal_candidate": False,
                "use_rotary_qk_metal_candidate": False,
                "use_sdpa_out_layout_metal_candidate": False,
            },
            "enabled_flag_for_this_run": "use_sdpa_out_layout_metal_candidate",
            "single_variable_guard": "prior Attention projection-rank, pretranspose, q/k RMSNorm-layout, and RoPE candidate flags are forced off during this probe",
            "lora_path": "falls back to the existing baseline post-SDPA transpose/reshape path whenever lora is not None",
            "kernel": "mx.fast.metal_kernel reads SDPA [B,H,S,D] and writes merged-head [B,S,H*D] without arithmetic",
        },
        "layout_shape_dtype_contract": layout_shape_dtype,
        "layout_parity": layout_parity,
        "layout_parity_exact": layout_exact,
        "layout_parity_ok": layout_parity_ok,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention flag plus a cached custom Metal copy kernel helper; no weights, quantization, sigma/NFE, cache, or projection-rank change",
            "compile_cost": "first candidate call records custom Metal JIT/setup cost; warm interleaved samples measure the cached kernel path",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other sequence lengths and dtypes must be remeasured",
            "maintainability": "local post-SDPA layout scheduling switch with explicit LoRA fallback; prior Attention candidates are forced off during this probe",
            "strict_equivalence": "kernel performs no arithmetic; direct layout parity must be exact before timing promotion",
            "active_no_fixed_threshold_directive": "decision uses exact/tolerance parity, CI/noise, Amdahl contribution, memory, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_attention_rotary_qk_metal(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe q/k rotary application with one custom Metal kernel before SDPA."""

    qkv_median = segment_stats.get("qkv_quantized_matmul", {}).get("median_seconds")
    layout_median = segment_stats.get("qk_rmsnorm_rope_layout", {}).get("median_seconds")
    sdpa_median = segment_stats.get("mlx_fast_sdpa", {}).get("median_seconds")
    original_rotary_flag = bool(getattr(block.attn, "use_rotary_qk_metal_candidate", False))
    original_pretranspose_flag = bool(getattr(block.attn, "use_qkv_pretranspose_layout_candidate", False))
    original_qkv_2d_flag = bool(getattr(block.attn, "use_qkv_2d_projection_candidate", False))
    original_out_2d_flag = bool(getattr(block.attn, "use_out_2d_projection_candidate", False))

    def set_candidate(enabled: bool) -> None:
        # Keep this probe single-variable: do not combine it with prior Attention candidates.
        block.attn.use_qkv_2d_projection_candidate = False
        block.attn.use_out_2d_projection_candidate = False
        block.attn.use_qkv_pretranspose_layout_candidate = False
        block.attn.use_rotary_qk_metal_candidate = bool(enabled)

    def baseline_forward() -> mx.array:
        set_candidate(False)
        return block(x, modulation, adaln_indices, rotary)

    def candidate_forward() -> mx.array:
        set_candidate(True)
        return block(x, modulation, adaln_indices, rotary)

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_candidate(False)
        qkv_input = _attention_projection_input(block, x, modulation, adaln_indices, rotary, "qkv_proj")
        q, k, _v = block.attn._qkv_sdpa_tensors(qkv_input)
        baseline_q = apply_rotary(q, *rotary)
        baseline_k = apply_rotary(k, *rotary)
        metal_q, metal_k = apply_rotary_qk_metal(q, k, *rotary)
        mx.eval(qkv_input, q, k, baseline_q, baseline_k, metal_q, metal_k)
        mx.synchronize()
        rotary_parity = {
            "q": _diff_stats(baseline_q, metal_q),
            "k": _diff_stats(baseline_k, metal_k),
        }
        rotary_shape_dtype = {
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "cos_shape": list(rotary[0].shape),
            "sin_shape": list(rotary[1].shape),
            "q_dtype": str(q.dtype),
            "k_dtype": str(k.dtype),
            "cos_dtype": str(rotary[0].dtype),
            "sin_dtype": str(rotary[1].dtype),
            "metal_q_shape": list(metal_q.shape),
            "metal_k_shape": list(metal_k.shape),
            "metal_q_dtype": str(metal_q.dtype),
            "metal_k_dtype": str(metal_k.dtype),
            "shape_matches_baseline": bool(baseline_q.shape == metal_q.shape and baseline_k.shape == metal_k.shape),
            "dtype_matches_baseline": bool(baseline_q.dtype == metal_q.dtype and baseline_k.dtype == metal_k.dtype),
            "sdpa_layout": "[B,H,S,D]",
            "kernel_outputs": "one q_out and one k_out tensor from a single mx.fast.metal_kernel launch",
        }

        first_started = time.perf_counter()
        first = candidate_forward()
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 6421,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    except Exception as exc:
        after = _metrics()
        return {
            "name": "attention_rotary_qk_metal",
            "target_segment": "qk_rmsnorm_rope_layout",
            "target_boundary": "q/k rotary application over [B,H,S,D] before scaled-dot-product attention",
            "opt_in_only": True,
            "disabled_by_default": True,
            "default_behavior_unchanged": True,
            "production_integrated": True,
            "strict_exact_semantics": False,
            "candidate_available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics_before": before,
            "metrics_after": after,
            "metrics_delta": _delta(before, after),
            "promote": False,
            "decision": "blocked_unsupported",
            "decision_reason": "custom Metal q/k RoPE kernel could not be constructed or executed locally",
        }
    finally:
        block.attn.use_rotary_qk_metal_candidate = original_rotary_flag
        block.attn.use_qkv_pretranspose_layout_candidate = original_pretranspose_flag
        block.attn.use_qkv_2d_projection_candidate = original_qkv_2d_flag
        block.attn.use_out_2d_projection_candidate = original_out_2d_flag

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    rotary_parity_ok = all(
        tensor_stats["max_abs"] <= args.parity_atol and tensor_stats["rel_l2"] <= args.parity_rel_l2
        for tensor_stats in rotary_parity.values()
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    all_parity_ok = bool(rotary_parity_ok and parity_ok and first_parity_ok)
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "custom Metal q/k RoPE path is not within the configured strict DiT parity bounds"
        promoted = False
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the custom Metal q/k RoPE candidate is slower than baseline"
        promoted = False
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the custom Metal q/k RoPE candidate "
            "from measurement noise; no fixed percentage cutoff was used." + memory_suffix
        )
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "custom Metal q/k RoPE candidate is faster than noise, but pageout/swapout observation regressed"
        promoted = False
    else:
        decision = "accept_opt_in_candidate"
        reason = (
            "custom Metal q/k RoPE candidate satisfies parity bounds, is disabled by default, memory-clean, "
            "and faster than interleaved baseline outside measured noise"
        )
        promoted = True

    strict_rotary_zero = all(
        tensor_stats["max_abs"] == 0.0 and tensor_stats["rel_l2"] == 0.0
        for tensor_stats in rotary_parity.values()
    )
    return {
        "name": "attention_rotary_qk_metal",
        "target_segment": "qk_rmsnorm_rope_layout",
        "target_boundary": "q/k rotary application over [B,H,S,D] before scaled-dot-product attention",
        "selection_rationale": (
            f"Current segmented medians are qkv_quantized_matmul={qkv_median} s, "
            f"qk_rmsnorm_rope_layout={layout_median} s, and mlx_fast_sdpa={sdpa_median} s. "
            "Prior Attention probes changed projection rank or QKV transpose scheduling; this single-variable "
            "candidate leaves QMM, q/k RMSNorm, and SDPA unchanged while replacing only the two baseline "
            "apply_rotary slice/concat/multiply/add paths with one q/k custom Metal kernel."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": False,
        "parity_bounded_not_assumed_exact": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "candidate_available": True,
        "implementation_switch": {
            "class": "minimax_h3_mlx.dit.Attention",
            "default_flags": {
                "use_qkv_2d_projection_candidate": False,
                "use_out_2d_projection_candidate": False,
                "use_qkv_pretranspose_layout_candidate": False,
                "use_rotary_qk_metal_candidate": False,
            },
            "enabled_flag_for_this_run": "use_rotary_qk_metal_candidate",
            "single_variable_guard": "prior Attention projection-rank and pretranspose candidate flags are forced off during this probe",
            "lora_path": "falls back to the existing apply_rotary path whenever lora is not None",
            "kernel": "mx.fast.metal_kernel reads q, k, cos, sin and writes q_out/k_out for the full [B,H,S,D] tensors in one launch",
        },
        "rotary_shape_dtype_contract": rotary_shape_dtype,
        "rotary_parity": rotary_parity,
        "rotary_parity_ok": rotary_parity_ok,
        "strict_rotary_parity_zero": strict_rotary_zero,
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "strict_full_block_parity_zero": (
            parity["max_abs"] == 0.0
            and parity["rel_l2"] == 0.0
            and first_parity["max_abs"] == 0.0
            and first_parity["rel_l2"] == 0.0
        ),
        "parity_ok": all_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a full-generation rerun",
        },
        "tradeoff_summary": {
            "implementation_complexity": "one disabled-by-default Attention flag plus a cached custom Metal kernel helper; no weights, quantization, sigma/NFE, cache, or projection-rank change",
            "compile_cost": "first candidate call records custom Metal JIT/setup cost; warm interleaved samples measure the cached kernel path",
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache/RSS deltas",
            "resolution_scaling": "evidence is only for the selected 320x192 packed sequence; other sequence lengths and dtypes must be remeasured",
            "maintainability": "local q/k RoPE scheduling switch with explicit LoRA fallback; prior Attention candidates are forced off during this probe",
            "strict_equivalence": "custom Metal math is not assumed exact; direct q/k rotary parity and full block parity are checked before timing promotion",
            "active_no_fixed_threshold_directive": "decision uses parity bounds, CI/noise, Amdahl contribution, memory, and maintainability; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }



def _candidate_compile(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    def baseline_forward() -> mx.array:
        return block(x, modulation, adaln_indices, rotary)

    def forward(hidden: mx.array) -> mx.array:
        return block(hidden, modulation, adaln_indices, rotary)

    compiled = mx.compile(forward)
    _reset_mlx_peak()
    before = _metrics()
    started = time.perf_counter()
    first = compiled(x)
    mx.eval(first)
    mx.synchronize()
    compile_seconds = time.perf_counter() - started
    after_first = _metrics()
    first_parity = _diff_stats(baseline_out, first)

    interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
        baseline_forward,
        lambda: compiled(x),
        warmups=args.interleaved_warmups,
        repeats=args.interleaved_repeats,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed + 1701,
    )
    after = _metrics()
    parity = _diff_stats(paired_baseline_out, out)
    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    pageouts_delta = _delta(before, after).get("vm_pageouts")
    swapouts_delta = _delta(before, after).get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    noise_decision = interleaved.get("noise_decision")
    stable_faster = noise_decision == "candidate_faster_than_noise"
    stable_slower = noise_decision == "candidate_slower_than_noise"
    if not parity_ok or not first_parity_ok:
        decision = "reject_parity"
        reason = "strict-equivalence check failed, so the opt-in compile candidate cannot be used"
    elif stable_slower:
        decision = "reject_slower_than_noise"
        reason = "interleaved bootstrap CI says the compiled candidate is slower than baseline"
    elif not stable_faster:
        decision = "reject_unproven_noise"
        memory_suffix = " Memory observation also regressed during the interleaved candidate phase." if not memory_ok else ""
        reason = (
            "interleaved timing and bootstrap CI do not separate the candidate from measurement noise; "
            "the candidate remains disabled without using any fixed percentage cutoff." + memory_suffix
        )
    elif not memory_ok:
        decision = "reject_memory"
        reason = "candidate is faster than noise, but memory observation regressed during the interleaved candidate phase"
    elif stable_faster:
        decision = "reject_default_keep_experimental"
        reason = (
            "interleaved timing indicates a real warm-block improvement, but this stateless mx.compile "
            "candidate is not production-integrated and carries first-call compile/cache-shape lifecycle cost; "
            "reject promotion/default use and keep it only as an opt-in experimental diagnostic"
        )
    else:
        decision = "reject_unproven_noise"
        reason = (
            "interleaved timing and bootstrap CI do not separate the candidate from measurement noise; "
            "the candidate remains disabled without using any fixed percentage cutoff"
        )
    promoted = False
    return {
        "name": "stateless_mx_compile_full_block",
        "selection_rationale": (
            "Trace and segment evidence point away from the small Q/K RMSNorm+RoPE/AdaLN pointwise "
            "fragments: the dominant boundaries are 4-bit MLX affine_qmm dequant+GEMM calls in "
            "FFN, QKV, and out projection. Stateless mx.compile is therefore only a strict-exact, "
            "opt-in diagnostic to test whether MLX graph scheduling reduces materialization around "
            "those matmul-heavy boundaries; it is not a production generation optimization."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": False,
        "compile_first_call_seconds": compile_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "parity_vs_interleaved_baseline": parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "parity_ok": parity_ok and first_parity_ok,
        "noise_decision": noise_decision,
        "noise_evidence_supports_faster": stable_faster,
        "memory_gate_ok": memory_ok,
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": _delta(before, after),
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block-0 delta to every DiT block and denoiser evaluation; this is an upper-bound diagnostic, not a rerun of full generation",
        },
        "tradeoff_summary": {
            "compile_cost": (
                "first compiled call cost is recorded; compiling/caching every real block/shape would need "
                "explicit lifecycle management before production use"
            ),
            "memory": "candidate/interleaved phase reports pageout/swapout deltas and MLX peak/cache deltas",
            "resolution_scaling": "evidence is only for the 320x192 packed sequence; MLX may choose different kernels at larger S/M/N",
            "maintainability": "one-line diagnostic is simple, but default production integration is nontrivial because blocks, shapes, cache invalidation, LoRA, and capture/debug behavior must be managed",
            "strict_equivalence": "checked against both the pre-candidate baseline and the last interleaved baseline output",
            "active_no_fixed_threshold_directive": "decision uses CI/noise, Amdahl contribution, compile cost, memory, maintainability, resolution scaling, and strict parity; no uniform percent cutoff is applied",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def _candidate_promoted_dense_dequant_combo(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    baseline_out: mx.array,
    baseline_stats: dict[str, Any],
    segment_stats: dict[str, dict[str, Any]],
    cfg: DiTConfig,
    sequence_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Profile the promoted strict-parity dense-dequant production combo."""

    profile = str(args.dense_dequant_profile)
    if profile == DENSE_DEQUANT_PROFILE_OFF:
        return {
            "name": "promoted_dense_dequant_combo",
            "target_segment": "attention_qkv + attention_out + fc1_swiglu_fc2",
            "opt_in_only": True,
            "disabled_by_default": True,
            "candidate_available": False,
            "parity_ok": False,
            "memory_gate_ok": False,
            "promote": False,
            "decision": "reject_config",
            "decision_reason": "--dense-dequant-profile=off disables the combo; choose qkv-fc2-out-resident or qkv-fc2-out-tiled",
        }

    block_flag_names = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    attention_flag_names = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_out_tiled_dense_dequant_candidate",
        "use_qkv_tiled_dense_dequant_candidate",
        "use_qkv_headgroup_row_sliced_qmm_candidate",
        "use_qkv_input_chunked_qmm_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_head_batch_rank3_candidate",
        "use_sdpa_headgroup_split_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    ffn_flag_names = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc1_split_gate_value_quantized_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_fc2_input_chunked_qmm_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
        "use_ffn_subgraph_compile_candidate",
    )
    original_block_flags = {name: bool(getattr(block, name, False)) for name in block_flag_names}
    original_attention_flags = {name: bool(getattr(block.attn, name, False)) for name in attention_flag_names}
    original_ffn_flags = {name: bool(getattr(block.mlp, name, False)) for name in ffn_flag_names}
    original_qkv_tile_size = int(getattr(block.attn, "qkv_tiled_output_channels", 2048))
    original_out_tile_size = int(getattr(block.attn, "out_tiled_output_channels", 2048))
    original_fc2_tile_size = int(getattr(block.mlp, "ffn_fc2_tiled_output_channels", 1024))

    def restore_original_state() -> None:
        for name, value in original_block_flags.items():
            setattr(block, name, value)
        for name, value in original_attention_flags.items():
            setattr(block.attn, name, value)
        for name, value in original_ffn_flags.items():
            setattr(block.mlp, name, value)
        block.attn.qkv_tiled_output_channels = original_qkv_tile_size
        block.attn.out_tiled_output_channels = original_out_tile_size
        block.mlp.ffn_fc2_tiled_output_channels = original_fc2_tile_size

    def set_profile(enabled: bool) -> None:
        apply_dense_dequant_profile_to_block(
            block,
            profile if enabled else DENSE_DEQUANT_PROFILE_OFF,
            attention_qkv_tile_size=int(args.attention_qkv_tile_size),
            ffn_fc2_tile_size=int(args.ffn_fc2_tile_size),
            attention_out_tile_size=int(args.attention_out_tile_size),
        )

    _reset_mlx_peak()
    before = _metrics()
    try:
        set_profile(False)
        baseline_check = block(x, modulation, adaln_indices, rotary)
        mx.eval(baseline_check)
        mx.synchronize()
        baseline_check_parity = _diff_stats(baseline_out, baseline_check)

        first_started = time.perf_counter()
        set_profile(True)
        first = block(x, modulation, adaln_indices, rotary)
        mx.eval(first)
        mx.synchronize()
        first_call_seconds = time.perf_counter() - first_started
        after_first = _metrics()
        first_parity = _diff_stats(baseline_out, first)
        qkv_info = block.attn.qkv_tiled_dense_dequant_info(int(args.attention_qkv_tile_size))
        fc2_info = block.mlp.fc2_tiled_dense_dequant_info(int(args.ffn_fc2_tile_size))
        if profile == DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT:
            out_info = block.attn.out_dense_dequant_cache_info()
            out_path = "attention_out_dense_dequant"
        else:
            out_info = block.attn.out_tiled_dense_dequant_info(int(args.attention_out_tile_size))
            out_path = "attention_out_tiled_dense_dequant"

        def baseline_forward() -> mx.array:
            set_profile(False)
            return block(x, modulation, adaln_indices, rotary)

        def candidate_forward() -> mx.array:
            set_profile(True)
            return block(x, modulation, adaln_indices, rotary)

        interleaved, paired_baseline_out, out = _time_interleaved_pairwise(
            baseline_forward,
            candidate_forward,
            warmups=args.interleaved_warmups,
            repeats=args.interleaved_repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed + 7919,
        )
        after = _metrics()
        parity = _diff_stats(paired_baseline_out, out)
    finally:
        restore_original_state()

    base_timing = interleaved["baseline_timing"]
    stats = interleaved["candidate_timing"]
    base_median = base_timing.get("median_seconds")
    cand_median = stats.get("median_seconds")
    speedup = (base_median / cand_median) if base_median and cand_median else None
    block_delta = (float(base_median) - float(cand_median)) if base_median and cand_median else None
    block_relative_delta = (block_delta / float(base_median)) if block_delta is not None and base_median else None
    block_calls = int(cfg.num_layers) * int(args.sigma_grid_points - 1)
    e2e_saving = (block_delta * block_calls) if block_delta is not None else None
    fixed_e2e_fraction = (
        e2e_saving / float(args.fixed_e2e_baseline_seconds)
        if e2e_saving is not None and args.fixed_e2e_baseline_seconds
        else None
    )
    baseline_check_ok = baseline_check_parity["max_abs"] <= args.parity_atol and baseline_check_parity["rel_l2"] <= args.parity_rel_l2
    first_parity_ok = first_parity["max_abs"] <= args.parity_atol and first_parity["rel_l2"] <= args.parity_rel_l2
    parity_ok = parity["max_abs"] <= args.parity_atol and parity["rel_l2"] <= args.parity_rel_l2
    all_parity_ok = bool(baseline_check_ok and first_parity_ok and parity_ok)
    metrics_delta = _delta(before, after)
    pageouts_delta = metrics_delta.get("vm_pageouts")
    swapouts_delta = metrics_delta.get("vm_swapouts")
    memory_ok = (pageouts_delta in (None, 0)) and (swapouts_delta in (None, 0))
    if not all_parity_ok:
        decision = "reject_parity"
        reason = "strict full-block parity failed for the promoted dense-dequant combination"
        promoted = False
    elif not memory_ok:
        decision = "reject_memory"
        reason = "promoted dense-dequant combination changed pageout/swapout counters during the interleaved profile"
        promoted = False
    else:
        decision = "accept_opt_in_profile"
        reason = (
            "promoted dense-dequant combination is strict-parity and memory-clean in the focused block profile; "
            "interleaved timing is reported as diagnostic because its components were individually promoted"
        )
        promoted = True

    return {
        "name": "promoted_dense_dequant_combo",
        "target_segment": "attention_qkv + attention_out + fc1_swiglu_fc2",
        "target_boundary": "main block qkv_proj, out_proj, and mlp.fc2 dense-dequant projection schedule",
        "selection_rationale": (
            "MacSol remains rejected by the exact-block cost model, so the replacement production path combines the "
            "already promoted strict-parity dense-dequant routes: attention_qkv_tiled_dense_dequant, "
            "ffn_fc2_tiled_dense_dequant, and exactly one Attention.out implementation."
        ),
        "opt_in_only": True,
        "strict_exact_semantics": True,
        "disabled_by_default": True,
        "production_integrated": True,
        "default_behavior_unchanged": True,
        "dense_dequant_profile": profile,
        "attention_out_path": out_path,
        "exactly_one_attention_out_path_enabled": out_path in ("attention_out_dense_dequant", "attention_out_tiled_dense_dequant"),
        "component_switches": {
            "attention_qkv": "use_qkv_tiled_dense_dequant_candidate",
            "ffn_fc2": "use_ffn_fc2_tiled_dense_dequant_candidate",
            "attention_out": out_path,
            "attention_qkv_tile_size": int(args.attention_qkv_tile_size),
            "ffn_fc2_tile_size": int(args.ffn_fc2_tile_size),
            "attention_out_tile_size": int(args.attention_out_tile_size),
        },
        "component_memory_info": {
            "attention_qkv_tiling": qkv_info,
            "ffn_fc2_tiling": fc2_info,
            "attention_out": out_info,
        },
        "first_candidate_call_seconds": first_call_seconds,
        "timing": stats,
        "interleaved_protocol": interleaved,
        "baseline_interleaved_timing": base_timing,
        "pre_candidate_sequential_baseline_timing": baseline_stats,
        "speedup_vs_baseline_median": speedup,
        "block_delta_seconds_candidate_saves": block_delta,
        "block_relative_delta_candidate_saves": block_relative_delta,
        "baseline_check_parity": baseline_check_parity,
        "parity_vs_pre_candidate_baseline_first_call": first_parity,
        "parity_vs_interleaved_baseline": parity,
        "parity_ok": all_parity_ok,
        "memory_gate_ok": memory_ok,
        "noise_decision": interleaved.get("noise_decision"),
        "metrics_before": before,
        "metrics_after_first_call": after_first,
        "metrics_after": after,
        "metrics_delta": metrics_delta,
        "amdahl_end_to_end_contribution": {
            "fixed_baseline_commit": args.fixed_e2e_baseline_commit,
            "fixed_end_to_end_seconds": args.fixed_e2e_baseline_seconds,
            "fixed_end_to_end_peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
            "estimated_block_calls_per_generation": block_calls,
            "per_block_median_saving_seconds": block_delta,
            "idealized_all_blocks_saving_seconds": e2e_saving,
            "idealized_fraction_of_fixed_end_to_end": fixed_e2e_fraction,
            "assumption": "applies one measured block delta to every DiT block and denoiser evaluation; generation A/B remains the decisive end-to-end check",
        },
        "promote": promoted,
        "decision": decision,
        "decision_reason": reason,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/MiniMax-H3-MLX-4bit")
    parser.add_argument("--block-load-mode", default="mlx", choices=("mlx",))
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--spatial-compression", type=int, default=16)
    parser.add_argument("--sigma-grid-points", type=int, default=5)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--video-sigma-shift", type=float, default=12.0)
    parser.add_argument("--audio-sigma-shift", type=float, default=3.0)
    parser.add_argument("--activation-dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--interleaved-warmups", type=int, default=3)
    parser.add_argument("--interleaved-repeats", type=int, default=11)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--capture", choices=("none", "full_block"), default="none")
    parser.add_argument("--trace-source", default=None, help="Summarize an existing .gputrace instead of starting a new capture.")
    parser.add_argument(
        "--candidate",
        choices=ACTIVE_HOTPATH_CANDIDATES,
        default="none",
        help=(
            "active real-block profiler candidate; rejected/tiny-only micro-candidates are "
            "archive-only and intentionally not accepted"
        ),
    )
    parser.add_argument(
        "--dense-dequant-profile",
        choices=DENSE_DEQUANT_GENERATION_PROFILES,
        default=DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--attention-2d-target",
        choices=("auto", "qkv_proj", "out_proj"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ffn-sequence-chunk-size",
        type=int,
        default=512,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ffn-fc1-tile-size",
        type=int,
        default=2048,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ffn-fc2-tile-size",
        type=int,
        default=1024,
        help="Output channels per transient dense-dequant tile for --candidate ffn_fc2_tiled_dense_dequant.",
    )
    parser.add_argument(
        "--ffn-fc2-input-chunk-groups",
        type=int,
        default=56,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ffn-hidden-tile-groups",
        type=int,
        default=56,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--attention-qkv-tile-size",
        type=int,
        default=2048,
        help="Output channels per transient dense-dequant tile for --candidate attention_qkv_tiled_dense_dequant.",
    )
    parser.add_argument(
        "--attention-qkv-headgroup-size",
        type=int,
        default=8,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--attention-qkv-input-chunk-groups",
        type=int,
        default=42,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--attention-sdpa-headgroup-size",
        type=int,
        default=8,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--attention-out-tile-size",
        type=int,
        default=2048,
        help="Output channels per transient dense-dequant tile for --candidate attention_out_tiled_dense_dequant.",
    )
    parser.add_argument("--fixed-e2e-baseline-seconds", type=float, default=122.686)
    parser.add_argument("--fixed-e2e-baseline-memory-gb", type=float, default=14.201)
    parser.add_argument("--fixed-e2e-baseline-commit", default="deda96c")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--tiny", action="store_true", help="Use a tiny random block for fast harness verification; not real 4-bit evidence.")
    parser.add_argument("--parity-atol", type=float, default=1e-4)
    parser.add_argument("--parity-rel-l2", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    command = [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]

    started = time.perf_counter()
    safety_preflight = _safety_preflight()
    env_before = _metrics()
    _log(
        events,
        "starting DiT block hotpath profiler",
        out=str(out_path),
        tiny=args.tiny,
        disk_free_gib=safety_preflight["disk_free_gib"],
        other_high_memory_processes=safety_preflight["other_high_memory_process_count"],
    )
    _reset_mlx_peak()
    block, cfg, block_meta = _load_block(args, events)
    x, modulation, adaln_indices, rotary, sequence_meta = _build_inputs(cfg, args, block, events)

    baseline_before = _metrics()
    baseline_stats, baseline_out = _time_callable(
        lambda: block(x, modulation, adaln_indices, rotary),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    baseline_after = _metrics()
    _log(events, "timed uninstrumented full block", median_seconds=baseline_stats.get("median_seconds"))

    segment_stats, segmented_out = _time_segmented(
        block,
        x,
        modulation,
        adaln_indices,
        rotary,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    segmented_parity = _diff_stats(baseline_out, segmented_out)
    segment_rank = _rank_segments(segment_stats)
    _log(events, "timed instrumented block segments", top_segment=segment_rank[0]["segment"] if segment_rank else None)

    trace: dict[str, Any]
    if args.trace_source:
        trace_path = Path(args.trace_source)
        trace = {
            "ok": trace_path.exists(),
            "path": str(trace_path),
            "scope": "existing Metal capture summarized for kernel/materialization boundaries",
            **_scan_capture(trace_path, cfg=cfg, sequence_meta=sequence_meta, segment_stats=segment_stats),
        }
        _log(events, "summarized existing Metal trace", ok=trace.get("ok"), path=trace.get("path"))
    elif args.capture == "full_block":
        capture_path = out_path.with_suffix(".gputrace")
        trace = _capture_full_block(
            capture_path,
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            cfg=cfg,
            sequence_meta=sequence_meta,
            segment_stats=segment_stats,
        )
        _log(events, "captured full block Metal trace", ok=trace.get("ok"), path=trace.get("path"))
    else:
        trace = {
            "ok": False,
            "mode": "not_requested",
            "note": "No Metal capture requested for this run; segmented timings contain explicit mx.eval boundaries and are not kernel-boundary proof.",
        }

    candidate: dict[str, Any]
    if args.candidate == "compile_block":
        _log(events, "running opt-in compile candidate")
        candidate = _candidate_compile(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in compile candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_mx_split_swiglu":
        _log(events, "running opt-in FFN mx.split SwiGLU candidate")
        candidate = _candidate_ffn_mx_split(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN mx.split SwiGLU candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate in ("ffn_metal_swiglu", "ffn_fused_swiglu_metal"):
        _log(events, "running opt-in FFN Metal SwiGLU candidate", candidate=args.candidate)
        candidate = _candidate_ffn_fused_swiglu_metal(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
            candidate_name=args.candidate,
        )
        _log(
            events,
            "completed opt-in FFN Metal SwiGLU candidate",
            candidate=args.candidate,
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_projection_2d_qmm":
        _log(events, "running opt-in FFN fc1/fc2 2D projection QMM candidate")
        candidate = _candidate_ffn_projection_2d_qmm(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN fc1/fc2 2D projection QMM candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_fc1_rank2_qmm":
        _log(events, "running opt-in FFN fc1-only rank-2 QMM candidate")
        candidate = _candidate_ffn_fc1_rank2_qmm(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN fc1-only rank-2 QMM candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_fc1_split_gate_value_quantized_qmm":
        _log(events, "running opt-in FFN fc1 split gate/value quantized QMM candidate")
        candidate = _candidate_ffn_fc1_split_gate_value_quantized_qmm(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN fc1 split gate/value quantized QMM candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_fc2_rank2_qmm":
        _log(events, "running opt-in FFN fc2-only rank-2 QMM candidate")
        candidate = _candidate_ffn_fc2_rank2_qmm(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN fc2-only rank-2 QMM candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_fc2_input_chunked_qmm":
        _log(
            events,
            "running opt-in FFN fc2 input-group chunked QMM candidate",
            chunk_groups=args.ffn_fc2_input_chunk_groups,
        )
        candidate = _candidate_ffn_fc2_input_chunked_qmm(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN fc2 input-group chunked QMM candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_fc1_dense_dequant":
        _log(events, "running opt-in FFN fc1 resident dense-dequant candidate")
        candidate = _candidate_ffn_fc1_dense_dequant(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN fc1 resident dense-dequant candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_fc1_tiled_dense_dequant":
        _log(events, "running opt-in FFN fc1 transient tiled dense-dequant candidate", tile_size=args.ffn_fc1_tile_size)
        candidate = _candidate_ffn_fc1_tiled_dense_dequant(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN fc1 transient tiled dense-dequant candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_fc2_dense_dequant":
        _log(events, "running opt-in FFN fc2 resident dense-dequant candidate")
        candidate = _candidate_ffn_fc2_dense_dequant(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN fc2 resident dense-dequant candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_fc2_tiled_dense_dequant":
        _log(events, "running opt-in FFN fc2 transient tiled dense-dequant candidate", tile_size=args.ffn_fc2_tile_size)
        candidate = _candidate_ffn_fc2_tiled_dense_dequant(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN fc2 transient tiled dense-dequant candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_hidden_tile_stream":
        _log(events, "running opt-in FFN hidden-channel streamed candidate", tile_groups=args.ffn_hidden_tile_groups)
        candidate = _candidate_ffn_hidden_tile_stream(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN hidden-channel streamed candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_pre_fc1_contiguous":
        _log(events, "running opt-in FFN pre-fc1 contiguous candidate")
        candidate = _candidate_ffn_pre_fc1_contiguous(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN pre-fc1 contiguous candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_pre_fc2_contiguous":
        _log(events, "running opt-in FFN pre-fc2 contiguous candidate")
        candidate = _candidate_ffn_pre_fc2_contiguous(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN pre-fc2 contiguous candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_sequence_chunked":
        _log(events, "running opt-in FFN sequence-chunked candidate", chunk_size=args.ffn_sequence_chunk_size)
        candidate = _candidate_ffn_sequence_chunked(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN sequence-chunked candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "ffn_subgraph_compile":
        _log(events, "running opt-in FFN subgraph compile candidate")
        candidate = _candidate_ffn_subgraph_compile(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in FFN subgraph compile candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "adaln_packed_gather":
        _log(events, "running opt-in AdaLN packed gather candidate")
        candidate = _candidate_adaln_packed_gather(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in AdaLN packed gather candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "indexed_adaln_affine_metal":
        _log(events, "running opt-in indexed AdaLN affine Metal candidate")
        candidate = _candidate_indexed_adaln_affine_metal(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in indexed AdaLN affine Metal candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "indexed_gated_residual_metal":
        _log(events, "running opt-in indexed gated-residual Metal candidate")
        candidate = _candidate_indexed_gated_residual_metal(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in indexed gated-residual Metal candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_projection_2d_qmm":
        _log(events, "running opt-in Attention 2D projection QMM candidate", target=args.attention_2d_target)
        candidate = _candidate_attention_projection_2d_qmm(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention 2D projection QMM candidate",
            decision=candidate.get("decision"),
            target=candidate.get("target_projection"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_out_dense_dequant":
        _log(events, "running opt-in Attention out_proj resident dense-dequant candidate")
        candidate = _candidate_attention_out_dense_dequant(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention out_proj resident dense-dequant candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_out_tiled_dense_dequant":
        _log(
            events,
            "running opt-in Attention out_proj transient tiled dense-dequant candidate",
            tile_size=args.attention_out_tile_size,
        )
        candidate = _candidate_attention_out_tiled_dense_dequant(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention out_proj transient tiled dense-dequant candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_qkv_tiled_dense_dequant":
        _log(
            events,
            "running opt-in Attention qkv_proj transient tiled dense-dequant candidate",
            tile_size=args.attention_qkv_tile_size,
        )
        candidate = _candidate_attention_qkv_tiled_dense_dequant(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention qkv_proj transient tiled dense-dequant candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "promoted_dense_dequant_combo":
        _log(
            events,
            "running promoted dense-dequant combo profile",
            dense_dequant_profile=args.dense_dequant_profile,
            attention_qkv_tile_size=args.attention_qkv_tile_size,
            ffn_fc2_tile_size=args.ffn_fc2_tile_size,
            attention_out_tile_size=args.attention_out_tile_size,
        )
        candidate = _candidate_promoted_dense_dequant_combo(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed promoted dense-dequant combo profile",
            decision=candidate.get("decision"),
            profile=candidate.get("dense_dequant_profile"),
            attention_out_path=candidate.get("attention_out_path"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_qkv_headgroup_row_sliced_qmm":
        _log(
            events,
            "running opt-in Attention qkv_proj whole-head row-sliced QMM candidate",
            heads_per_slice=args.attention_qkv_headgroup_size,
        )
        candidate = _candidate_attention_qkv_headgroup_row_sliced_qmm(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention qkv_proj whole-head row-sliced QMM candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_qkv_input_chunked_qmm":
        _log(
            events,
            "running opt-in Attention qkv_proj input-group chunked QMM candidate",
            chunk_groups=args.attention_qkv_input_chunk_groups,
        )
        candidate = _candidate_attention_qkv_input_chunked_qmm(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention qkv_proj input-group chunked QMM candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_pre_qkv_contiguous":
        _log(events, "running opt-in Attention pre-QKV input contiguous candidate")
        candidate = _candidate_attention_pre_qkv_contiguous(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention pre-QKV input contiguous candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_qkv_pretranspose_layout":
        _log(events, "running opt-in Attention QKV pretranspose layout candidate")
        candidate = _candidate_attention_qkv_pretranspose_layout(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention QKV pretranspose layout candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_pre_sdpa_contiguous":
        _log(events, "running opt-in Attention pre-SDPA q/k/v contiguous candidate")
        candidate = _candidate_attention_pre_sdpa_contiguous(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention pre-SDPA q/k/v contiguous candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_sdpa_head_batch_rank3":
        _log(events, "running opt-in Attention SDPA head-batch rank3 candidate")
        candidate = _candidate_attention_sdpa_head_batch_rank3(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention SDPA head-batch rank3 candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_sdpa_headgroup_split_rank4":
        _log(
            events,
            "running opt-in Attention rank-4 SDPA head-group split candidate",
            heads_per_group=args.attention_sdpa_headgroup_size,
        )
        candidate = _candidate_attention_sdpa_headgroup_split_rank4(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention rank-4 SDPA head-group split candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_pre_out_proj_contiguous":
        _log(events, "running opt-in Attention pre-out_proj output contiguous candidate")
        candidate = _candidate_attention_pre_out_proj_contiguous(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention pre-out_proj output contiguous candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_qkv_rmsnorm_sdpa_metal":
        _log(events, "running opt-in Attention q/k RMSNorm SDPA-layout Metal candidate")
        candidate = _candidate_attention_qkv_rmsnorm_sdpa_metal(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention q/k RMSNorm SDPA-layout Metal candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_qkv_rmsnorm_rotary_sdpa_metal":
        _log(events, "running opt-in Attention q/k RMSNorm+RoPE SDPA-layout Metal candidate")
        candidate = _candidate_attention_qkv_rmsnorm_rotary_sdpa_metal(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention q/k RMSNorm+RoPE SDPA-layout Metal candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_sdpa_out_layout_metal":
        _log(events, "running opt-in Attention SDPA-output layout Metal candidate")
        candidate = _candidate_attention_sdpa_out_layout_metal(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention SDPA-output layout Metal candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    elif args.candidate == "attention_rotary_qk_metal":
        _log(events, "running opt-in Attention q/k RoPE Metal candidate")
        candidate = _candidate_attention_rotary_qk_metal(
            block,
            x,
            modulation,
            adaln_indices,
            rotary,
            baseline_out,
            baseline_stats,
            segment_stats,
            cfg,
            sequence_meta,
            args,
        )
        _log(
            events,
            "completed opt-in Attention q/k RoPE Metal candidate",
            decision=candidate.get("decision"),
            speedup=candidate.get("speedup_vs_baseline_median"),
        )
    else:
        candidate = {
            "name": None,
            "decision": "not_attempted",
            "promote": False,
            "reason": "No candidate requested in this profiling run.",
        }

    after = _metrics()
    artifact = {
        "schema_version": 2,
        "kind": "dit_block_hotpath_profile",
        "created_at": _now(),
        "ok": True,
        "command": command,
        "cwd": str(Path.cwd()),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "mtl_capture_enabled": os.environ.get("MTL_CAPTURE_ENABLED"),
        },
        "safety_preflight": safety_preflight,
        "block": block_meta,
        "sequence": sequence_meta,
        "timing_protocol": {
            "sequential_warmups": args.warmups,
            "sequential_repeats": args.repeats,
            "interleaved_warmup_pairs": args.interleaved_warmups,
            "interleaved_measured_pairs": args.interleaved_repeats,
            "bootstrap_resamples": args.bootstrap_resamples,
            "baseline_full_block": "one TransformerBlock call, one mx.eval at output; used for standalone block timing and parity reference",
            "segmented": "same math split into named subsegments with mx.eval after each segment for Amdahl attribution",
            "candidate_comparison": "paired interleaved alternating baseline/candidate order with bootstrap CI over paired candidate-minus-baseline deltas",
            "decision_rule": "no fixed percentage threshold; require strict parity, inspect whether timing delta separates from noise, then weigh Amdahl contribution, compile cost, memory, resolution scaling, and maintainability",
            "fixed_end_to_end_baseline": {
                "commit": args.fixed_e2e_baseline_commit,
                "seconds": args.fixed_e2e_baseline_seconds,
                "peak_memory_gb": args.fixed_e2e_baseline_memory_gb,
                "note": "directive-fixed full-generation baseline; not rerun by this block profiler",
            },
        },
        "baseline_full_block_timing": baseline_stats,
        "interleaved_baseline_full_block_timing": candidate.get("baseline_interleaved_timing"),
        "baseline_metrics_before": baseline_before,
        "baseline_metrics_after": baseline_after,
        "baseline_metrics_delta": _delta(baseline_before, baseline_after),
        "segment_timing": segment_stats,
        "segment_timing_ranked": segment_rank,
        "segmented_vs_full_parity": segmented_parity,
        "trace_kernel_boundary_summary": trace,
        "candidate": candidate,
        "memory_observations": {
            "process_before": env_before,
            "process_after": after,
            "process_delta": _delta(env_before, after),
            "sustained_swap_or_pageout_observed_in_process": bool(
                (_delta(env_before, after).get("vm_pageouts") or 0) > 0
                or (_delta(env_before, after).get("vm_swapouts") or 0) > 0
            ),
        },
        "decision": {
            "promote_candidate": bool(candidate.get("promote")),
            "status": candidate.get("decision", "not_attempted"),
            "reason": candidate.get("decision_reason") or candidate.get("reason"),
            "strongest_measured_segment": segment_rank[0] if segment_rank else None,
        },
        "raw_log_events": events,
        "elapsed_seconds": time.perf_counter() - started,
    }
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    log_path = out_path.with_suffix(".log")
    log_path.write_text("\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n")
    _log(events, "wrote profile artifact", out=str(out_path), log=str(log_path))
    # Rewrite once more so the final write event is present in the machine-readable record.
    artifact["raw_log_events"] = events
    artifact["log_path"] = str(log_path)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
