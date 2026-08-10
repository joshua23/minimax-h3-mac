#!/usr/bin/env python3
"""Probe the MacSol v5 64x64 exact-tile MLX/Metal microkernel.

This is a bounded decision gate, not generation integration.  It tests whether a threadgroup-
cooperative 64-query x 64-key BF16 exact tile is worth expanding into a full source-faithful
Sol-Attn sparse online-softmax forward.  The v5 tile kernel stores only transient threadgroup
scores/softmax state and returns per-tile outputs for parity/timing; it does not materialize dense
SxS scores, dense masks, persistent route tensors, or change dense defaults.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from minimax_h3_mlx.macsol_metal import (  # noqa: E402
    MACSOL_METAL_V2_STATUS,
    MACSOL_METAL_V3_STATUS,
    MACSOL_METAL_V4_STATUS,
    MACSOL_METAL_V5_DECISION_NOTE,
    MACSOL_METAL_V5_DESCRIPTION,
    MACSOL_METAL_V5_REJECTION_REASON,
    MACSOL_METAL_V5_STATUS,
    has_macsol_metal_v2,
    has_macsol_metal_v3,
    has_macsol_metal_v4,
    has_macsol_metal_v5,
    macsol_attention_metal_subset,
    macsol_attention_metal_subset_v2,
    macsol_attention_metal_subset_v3,
    macsol_attention_metal_subset_v4,
    macsol_exact_tile_metal_v5,
    macsol_prepare_metal,
)
from minimax_h3_mlx.macsol_reference import (  # noqa: E402
    H3PackedLengths,
    build_macsol_routing,
    dense_attention_reference,
    h3_macsol_config,
)

import scripts.probe_macsol_metal as v4probe  # noqa: E402


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _cmd(argv: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        out = subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=timeout)
        return {"ok": True, "stdout": out.strip()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return None


def _toolchain_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "mlx": _version("mlx"),
        "numpy": _version("numpy"),
        "pytest": _version("pytest"),
        "mlx_fast_metal_kernel": hasattr(getattr(mx, "fast", None), "metal_kernel"),
        "default_device": str(mx.default_device()),
        "xcode_select": _cmd(["xcode-select", "-p"]),
        "xcrun_metal": _cmd(["xcrun", "--find", "metal"]),
        "xcrun_metallib": _cmd(["xcrun", "--find", "metallib"]),
    }


def _error_metrics(got: mx.array, ref: mx.array) -> dict[str, float]:
    return v4probe._error_metrics(got, ref)


def _local_exact_tiles(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    q_block_start: int,
    kv_block_start: int,
    kv_block_count: int,
    head_start: int,
    head_count: int,
    block_size: int,
    scale: float,
) -> mx.array:
    """Dense local exact attention for independent 64x64 tiles, matching v5 output shape."""

    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    qlo = int(q_block_start) * int(block_size)
    qhi = qlo + int(block_size)
    batch_outputs: list[mx.array] = []
    for b in range(int(q.shape[0])):
        head_outputs: list[mx.array] = []
        for h in range(int(head_start), int(head_start) + int(head_count)):
            q_block = q32[b, h, qlo:qhi]
            tiles: list[mx.array] = []
            for kv in range(int(kv_block_start), int(kv_block_start) + int(kv_block_count)):
                klo = kv * int(block_size)
                khi = klo + int(block_size)
                scores = mx.matmul(q_block, k32[b, h, klo:khi].T) * scale
                tiles.append(mx.matmul(mx.softmax(scores, axis=-1), v32[b, h, klo:khi]))
            head_outputs.append(mx.stack(tiles, axis=0))
        batch_outputs.append(mx.stack(head_outputs, axis=0))
    return mx.stack(batch_outputs, axis=0)


def _synthetic_qkv(*, sequence: int, heads: int, head_dim: int, seed: int, dtype: str) -> tuple[mx.array, mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    shape = (1, int(heads), int(sequence), int(head_dim))
    q = mx.array(rng.standard_normal(shape, dtype=np.float32))
    k = mx.array(rng.standard_normal(shape, dtype=np.float32))
    v = mx.array(rng.standard_normal(shape, dtype=np.float32))
    if dtype == "bf16":
        q = q.astype(mx.bfloat16)
        k = k.astype(mx.bfloat16)
        v = v.astype(mx.bfloat16)
    return q, k, v


def _architecture_comparison() -> list[dict[str, Any]]:
    return [
        {
            "name": "selected_score_tile_threadgroup_microkernel",
            "decision": "selected_for_bounded_v5_probe",
            "dataflow": "one 256-thread group computes a full 64x64 QK score tile into transient threadgroup memory, row softmax state, then P*V per output dimension",
            "why": "materially different from v1-v4 row-owned/row-wave kernels; directly tests the exact-token tile primitive needed by source-faithful Sol-Attn",
        },
        {
            "name": "row_simd_with_kv_threadgroup_cache",
            "decision": "rejected_before_code",
            "dataflow": "cache K/V tile but keep one SIMD-group per row and sweep rows in waves",
            "why": "too close to rejected v2/v4; previous row-wave design under-occupied Apple GPU and still serialized row work",
        },
        {
            "name": "global_two_pass_score_or_route_materialization",
            "decision": "rejected_before_code",
            "dataflow": "emit score/route buffers, then run a second reduction/matmul",
            "why": "violates normal MacSol boundary by persisting dense-ish score/route tensors and would be memory-hostile at S=19540",
        },
        {
            "name": "centroid_first_approximate_branch",
            "decision": "rejected_before_code",
            "dataflow": "optimize centroid/Vsum approximate blocks first and leave exact token blocks as scalar fallback",
            "why": "tau-0 real H3 route still has high exact-block density; exact-token work is the unresolved compute hotpath",
        },
    ]


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    q, k, v = _synthetic_qkv(sequence=192, heads=1, head_dim=128, seed=args.seed, dtype=args.synthetic_dtype)
    scale = float(q.shape[-1] ** -0.5)
    got = macsol_exact_tile_metal_v5(
        q,
        k,
        v,
        q_block_start=0,
        kv_block_start=1,
        kv_block_count=2,
        head_count=1,
        block_size=args.block_size,
        scale=scale,
    ).output
    ref = _local_exact_tiles(
        q,
        k,
        v,
        q_block_start=0,
        kv_block_start=1,
        kv_block_count=2,
        head_start=0,
        head_count=1,
        block_size=args.block_size,
        scale=scale,
    )
    q64, k64, v64 = _synthetic_qkv(sequence=64, heads=1, head_dim=128, seed=args.seed + 1, dtype=args.synthetic_dtype)
    dense64 = dense_attention_reference(q64, k64, v64)
    got64 = macsol_exact_tile_metal_v5(q64, k64, v64, head_count=1, block_size=args.block_size).output[:, :, 0]
    mx.eval(got, ref, got64, dense64)
    tile_error = _error_metrics(got, ref)
    all_exact_error = _error_metrics(got64, dense64)
    timings = v4probe._time_interleaved(
        args.synthetic_warm_runs,
        [
            (
                "dense_local_exact_tile_reference",
                lambda: _local_exact_tiles(
                    q,
                    k,
                    v,
                    q_block_start=0,
                    kv_block_start=1,
                    kv_block_count=1,
                    head_start=0,
                    head_count=1,
                    block_size=args.block_size,
                    scale=scale,
                ),
            ),
            (
                "metal_v5_single_exact_tile",
                lambda: macsol_exact_tile_metal_v5(
                    q,
                    k,
                    v,
                    q_block_start=0,
                    kv_block_start=1,
                    kv_block_count=1,
                    head_count=1,
                    block_size=args.block_size,
                    scale=scale,
                ).output,
            ),
            (
                "metal_v5_two_contiguous_exact_tiles",
                lambda: macsol_exact_tile_metal_v5(
                    q,
                    k,
                    v,
                    q_block_start=0,
                    kv_block_start=1,
                    kv_block_count=2,
                    head_count=1,
                    block_size=args.block_size,
                    scale=scale,
                ).output,
            ),
        ],
    )
    checks = {
        "metal_v5_api_available": has_macsol_metal_v5(),
        "two_tile_bf16_d128_matches_local_dense": tile_error["max_abs"] <= args.synthetic_tolerance,
        "single_all_exact_64_sequence_matches_dense": all_exact_error["max_abs"] <= args.synthetic_tolerance,
    }
    return {
        "ok": all(checks.values()),
        "parameters": {"sequence": 192, "heads": 1, "head_dim": 128, "dtype": args.synthetic_dtype, "block_size": args.block_size, "scale": scale},
        "checks": checks,
        "errors": {
            "metal_v5_two_tiles_vs_local_dense": tile_error,
            "metal_v5_all_exact_64_sequence_vs_dense": all_exact_error,
        },
        "timings": timings,
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    process_scan = v4probe._process_scan()
    if process_scan.get("other_high_memory_processes"):
        return {"ok": False, "skipped": True, "blocker": "other_high_memory_process_present", "process_scan": process_scan}

    realmod = v4probe._load_real_harness()
    real_args = realmod.parse_args(
        [
            "--model-dir",
            args.model_dir,
            "--block-index",
            str(args.block_index),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--duration",
            str(args.duration),
            "--text-tokens",
            str(args.text_tokens),
            "--sigma-grid-points",
            str(args.sigma_grid_points),
            "--step-index",
            str(args.step_index),
            "--activation-dtype",
            args.activation_dtype,
            "--block-size",
            str(args.block_size),
            "--taus",
            str(args.tau),
        ]
    )
    events: list[dict[str, Any]] = []
    load_before = v4probe._metrics()
    block, cfg_model, block_meta = realmod._load_block(real_args, events)
    load_after = v4probe._metrics()
    inputs_before = v4probe._metrics()
    x, modulation, adaln_indices, rotary, position_ids, lengths, sequence_meta = realmod._build_inputs(cfg_model, real_args, block, events)
    inputs_after = v4probe._metrics()
    capture_before = v4probe._metrics()
    q, k, v, capture_meta = realmod._capture_attention_qkv(block, x, modulation, adaln_indices, rotary)
    capture_after = v4probe._metrics()

    sequence_length = int(q.shape[2])
    head_dim = int(q.shape[-1])
    scale = float(head_dim ** -0.5)
    block_count = int(math.ceil(sequence_length / args.block_size))
    full_block_count = int(sequence_length // args.block_size)
    q_block_start = v4probe._auto_q_block(lengths, sequence_length, args.block_size, args.real_q_block_start)
    if (q_block_start + 1) * args.block_size > sequence_length:
        raise ValueError(f"v5 exact-tile gate requires a full query block, got q_block_start={q_block_start}")
    kv_block_start = int(args.real_kv_block_start if args.real_kv_block_start >= 0 else q_block_start)
    if (kv_block_start + 1) * args.block_size > sequence_length:
        raise ValueError(f"v5 exact-tile gate requires a full KV block, got kv_block_start={kv_block_start}")
    head_start = int(args.real_head_start)
    head_count = int(args.real_head_count)
    if head_start + head_count > int(q.shape[1]):
        raise ValueError(f"requested heads [{head_start},{head_start + head_count}) exceed {q.shape[1]}")

    cfg = h3_macsol_config(lengths, tau=args.tau, block_size=args.block_size)
    routing = build_macsol_routing(q, k, cfg, scale=scale)
    routing_stats = routing.summary()
    routing_stats.update(realmod._effective_exact_block_stats(routing, lengths.prefix_tokens))
    exact_indices = np.nonzero(routing.exact_block_mask[0, head_start, q_block_start])[0].astype(int).tolist()
    exact_count = len(exact_indices)

    prep = macsol_prepare_metal(q, k, v, block_size=args.block_size, tau=args.tau, scale=scale)
    metal_v2 = macsol_attention_metal_subset_v2(q, k, v, cfg, q_block_start=q_block_start, q_block_count=1, head_start=head_start, head_count=head_count, scale=scale, preparation=prep).output
    metal_v3 = macsol_attention_metal_subset_v3(q, k, v, cfg, q_block_start=q_block_start, q_block_count=1, head_start=head_start, head_count=head_count, scale=scale, preparation=prep).output
    metal_v4 = macsol_attention_metal_subset_v4(q, k, v, cfg, q_block_start=q_block_start, q_block_count=1, head_start=head_start, head_count=head_count, scale=scale, preparation=prep).output
    dense = v4probe._dense_subset(q, k, v, head_start=head_start, head_count=head_count, q_block_start=q_block_start, q_block_count=1, block_size=args.block_size, scale=scale)
    reference = v4probe._reference_subset(q, k, v, routing, cfg, head_start=head_start, head_count=head_count, q_block_start=q_block_start, q_block_count=1, scale=scale)
    v5_tile = macsol_exact_tile_metal_v5(q, k, v, q_block_start=q_block_start, kv_block_start=kv_block_start, kv_block_count=1, head_start=head_start, head_count=head_count, block_size=args.block_size, scale=scale).output
    tile_ref = _local_exact_tiles(q, k, v, q_block_start=q_block_start, kv_block_start=kv_block_start, kv_block_count=1, head_start=head_start, head_count=head_count, block_size=args.block_size, scale=scale)
    mx.eval(metal_v2, metal_v3, metal_v4, dense, reference, v5_tile, tile_ref)

    errors = {
        "metal_v2_cooperative_vs_macsol_reference_subset": _error_metrics(metal_v2, reference),
        "metal_v3_qblock_tiled_vs_macsol_reference_subset": _error_metrics(metal_v3, reference),
        "metal_v4_qblock_simd_vs_macsol_reference_subset": _error_metrics(metal_v4, reference),
        "macsol_reference_vs_dense_subset": _error_metrics(reference, dense),
        "metal_v5_exact_tile_vs_local_dense_tile": _error_metrics(v5_tile, tile_ref),
    }

    timings = v4probe._time_interleaved(
        args.real_warm_runs,
        [
            (
                "dense_sampled_row_block",
                lambda: v4probe._dense_subset(q, k, v, head_start=head_start, head_count=head_count, q_block_start=q_block_start, q_block_count=1, block_size=args.block_size, scale=scale),
            ),
            (
                "reference_sampled_row_block",
                lambda: v4probe._reference_subset(q, k, v, routing, cfg, head_start=head_start, head_count=head_count, q_block_start=q_block_start, q_block_count=1, scale=scale),
            ),
            (
                "metal_v1_scalar_sampled_row_block",
                lambda: macsol_attention_metal_subset(q, k, v, cfg, q_block_start=q_block_start, q_block_count=1, head_start=head_start, head_count=head_count, scale=scale).output,
            ),
            (
                "metal_v2_cooperative_sampled_row_block",
                lambda: macsol_attention_metal_subset_v2(q, k, v, cfg, q_block_start=q_block_start, q_block_count=1, head_start=head_start, head_count=head_count, scale=scale).output,
            ),
            (
                "metal_v3_qblock_tiled_sampled_row_block",
                lambda: macsol_attention_metal_subset_v3(q, k, v, cfg, q_block_start=q_block_start, q_block_count=1, head_start=head_start, head_count=head_count, scale=scale).output,
            ),
            (
                "metal_v4_qblock_simd_sampled_row_block",
                lambda: macsol_attention_metal_subset_v4(q, k, v, cfg, q_block_start=q_block_start, q_block_count=1, head_start=head_start, head_count=head_count, scale=scale).output,
            ),
            (
                "metal_v5_single_exact_64x64_tile",
                lambda: macsol_exact_tile_metal_v5(q, k, v, q_block_start=q_block_start, kv_block_start=kv_block_start, kv_block_count=1, head_start=head_start, head_count=head_count, block_size=args.block_size, scale=scale).output,
            ),
            (
                "metal_v5_all_contiguous_full_exact_tiles",
                lambda: macsol_exact_tile_metal_v5(q, k, v, q_block_start=q_block_start, kv_block_start=0, kv_block_count=full_block_count, head_start=head_start, head_count=head_count, block_size=args.block_size, scale=scale).output,
            ),
        ],
    )

    dense_median = timings["dense_sampled_row_block"]["median_elapsed_seconds"]
    reference_median = timings["reference_sampled_row_block"]["median_elapsed_seconds"]
    v5_single_median = timings["metal_v5_single_exact_64x64_tile"]["median_elapsed_seconds"]
    v5_all_median = timings["metal_v5_all_contiguous_full_exact_tiles"]["median_elapsed_seconds"]
    amortized_tile = (v5_all_median / full_block_count) if v5_all_median and full_block_count else None
    projected_selected = (amortized_tile * exact_count) if amortized_tile is not None else None
    projected_unamortized = (v5_single_median * exact_count) if v5_single_median is not None else None
    parity_ok = errors["metal_v5_exact_tile_vs_local_dense_tile"]["max_abs"] <= args.real_tile_tolerance

    return {
        "ok": bool(parity_ok),
        "parameters": {
            "model_dir": args.model_dir,
            "block_index": args.block_index,
            "height": args.height,
            "width": args.width,
            "duration": args.duration,
            "activation_dtype": args.activation_dtype,
            "tau": args.tau,
            "block_size": args.block_size,
            "head_start": head_start,
            "head_count": head_count,
            "q_block_start": q_block_start,
            "kv_block_start_for_single_tile": kv_block_start,
            "q_token_range": [q_block_start * args.block_size, (q_block_start + 1) * args.block_size],
            "kv_token_range_for_single_tile": [kv_block_start * args.block_size, (kv_block_start + 1) * args.block_size],
            "sequence_length": sequence_length,
            "head_dim": head_dim,
            "scale": scale,
            "block_count_including_tail": block_count,
            "full_64_token_block_count_timed_by_v5": full_block_count,
            "full_dense_not_materialized": True,
        },
        "block_load": {"meta": block_meta, "metrics_before": load_before, "metrics_after": load_after, "metrics_delta": v4probe._delta(load_before, load_after)},
        "input_build": {"metrics_before": inputs_before, "metrics_after": inputs_after, "metrics_delta": v4probe._delta(inputs_before, inputs_after)},
        "qkv_capture": {**capture_meta, "metrics_before": capture_before, "metrics_after": capture_after, "metrics_delta": v4probe._delta(capture_before, capture_after)},
        "sequence": sequence_meta,
        "routing_stats": routing_stats,
        "exact_blocks_for_sampled_q_block_head0": {"count": exact_count, "first_32_indices": exact_indices[:32]},
        "errors": errors,
        "timings": timings,
        "speedups_or_lower_bounds": {
            "metal_v5_single_tile_speedup_vs_dense_row_block": dense_median / v5_single_median if dense_median and v5_single_median else None,
            "metal_v5_all_contiguous_tiles_speedup_vs_dense_row_block": dense_median / v5_all_median if dense_median and v5_all_median else None,
            "metal_v5_single_tile_speedup_vs_mlx_reference_row_block": reference_median / v5_single_median if reference_median and v5_single_median else None,
            "amortized_v5_tile_seconds_from_all_contiguous_tiles": amortized_tile,
            "projected_selected_exact_tiles_seconds_amortized_lower_bound": projected_selected,
            "projected_selected_exact_tiles_seconds_unamortized": projected_unamortized,
            "dense_sampled_row_block_seconds": dense_median,
            "reference_sampled_row_block_seconds": reference_median,
        },
    }


def _decision(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if record.get("failure"):
        return {
            "decision": "blocker",
            "execution_status": "failed",
            "failure_class": "runtime_or_environment_blocker",
            "idea_status": "undetermined",
            "reason": record["failure"].get("message", "unhandled failure"),
        }
    synthetic = record.get("synthetic") or {}
    real = record.get("real") or {}
    if not synthetic.get("ok"):
        return {
            "decision": "reject",
            "execution_status": "completed",
            "failure_class": None,
            "idea_status": "rejected_correctness",
            "reason": "synthetic v5 exact-tile parity failed",
            "checks": synthetic.get("checks"),
        }
    if args.skip_real:
        return {
            "decision": "synthetic_only_continue_needed",
            "execution_status": "completed_partial",
            "failure_class": None,
            "idea_status": "undetermined_without_real_qkv",
            "reason": "real-QKV gate skipped",
        }
    if real.get("skipped"):
        return {
            "decision": "blocker",
            "execution_status": "blocked",
            "failure_class": real.get("blocker"),
            "idea_status": "undetermined",
            "reason": real.get("blocker", "real probe skipped"),
        }
    if not real.get("ok"):
        return {
            "decision": "reject",
            "execution_status": "completed",
            "failure_class": None,
            "idea_status": "rejected_correctness",
            "reason": "real-QKV v5 exact tile did not match local dense tile",
            "errors": real.get("errors", {}).get("metal_v5_exact_tile_vs_local_dense_tile"),
        }
    timing = real.get("speedups_or_lower_bounds") or {}
    dense = timing.get("dense_sampled_row_block_seconds")
    single = (real.get("timings") or {}).get("metal_v5_single_exact_64x64_tile", {}).get("median_elapsed_seconds")
    projected = timing.get("projected_selected_exact_tiles_seconds_amortized_lower_bound")
    if dense is not None and single is not None and single >= dense:
        return {
            "decision": "reject_for_now_timing",
            "execution_status": "completed",
            "failure_class": None,
            "idea_status": "rejected_timing",
            "reason": "single v5 64x64 exact tile is slower than sampled dense row-block attention",
            "dense_seconds": dense,
            "single_tile_seconds": single,
        }
    if dense is not None and projected is not None and projected >= dense:
        return {
            "decision": "reject_for_now_timing",
            "execution_status": "completed",
            "failure_class": None,
            "idea_status": "rejected_timing_lower_bound",
            "reason": "amortized v5 exact-tile lower bound for route-selected exact blocks already exceeds sampled dense before adding routing, approximate blocks, and cross-tile online softmax",
            "dense_seconds": dense,
            "projected_selected_exact_tiles_seconds_amortized_lower_bound": projected,
            "timing": timing,
        }
    return {
        "decision": "microkernel_signal_viable_needs_full_v5_integration",
        "execution_status": "completed_microbench",
        "failure_class": None,
        "idea_status": "promising_but_not_full_engine",
        "reason": "v5 exact-tile parity passed and amortized exact-tile lower bound is below sampled dense; full sparse online-softmax integration would be required before any keep decision",
        "timing": timing,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--synthetic-dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--synthetic-warm-runs", type=int, default=3)
    parser.add_argument("--synthetic-tolerance", type=float, default=5e-4)
    parser.add_argument("--real-tile-tolerance", type=float, default=2e-3)
    parser.add_argument("--model-dir", default="models/MiniMax-H3-MLX-4bit")
    parser.add_argument("--block-index", type=int, default=10)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--sigma-grid-points", type=int, default=5)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--activation-dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--real-head-start", type=int, default=0)
    parser.add_argument("--real-head-count", type=int, default=1)
    parser.add_argument("--real-q-block-start", type=int, default=-1, help="negative means middle target block")
    parser.add_argument("--real-kv-block-start", type=int, default=-1, help="negative means the same block as q_block_start")
    parser.add_argument("--real-warm-runs", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(args.out_dir) if args.out_dir else Path("experiments") / f"macsol_metal_v5_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_before = v4probe._metrics()
    record: dict[str, Any] = {
        "ok": False,
        "recorded_at": _now(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        "preflight": {
            "toolchain": _toolchain_info(),
            "metrics_before": metrics_before,
            "process_scan": v4probe._process_scan(),
            "macsol_metal_v2_api_available": has_macsol_metal_v2(),
            "macsol_metal_v3_api_available": has_macsol_metal_v3(),
            "macsol_metal_v4_api_available": has_macsol_metal_v4(),
            "macsol_metal_v5_api_available": has_macsol_metal_v5(),
        },
        "prior_art_recheck": {
            "sol_engine_branch_head_observed": "01d4bfafc18b6e3c9955052e48890678ba6c0222",
            "pinned_semantics_source_used": "NVlabs/Sana df2e90ec912aa5e61d71650a93d2c57631df8ec2 sol_attn/triton_ref/fwd.py and project Wiki",
            "portable_semantics": "64-token blocks, threshold/sink/neighbor exactness, centroid/Vsum approximation, single online-softmax state",
        },
        "architecture_comparison": _architecture_comparison(),
        "candidate_contract": {
            "v5_status": MACSOL_METAL_V5_STATUS,
            "v5_description": MACSOL_METAL_V5_DESCRIPTION,
            "v5_decision_note": MACSOL_METAL_V5_DECISION_NOTE,
            "v5_rejection_reason": MACSOL_METAL_V5_REJECTION_REASON,
            "v2_status": MACSOL_METAL_V2_STATUS,
            "v3_status": MACSOL_METAL_V3_STATUS,
            "v4_status": MACSOL_METAL_V4_STATUS,
            "default_off_not_generation_path": True,
            "not_full_sparse_forward": True,
            "no_activation_int8": True,
            "no_dense_sxs_scores_or_dense_mask_in_normal_path": True,
        },
    }
    try:
        record["synthetic"] = run_synthetic(args)
        if not args.skip_real:
            record["real"] = run_real(args)
        record["ok"] = bool(record.get("synthetic", {}).get("ok")) and (args.skip_real or bool(record.get("real", {}).get("ok")))
    except Exception as exc:
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        record["ok"] = False
    metrics_after = v4probe._metrics()
    record["postflight"] = {"metrics_after": metrics_after, "metrics_delta": v4probe._delta(metrics_before, metrics_after)}
    record["decision"] = _decision(record, args)
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(record, indent=2 if args.pretty else None, sort_keys=True))
    latest = Path("experiments") / "macsol_metal_v5_latest.json"
    latest.write_text(json.dumps({"path": str(result_path), "recorded_at": record["recorded_at"], "decision": record["decision"]}, indent=2, sort_keys=True))
    print(json.dumps({"out": str(result_path), "ok": record["ok"], "decision": record["decision"]}, indent=2, sort_keys=True))
    # A timing/correctness rejection is a completed bounded decision, not a process failure.
    return 0 if record["decision"].get("decision") in {"reject_for_now_timing", "reject", "microkernel_signal_viable_needs_full_v5_integration", "synthetic_only_continue_needed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
