#!/usr/bin/env python3
"""Probe archived/rejected MLX/Metal MacSol v1-v4 provenance paths.

The probe is intentionally bounded: synthetic runs preserve kernel parity evidence against
``macsol_reference.py``; real runs capture one 960x544 H3 Q/K/V tensor and benchmark a safe
contiguous head/query-block subset against sampled dense, MLX reference, rejected scalar Metal,
rejected SIMD-group cooperative v2 Metal, rejected q-block tiled v3 Metal, and rejected
q-block/SIMD-group v4 Metal behavior without materializing a full dense SxS matrix.  This script
records provenance only: no v1-v4 Metal path is a production, generation, or deployable profile
speed path, and this probe never promotes one automatically.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
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

from minimax_h3_mlx.macsol_metal import (  # noqa: E402
    MACSOL_METAL_REJECTION_REASON,
    MACSOL_METAL_STATUS,
    MACSOL_METAL_V2_DESCRIPTION,
    MACSOL_METAL_V2_REJECTION_REASON,
    MACSOL_METAL_V2_STATUS,
    MACSOL_METAL_V3_DESCRIPTION,
    MACSOL_METAL_V3_REJECTION_REASON,
    MACSOL_METAL_V3_STATUS,
    MACSOL_METAL_V4_DESCRIPTION,
    MACSOL_METAL_V4_REJECTION_REASON,
    MACSOL_METAL_V4_STATUS,
    has_macsol_metal,
    has_macsol_metal_v2,
    has_macsol_metal_v3,
    has_macsol_metal_v4,
    macsol_attention_metal_subset,
    macsol_attention_metal_subset_v2,
    macsol_attention_metal_subset_v3,
    macsol_attention_metal_subset_v4,
    macsol_prepare_metal,
)
from minimax_h3_mlx.macsol_reference import (  # noqa: E402
    H3PackedLengths,
    MacSolReferenceConfig,
    block_ranges,
    build_macsol_routing,
    dense_attention_reference,
    h3_macsol_config,
    h3_macsol_reference_attention,
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _memsize() -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
    except Exception:
        return None


def _current_rss_kib() -> int | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(out.strip())
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


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(after_value, (int, float)) and isinstance(before_value, (int, float)):
            out[key] = after_value - before_value
    return out


def _process_scan() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    try:
        ps_out = subprocess.check_output(["ps", "-axo", "pid,ppid,%cpu,%mem,rss,comm,args"], text=True)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "rows": []}
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
    high_threshold = 8 * 1024 * 1024
    return {
        "ok": True,
        "filter": "python|mlx|generate|ffmpeg|argus",
        "high_memory_threshold_rss_kib": high_threshold,
        "other_high_memory_processes": [
            row for row in rows if row["pid"] != os.getpid() and row["rss_kib"] >= high_threshold
        ],
        "rows": rows[:25],
    }


def _error_metrics(got: mx.array, ref: mx.array) -> dict[str, float]:
    diff = got.astype(mx.float32) - ref.astype(mx.float32)
    mx.eval(diff, ref)
    arr = np.asarray(diff, dtype=np.float32)
    ref_arr = np.asarray(ref.astype(mx.float32), dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    ref_rms = float(np.sqrt(np.mean(np.square(ref_arr)))) if ref_arr.size else 0.0
    return {
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "mean_abs": float(np.mean(np.abs(arr))) if arr.size else 0.0,
        "rms_abs": rms,
        "reference_rms": ref_rms,
        "relative_rms": rms / max(ref_rms, 1e-12),
    }


def _stack_block_summaries(x: mx.array, ranges: tuple[tuple[int, int], ...], *, reducer: str) -> mx.array:
    parts: list[mx.array] = []
    x32 = x.astype(mx.float32)
    for lo, hi in ranges:
        block = x32[:, :, lo:hi, :]
        if reducer == "mean":
            parts.append(mx.mean(block, axis=2))
        elif reducer == "sum":
            parts.append(mx.sum(block, axis=2))
        else:
            raise ValueError(f"unknown reducer {reducer!r}")
    return mx.stack(parts, axis=2)


def _stable_exact_approx_block(
    q_block: mx.array,
    exact_k: mx.array | None,
    exact_v: mx.array | None,
    approx_k: mx.array | None,
    approx_vsum: mx.array | None,
    approx_lengths: mx.array | None,
    *,
    scale: float,
    head_dim: int,
) -> mx.array:
    pieces: list[mx.array] = []
    exact_scores = None
    approx_scores = None
    if exact_k is not None and exact_k.shape[0] > 0:
        exact_scores = mx.matmul(q_block, exact_k.T) * scale
        pieces.append(mx.max(exact_scores, axis=-1, keepdims=True))
    if approx_k is not None and approx_k.shape[0] > 0:
        approx_scores = mx.matmul(q_block, approx_k.T) * scale
        pieces.append(mx.max(approx_scores, axis=-1, keepdims=True))
    if not pieces:
        raise ValueError("sampled query block has no exact or approximate KV blocks")
    row_max = pieces[0]
    for piece in pieces[1:]:
        row_max = mx.maximum(row_max, piece)

    tokens = int(q_block.shape[0])
    numerator = mx.zeros((tokens, head_dim), dtype=mx.float32)
    denominator = mx.zeros((tokens, 1), dtype=mx.float32)
    if exact_scores is not None and exact_v is not None:
        exact_weights = mx.exp(exact_scores - row_max)
        numerator = numerator + mx.matmul(exact_weights, exact_v)
        denominator = denominator + mx.sum(exact_weights, axis=-1, keepdims=True)
    if approx_scores is not None and approx_vsum is not None and approx_lengths is not None:
        approx_weights = mx.exp(approx_scores - row_max)
        numerator = numerator + mx.matmul(approx_weights, approx_vsum)
        denominator = denominator + mx.sum(approx_weights * approx_lengths[None, :], axis=-1, keepdims=True)
    return numerator / denominator


def _dense_subset(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    head_start: int,
    head_count: int,
    q_block_start: int,
    q_block_count: int,
    block_size: int,
    scale: float,
) -> mx.array:
    qlo = int(q_block_start) * int(block_size)
    qhi = min(int(q.shape[2]), qlo + int(q_block_count) * int(block_size))
    outs: list[mx.array] = []
    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    for h in range(int(head_start), int(head_start) + int(head_count)):
        scores = mx.matmul(q32[0, h, qlo:qhi, :], k32[0, h].T) * scale
        weights = mx.softmax(scores, axis=-1)
        outs.append(mx.matmul(weights, v32[0, h]))
    return mx.stack(outs, axis=0)[None, :, :, :]


def _reference_subset(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    routing: Any,
    config: MacSolReferenceConfig,
    *,
    head_start: int,
    head_count: int,
    q_block_start: int,
    q_block_count: int,
    scale: float,
) -> mx.array:
    k_mean = _stack_block_summaries(k, routing.kv_block_ranges, reducer="mean")
    v_sum = _stack_block_summaries(v, routing.kv_block_ranges, reducer="sum")
    mx.eval(k_mean, v_sum)
    kv_lengths_np = np.array([hi - lo for lo, hi in routing.kv_block_ranges], dtype=np.float32)
    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    head_dim = int(q.shape[-1])
    prefix_tokens = int(config.prefix_query_tokens if config.force_prefix_queries_dense else 0)

    head_outputs: list[mx.array] = []
    for h in range(int(head_start), int(head_start) + int(head_count)):
        q_outputs: list[mx.array] = []
        for qi in range(int(q_block_start), int(q_block_start) + int(q_block_count)):
            qlo, qhi = routing.q_block_ranges[qi]
            q_block = q32[0, h, qlo:qhi, :]
            exact_blocks = np.nonzero(routing.exact_block_mask[0, h, qi])[0].astype(int).tolist()
            approx_blocks = [idx for idx in range(len(routing.kv_block_ranges)) if idx not in exact_blocks]
            exact_k_parts = []
            exact_v_parts = []
            for kj in exact_blocks:
                klo, khi = routing.kv_block_ranges[kj]
                exact_k_parts.append(k32[0, h, klo:khi, :])
                exact_v_parts.append(v32[0, h, klo:khi, :])
            exact_k = mx.concatenate(exact_k_parts, axis=0) if exact_k_parts else None
            exact_v = mx.concatenate(exact_v_parts, axis=0) if exact_v_parts else None
            if approx_blocks:
                approx_k = mx.stack([k_mean[0, h, kj, :] for kj in approx_blocks], axis=0)
                approx_vsum = mx.stack([v_sum[0, h, kj, :] for kj in approx_blocks], axis=0)
                approx_lengths = mx.array(kv_lengths_np[approx_blocks], dtype=mx.float32)
            else:
                approx_k = None
                approx_vsum = None
                approx_lengths = None
            approx = _stable_exact_approx_block(
                q_block,
                exact_k,
                exact_v,
                approx_k,
                approx_vsum,
                approx_lengths,
                scale=scale,
                head_dim=head_dim,
            )
            if qlo < prefix_tokens:
                dense = _dense_subset(
                    q,
                    k,
                    v,
                    head_start=h,
                    head_count=1,
                    q_block_start=qi,
                    q_block_count=1,
                    block_size=int(config.block_size),
                    scale=scale,
                )[0, 0]
                prefix_count = max(0, min(qhi, prefix_tokens) - qlo)
                if prefix_count >= qhi - qlo:
                    approx = dense
                elif prefix_count > 0:
                    approx = mx.concatenate([dense[:prefix_count], approx[prefix_count:]], axis=0)
            q_outputs.append(approx)
        head_outputs.append(mx.concatenate(q_outputs, axis=0))
    return mx.stack(head_outputs, axis=0)[None, :, :, :]


def _summarize_timing(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "label": label,
        "runs": rows,
        "median_elapsed_seconds": float(statistics.median([row["elapsed_seconds"] for row in rows])) if rows else None,
        "min_elapsed_seconds": min((row["elapsed_seconds"] for row in rows), default=None),
        "max_elapsed_seconds": max((row["elapsed_seconds"] for row in rows), default=None),
    }


def _time_runs(label: str, runs: int, fn: Callable[[], mx.array]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in range(int(runs)):
        _reset_mlx_peak()
        before = _metrics()
        started = time.perf_counter()
        result = fn()
        mx.eval(result)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        after = _metrics()
        rows.append(
            {
                "run": run,
                "elapsed_seconds": elapsed,
                "metrics_before": before,
                "metrics_after": after,
                "metrics_delta": _delta(before, after),
            }
        )
    return _summarize_timing(label, rows)


def _time_interleaved(runs: int, entries: list[tuple[str, Callable[[], mx.array]]]) -> dict[str, Any]:
    """Run labels in alternating order so warm timing compares the same memory/runtime state."""
    rows_by_label: dict[str, list[dict[str, Any]]] = {label: [] for label, _ in entries}
    for run in range(int(runs)):
        for label, fn in entries:
            _reset_mlx_peak()
            before = _metrics()
            started = time.perf_counter()
            result = fn()
            mx.eval(result)
            mx.synchronize()
            elapsed = time.perf_counter() - started
            after = _metrics()
            rows_by_label[label].append(
                {
                    "run": run,
                    "elapsed_seconds": elapsed,
                    "metrics_before": before,
                    "metrics_after": after,
                    "metrics_delta": _delta(before, after),
                }
            )
    return {label: _summarize_timing(label, rows) for label, rows in rows_by_label.items()}


def _synthetic_qkv(lengths: H3PackedLengths, *, heads: int, head_dim: int, seed: int, dtype: str) -> tuple[mx.array, mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    shape = (1, int(heads), lengths.sequence_length, int(head_dim))
    q = mx.array(rng.standard_normal(shape, dtype=np.float32))
    k = mx.array(rng.standard_normal(shape, dtype=np.float32))
    v = mx.array(rng.standard_normal(shape, dtype=np.float32))
    if dtype == "bf16":
        q = q.astype(mx.bfloat16)
        k = k.astype(mx.bfloat16)
        v = v.astype(mx.bfloat16)
    return q, k, v


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    lengths = H3PackedLengths(
        text_tokens=33,
        conditioning_video_tokens=31,
        audio_tokens=1,
        target_video_tokens=195,
    )
    q, k, v = _synthetic_qkv(lengths, heads=2, head_dim=16, seed=args.seed, dtype=args.synthetic_dtype)
    cfg = h3_macsol_config(lengths, tau=args.tau, block_size=args.block_size)
    scale = float(q.shape[-1] ** -0.5)

    routing = build_macsol_routing(q, k, cfg, scale=scale)
    prep = macsol_prepare_metal(q, k, v, block_size=args.block_size, tau=args.tau, scale=scale)
    metal_v1 = macsol_attention_metal_subset(q, k, v, cfg, scale=scale, preparation=prep)
    metal_v2 = macsol_attention_metal_subset_v2(q, k, v, cfg, scale=scale, preparation=prep)
    metal_v3 = macsol_attention_metal_subset_v3(q, k, v, cfg, scale=scale, preparation=prep)
    metal_v4 = macsol_attention_metal_subset_v4(q, k, v, cfg, scale=scale, preparation=prep)
    reference = h3_macsol_reference_attention(q, k, v, lengths, tau=args.tau, block_size=args.block_size, scale=scale).output
    dense = dense_attention_reference(q, k, v, scale=scale)
    all_exact_cfg = MacSolReferenceConfig(
        block_size=args.block_size,
        tau=args.tau,
        sink_start=0,
        sink_tokens=lengths.sequence_length,
        prefix_query_tokens=0,
        force_prefix_queries_dense=False,
    )
    all_exact_metal_v1 = macsol_attention_metal_subset(q, k, v, all_exact_cfg, scale=scale).output
    all_exact_metal_v2 = macsol_attention_metal_subset_v2(q, k, v, all_exact_cfg, scale=scale).output
    all_exact_metal_v3 = macsol_attention_metal_subset_v3(q, k, v, all_exact_cfg, scale=scale).output
    all_exact_metal_v4 = macsol_attention_metal_subset_v4(q, k, v, all_exact_cfg, scale=scale).output
    mx.eval(
        metal_v1.output,
        metal_v2.output,
        metal_v3.output,
        metal_v4.output,
        reference,
        dense,
        all_exact_metal_v1,
        all_exact_metal_v2,
        all_exact_metal_v3,
        all_exact_metal_v4,
        prep.thresholds,
    )

    threshold_ref = mx.array(routing.thresholds[..., 0], dtype=mx.float32)
    threshold_error = _error_metrics(prep.thresholds, threshold_ref)
    metal_v1_reference_error = _error_metrics(metal_v1.output, reference)
    metal_v2_reference_error = _error_metrics(metal_v2.output, reference)
    metal_v3_reference_error = _error_metrics(metal_v3.output, reference)
    metal_v4_reference_error = _error_metrics(metal_v4.output, reference)
    metal_v2_v1_error = _error_metrics(metal_v2.output, metal_v1.output)
    metal_v3_v2_error = _error_metrics(metal_v3.output, metal_v2.output)
    metal_v4_v3_error = _error_metrics(metal_v4.output, metal_v3.output)
    metal_v4_dense_error = _error_metrics(metal_v4.output, dense)
    all_exact_v1_error = _error_metrics(all_exact_metal_v1, dense)
    all_exact_v2_error = _error_metrics(all_exact_metal_v2, dense)
    all_exact_v3_error = _error_metrics(all_exact_metal_v3, dense)
    all_exact_v4_error = _error_metrics(all_exact_metal_v4, dense)

    timings = _time_interleaved(
        args.synthetic_warm_runs,
        [
            ("dense_full", lambda: dense_attention_reference(q, k, v, scale=scale)),
            (
                "reference_full",
                lambda: h3_macsol_reference_attention(q, k, v, lengths, tau=args.tau, block_size=args.block_size, scale=scale).output,
            ),
            ("metal_v1_scalar_full", lambda: macsol_attention_metal_subset(q, k, v, cfg, scale=scale).output),
            ("metal_v2_cooperative_full", lambda: macsol_attention_metal_subset_v2(q, k, v, cfg, scale=scale).output),
            ("metal_v3_qblock_tiled_full", lambda: macsol_attention_metal_subset_v3(q, k, v, cfg, scale=scale).output),
            ("metal_v4_qblock_simd_full", lambda: macsol_attention_metal_subset_v4(q, k, v, cfg, scale=scale).output),
        ],
    )

    checks = {
        "metal_api_available": has_macsol_metal(),
        "metal_v2_api_available": has_macsol_metal_v2(),
        "metal_v3_api_available": has_macsol_metal_v3(),
        "metal_v4_api_available": has_macsol_metal_v4(),
        "thresholds_match_reference": threshold_error["max_abs"] <= args.threshold_tolerance,
        "metal_v1_matches_reference": metal_v1_reference_error["max_abs"] <= args.synthetic_tolerance,
        "metal_v2_matches_reference": metal_v2_reference_error["max_abs"] <= args.synthetic_tolerance,
        "metal_v3_matches_reference": metal_v3_reference_error["max_abs"] <= args.synthetic_tolerance,
        "metal_v4_matches_reference": metal_v4_reference_error["max_abs"] <= args.synthetic_tolerance,
        "all_exact_v1_matches_dense": all_exact_v1_error["max_abs"] <= args.synthetic_tolerance,
        "all_exact_v2_matches_dense": all_exact_v2_error["max_abs"] <= args.synthetic_tolerance,
        "all_exact_v3_matches_dense": all_exact_v3_error["max_abs"] <= args.synthetic_tolerance,
        "all_exact_v4_matches_dense": all_exact_v4_error["max_abs"] <= args.synthetic_tolerance,
        "reference_has_approximate_blocks": int(routing.summary()["approximate_block_pairs"]) > 0,
    }
    return {
        "ok": all(checks.values()),
        "parameters": {
            "lengths": lengths.as_dict(),
            "sequence_length": lengths.sequence_length,
            "heads": 2,
            "head_dim": 16,
            "dtype": args.synthetic_dtype,
            "tau": args.tau,
            "block_size": args.block_size,
            "scale": scale,
        },
        "checks": checks,
        "errors": {
            "thresholds_vs_reference": threshold_error,
            "metal_v1_scalar_vs_macsol_reference": metal_v1_reference_error,
            "metal_v2_cooperative_vs_macsol_reference": metal_v2_reference_error,
            "metal_v3_qblock_tiled_vs_macsol_reference": metal_v3_reference_error,
            "metal_v4_qblock_simd_vs_macsol_reference": metal_v4_reference_error,
            "metal_v2_cooperative_vs_v1_scalar": metal_v2_v1_error,
            "metal_v3_qblock_tiled_vs_v2_cooperative": metal_v3_v2_error,
            "metal_v4_qblock_simd_vs_v3_qblock_tiled": metal_v4_v3_error,
            "metal_v4_qblock_simd_vs_dense": metal_v4_dense_error,
            "all_exact_v1_scalar_vs_dense": all_exact_v1_error,
            "all_exact_v2_cooperative_vs_dense": all_exact_v2_error,
            "all_exact_v3_qblock_tiled_vs_dense": all_exact_v3_error,
            "all_exact_v4_qblock_simd_vs_dense": all_exact_v4_error,
        },
        "routing_stats": routing.summary(),
        "timings": timings,
    }


def _load_real_harness():
    path = ROOT / "scripts" / "profile_real_qkv_macsol.py"
    spec = importlib.util.spec_from_file_location("profile_real_qkv_macsol_imported", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load real-QKV harness from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _auto_q_block(lengths: H3PackedLengths, sequence_length: int, block_size: int, requested: int) -> int:
    block_count = int(math.ceil(sequence_length / block_size))
    if requested >= 0:
        if requested >= block_count:
            raise ValueError(f"requested q block {requested} is outside [0,{block_count})")
        return int(requested)
    first_target = max(0, lengths.prefix_tokens // block_size)
    last_target = block_count - 1
    return int((first_target + last_target) // 2)


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    process_scan = _process_scan()
    if process_scan.get("other_high_memory_processes"):
        return {
            "ok": False,
            "skipped": True,
            "blocker": "other_high_memory_process_present",
            "process_scan": process_scan,
        }

    realmod = _load_real_harness()
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
    load_before = _metrics()
    block, cfg_model, block_meta = realmod._load_block(real_args, events)
    load_after = _metrics()
    inputs_before = _metrics()
    x, modulation, adaln_indices, rotary, position_ids, lengths, sequence_meta = realmod._build_inputs(
        cfg_model, real_args, block, events
    )
    inputs_after = _metrics()
    capture_before = _metrics()
    q, k, v, capture_meta = realmod._capture_attention_qkv(block, x, modulation, adaln_indices, rotary)
    capture_after = _metrics()

    sequence_length = int(q.shape[2])
    head_dim = int(q.shape[-1])
    scale = float(head_dim ** -0.5)
    q_block_start = _auto_q_block(lengths, sequence_length, args.block_size, args.real_q_block_start)
    q_block_count = int(args.real_q_block_count)
    head_start = int(args.real_head_start)
    head_count = int(args.real_head_count)
    if head_start + head_count > int(q.shape[1]):
        raise ValueError(f"requested heads [{head_start},{head_start + head_count}) exceed {q.shape[1]}")

    cfg = h3_macsol_config(lengths, tau=args.tau, block_size=args.block_size)
    routing = build_macsol_routing(q, k, cfg, scale=scale)
    routing_stats = routing.summary()
    routing_stats.update(realmod._effective_exact_block_stats(routing, lengths.prefix_tokens))

    # Compute one set of outputs before timings so correctness failures are reported separately.
    prep = macsol_prepare_metal(q, k, v, block_size=args.block_size, tau=args.tau, scale=scale)
    metal_v1 = macsol_attention_metal_subset(
        q,
        k,
        v,
        cfg,
        q_block_start=q_block_start,
        q_block_count=q_block_count,
        head_start=head_start,
        head_count=head_count,
        scale=scale,
        preparation=prep,
    ).output
    metal_v2 = macsol_attention_metal_subset_v2(
        q,
        k,
        v,
        cfg,
        q_block_start=q_block_start,
        q_block_count=q_block_count,
        head_start=head_start,
        head_count=head_count,
        scale=scale,
        preparation=prep,
    ).output
    metal_v3 = macsol_attention_metal_subset_v3(
        q,
        k,
        v,
        cfg,
        q_block_start=q_block_start,
        q_block_count=q_block_count,
        head_start=head_start,
        head_count=head_count,
        scale=scale,
        preparation=prep,
    ).output
    metal_v4 = macsol_attention_metal_subset_v4(
        q,
        k,
        v,
        cfg,
        q_block_start=q_block_start,
        q_block_count=q_block_count,
        head_start=head_start,
        head_count=head_count,
        scale=scale,
        preparation=prep,
    ).output
    dense = _dense_subset(
        q,
        k,
        v,
        head_start=head_start,
        head_count=head_count,
        q_block_start=q_block_start,
        q_block_count=q_block_count,
        block_size=args.block_size,
        scale=scale,
    )
    reference = _reference_subset(
        q,
        k,
        v,
        routing,
        cfg,
        head_start=head_start,
        head_count=head_count,
        q_block_start=q_block_start,
        q_block_count=q_block_count,
        scale=scale,
    )
    mx.eval(metal_v1, metal_v2, metal_v3, metal_v4, dense, reference)
    metal_v1_vs_reference = _error_metrics(metal_v1, reference)
    metal_v2_vs_reference = _error_metrics(metal_v2, reference)
    metal_v3_vs_reference = _error_metrics(metal_v3, reference)
    metal_v4_vs_reference = _error_metrics(metal_v4, reference)
    metal_v2_vs_v1 = _error_metrics(metal_v2, metal_v1)
    metal_v3_vs_v2 = _error_metrics(metal_v3, metal_v2)
    metal_v4_vs_v3 = _error_metrics(metal_v4, metal_v3)
    metal_v4_vs_dense = _error_metrics(metal_v4, dense)
    reference_vs_dense = _error_metrics(reference, dense)

    timings = _time_interleaved(
        args.real_warm_runs,
        [
            (
                "dense_sampled_row_block",
                lambda: _dense_subset(
                    q,
                    k,
                    v,
                    head_start=head_start,
                    head_count=head_count,
                    q_block_start=q_block_start,
                    q_block_count=q_block_count,
                    block_size=args.block_size,
                    scale=scale,
                ),
            ),
            (
                "reference_sampled_row_block",
                lambda: _reference_subset(
                    q,
                    k,
                    v,
                    routing,
                    cfg,
                    head_start=head_start,
                    head_count=head_count,
                    q_block_start=q_block_start,
                    q_block_count=q_block_count,
                    scale=scale,
                ),
            ),
            (
                "metal_v1_scalar_sampled_row_block",
                lambda: macsol_attention_metal_subset(
                    q,
                    k,
                    v,
                    cfg,
                    q_block_start=q_block_start,
                    q_block_count=q_block_count,
                    head_start=head_start,
                    head_count=head_count,
                    scale=scale,
                ).output,
            ),
            (
                "metal_v2_cooperative_sampled_row_block",
                lambda: macsol_attention_metal_subset_v2(
                    q,
                    k,
                    v,
                    cfg,
                    q_block_start=q_block_start,
                    q_block_count=q_block_count,
                    head_start=head_start,
                    head_count=head_count,
                    scale=scale,
                ).output,
            ),
            (
                "metal_v3_qblock_tiled_sampled_row_block",
                lambda: macsol_attention_metal_subset_v3(
                    q,
                    k,
                    v,
                    cfg,
                    q_block_start=q_block_start,
                    q_block_count=q_block_count,
                    head_start=head_start,
                    head_count=head_count,
                    scale=scale,
                ).output,
            ),
            (
                "metal_v4_qblock_simd_sampled_row_block",
                lambda: macsol_attention_metal_subset_v4(
                    q,
                    k,
                    v,
                    cfg,
                    q_block_start=q_block_start,
                    q_block_count=q_block_count,
                    head_start=head_start,
                    head_count=head_count,
                    scale=scale,
                ).output,
            ),
        ],
    )

    v1_median = timings["metal_v1_scalar_sampled_row_block"]["median_elapsed_seconds"]
    v2_median = timings["metal_v2_cooperative_sampled_row_block"]["median_elapsed_seconds"]
    v3_median = timings["metal_v3_qblock_tiled_sampled_row_block"]["median_elapsed_seconds"]
    v4_median = timings["metal_v4_qblock_simd_sampled_row_block"]["median_elapsed_seconds"]
    dense_median = timings["dense_sampled_row_block"]["median_elapsed_seconds"]
    reference_median = timings["reference_sampled_row_block"]["median_elapsed_seconds"]
    speedup_vs_dense = (dense_median / v4_median) if dense_median and v4_median else None
    speedup_vs_reference = (reference_median / v4_median) if reference_median and v4_median else None
    speedup_vs_v1 = (v1_median / v4_median) if v1_median and v4_median else None
    speedup_vs_v2 = (v2_median / v4_median) if v2_median and v4_median else None
    speedup_vs_v3 = (v3_median / v4_median) if v3_median and v4_median else None
    parity_ok = (
        metal_v4_vs_reference["max_abs"] <= args.real_reference_tolerance
        or metal_v4_vs_reference["relative_rms"] <= args.real_reference_relative_tolerance
    )

    return {
        "ok": parity_ok,
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
            "q_block_count": q_block_count,
            "q_token_range": [q_block_start * args.block_size, min(sequence_length, (q_block_start + q_block_count) * args.block_size)],
            "scale": scale,
            "full_dense_not_materialized": True,
            "dense_comparison": "sampled query-block rows against all K/V tokens",
        },
        "block_load": {"meta": block_meta, "metrics_before": load_before, "metrics_after": load_after, "metrics_delta": _delta(load_before, load_after)},
        "input_build": {"metrics_before": inputs_before, "metrics_after": inputs_after, "metrics_delta": _delta(inputs_before, inputs_after)},
        "qkv_capture": {
            **capture_meta,
            "metrics_before": capture_before,
            "metrics_after": capture_after,
            "metrics_delta": _delta(capture_before, capture_after),
        },
        "sequence": sequence_meta,
        "routing_stats": routing_stats,
        "errors": {
            "metal_v1_scalar_vs_macsol_reference_subset": metal_v1_vs_reference,
            "metal_v2_cooperative_vs_macsol_reference_subset": metal_v2_vs_reference,
            "metal_v3_qblock_tiled_vs_macsol_reference_subset": metal_v3_vs_reference,
            "metal_v4_qblock_simd_vs_macsol_reference_subset": metal_v4_vs_reference,
            "metal_v2_cooperative_vs_v1_scalar_subset": metal_v2_vs_v1,
            "metal_v3_qblock_tiled_vs_v2_cooperative_subset": metal_v3_vs_v2,
            "metal_v4_qblock_simd_vs_v3_qblock_tiled_subset": metal_v4_vs_v3,
            "metal_v4_qblock_simd_vs_dense_subset": metal_v4_vs_dense,
            "macsol_reference_vs_dense_subset": reference_vs_dense,
        },
        "timings": timings,
        "speedups": {
            "metal_v4_qblock_simd_vs_dense_sampled_row_block": speedup_vs_dense,
            "metal_v4_qblock_simd_vs_mlx_reference_sampled_row_block": speedup_vs_reference,
            "metal_v4_qblock_simd_vs_v1_scalar_sampled_row_block": speedup_vs_v1,
            "metal_v4_qblock_simd_vs_v2_cooperative_sampled_row_block": speedup_vs_v2,
            "metal_v4_qblock_simd_vs_v3_qblock_tiled_sampled_row_block": speedup_vs_v3,
        },
    }


def _decision(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if record.get("failure"):
        return {"decision": "blocker", "reason": record["failure"].get("message", "unhandled failure")}
    synthetic = record.get("synthetic") or {}
    real = record.get("real") or {}
    if not synthetic.get("ok"):
        return {"decision": "reject", "reason": "synthetic parity failed", "checks": synthetic.get("checks")}
    if args.skip_real:
        return {
            "decision": "archive_only_synthetic_parity",
            "reason": "synthetic parity passed, but v1-v4 are rejected/provenance-only and real probe was skipped",
        }
    if real.get("skipped"):
        return {"decision": "blocker", "reason": real.get("blocker", "real probe skipped"), "process_scan": real.get("process_scan")}
    if not real.get("ok"):
        return {
            "decision": "reject",
            "reason": "real-QKV Metal v4 output did not match MLX MacSol reference subset",
            "errors": real.get("errors"),
        }
    speedups = real.get("speedups") or {}
    speedup_vs_dense = speedups.get("metal_v4_qblock_simd_vs_dense_sampled_row_block")
    decision = {
        "decision": "archive_only_rejected_v4",
        "reason": "v1-v4 are archived rejected probes; this script records parity/timing only and cannot promote a deployable path",
        "speedups": speedups,
    }
    if speedup_vs_dense is not None and speedup_vs_dense >= 1.05:
        decision["unexpected_speedup_note"] = (
            "treat as a new bounded candidate-design question, not as automatic promotion of archived v4"
        )
    return decision


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
    parser.add_argument("--synthetic-tolerance", type=float, default=5e-5)
    parser.add_argument("--threshold-tolerance", type=float, default=5e-5)
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
    parser.add_argument("--real-q-block-count", type=int, default=1)
    parser.add_argument("--real-warm-runs", type=int, default=2)
    parser.add_argument("--real-reference-tolerance", type=float, default=2e-3)
    parser.add_argument("--real-reference-relative-tolerance", type=float, default=1e-5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(args.out_dir) if args.out_dir else Path("experiments") / f"macsol_metal_v4_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "ok": False,
        "recorded_at": _now(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "hw_memsize_bytes": _memsize(),
        },
        "preflight": {
            "metrics": _metrics(),
            "process_scan": _process_scan(),
            "macsol_metal_api_available": has_macsol_metal(),
            "macsol_metal_v2_api_available": has_macsol_metal_v2(),
            "macsol_metal_v3_api_available": has_macsol_metal_v3(),
            "macsol_metal_v4_api_available": has_macsol_metal_v4(),
        },
        "archive_contract": {
            "active_deployable_candidate_status": "none_v1_v4_archive_only",
            "archived_probe_status": MACSOL_METAL_V4_STATUS,
            "v1_status": MACSOL_METAL_STATUS,
            "v1_rejection_reason": MACSOL_METAL_REJECTION_REASON,
            "v2_status": MACSOL_METAL_V2_STATUS,
            "v2_description": MACSOL_METAL_V2_DESCRIPTION,
            "v2_rejection_reason": MACSOL_METAL_V2_REJECTION_REASON,
            "v3_status": MACSOL_METAL_V3_STATUS,
            "v3_description": MACSOL_METAL_V3_DESCRIPTION,
            "v3_rejection_reason": MACSOL_METAL_V3_REJECTION_REASON,
            "v4_status": MACSOL_METAL_V4_STATUS,
            "v4_description": MACSOL_METAL_V4_DESCRIPTION,
            "v4_rejection_reason": MACSOL_METAL_V4_REJECTION_REASON,
            "default_off_not_generation_path": True,
            "internal_probe_only_not_speed_path": True,
            "v1_to_v4_archive_only_not_selectable_as_deployable_profile": True,
            "no_activation_int8": True,
            "routing_mode": "threshold only; budget_topk, threshold_h3_structure, and spatial-tube variants are rejected provenance-only",
            "no_dense_sxs_scores_or_dense_mask_in_metal_path": True,
            "summary_tensors": "Q mean, K mean, V sum, thresholds/proxy stats only",
            "forward": "v1 scalar, v2 SIMD-group cooperative, v3 q-block tiled, and v4 q-block/SIMD-group probes use one stable softmax denominator/numerator mixing exact token blocks and centroid/V-sum approximate blocks",
        },
    }
    try:
        record["synthetic"] = run_synthetic(args)
        if not args.skip_real:
            record["real"] = run_real(args)
        record["ok"] = bool(record.get("synthetic", {}).get("ok")) and (args.skip_real or bool(record.get("real", {}).get("ok")))
    except Exception as exc:  # record exact compile/runtime blocker instead of substituting a heuristic.
        record["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        record["ok"] = False
    record["decision"] = _decision(record, args)
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(record, indent=2 if args.pretty else None, sort_keys=True))
    latest = Path("experiments") / "macsol_metal_latest.json"
    latest.write_text(json.dumps({"path": str(result_path), "recorded_at": record["recorded_at"], "decision": record["decision"]}, indent=2))
    print(json.dumps({"out": str(result_path), "ok": record["ok"], "decision": record["decision"]}, indent=2))
    return 0 if record["ok"] or record["decision"].get("decision") in {"reject_for_now_timing", "reject"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
