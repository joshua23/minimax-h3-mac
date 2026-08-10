#!/usr/bin/env python3
"""Bounded synthetic correctness/statistics probe for the MacSol reference path.

The probe uses H3-like packed Q/K/V tensors only.  It does not load model weights, run generation,
modify generation defaults, or invoke any Metal sparse kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from minimax_h3_mlx.macsol_reference import (  # noqa: E402
    H3PackedLengths,
    MacSolReferenceConfig,
    dense_attention_reference,
    h3_macsol_reference_attention,
    macsol_reference_attention,
)

DEFAULT_OUT = "experiments/macsol_reference/latest.json"


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


def _synthetic_h3_qkv(lengths: H3PackedLengths, *, heads: int, head_dim: int, seed: int) -> tuple[mx.array, mx.array, mx.array]:
    """Build deterministic H3-like Q/K/V with mild within-block K/V variation."""
    rng = np.random.default_rng(seed)
    sequence = lengths.sequence_length
    blocks = (sequence + 63) // 64
    block_ids = np.minimum(np.arange(sequence) // 64, blocks - 1)

    q = rng.standard_normal((1, heads, sequence, head_dim), dtype=np.float32) * 0.35
    k_centers = rng.standard_normal((1, heads, blocks, head_dim), dtype=np.float32) * 0.45
    v_centers = rng.standard_normal((1, heads, blocks, head_dim), dtype=np.float32) * 0.30
    k = k_centers[:, :, block_ids, :] + rng.standard_normal((1, heads, sequence, head_dim), dtype=np.float32) * 0.04
    v = v_centers[:, :, block_ids, :] + rng.standard_normal((1, heads, sequence, head_dim), dtype=np.float32) * 0.04

    # Prefix rows carry a small deterministic offset so the exact-prefix check is not vacuous.
    prefix = lengths.prefix_tokens
    q[:, :, :prefix, :] += 0.10
    k[:, :, :prefix, :] -= 0.05
    return mx.array(q), mx.array(k), mx.array(v)


def _error_metrics(got: mx.array, dense: mx.array, *, prefix: int) -> dict[str, Any]:
    diff = (got.astype(mx.float32) - dense.astype(mx.float32))
    mx.eval(diff)
    arr = np.asarray(diff)
    dense_arr = np.asarray(dense.astype(mx.float32))
    prefix_abs = np.abs(arr[:, :, :prefix, :]) if prefix else np.zeros((0,), dtype=np.float32)
    target_abs = np.abs(arr[:, :, prefix:, :]) if prefix < arr.shape[2] else np.zeros((0,), dtype=np.float32)
    return {
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "mean_abs": float(np.mean(np.abs(arr))) if arr.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0,
        "dense_rms": float(np.sqrt(np.mean(np.square(dense_arr)))) if dense_arr.size else 0.0,
        "relative_rms": (
            float(np.sqrt(np.mean(np.square(arr))) / max(np.sqrt(np.mean(np.square(dense_arr))), 1e-12))
            if dense_arr.size
            else 0.0
        ),
        "prefix_max_abs": float(np.max(prefix_abs)) if prefix_abs.size else 0.0,
        "target_max_abs": float(np.max(target_abs)) if target_abs.size else 0.0,
        "target_mean_abs": float(np.mean(target_abs)) if target_abs.size else 0.0,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    lengths = H3PackedLengths(
        text_tokens=args.text_tokens,
        conditioning_video_tokens=args.conditioning_video_tokens,
        audio_tokens=args.audio_tokens,
        target_video_tokens=args.target_video_tokens,
    )
    q, k, v = _synthetic_h3_qkv(lengths, heads=args.heads, head_dim=args.head_dim, seed=args.seed)
    _reset_mlx_peak()
    before = _metrics()
    started = time.perf_counter()

    dense = dense_attention_reference(q, k, v)
    approx = h3_macsol_reference_attention(q, k, v, lengths, tau=args.tau, block_size=64)
    all_exact = macsol_reference_attention(
        q,
        k,
        v,
        MacSolReferenceConfig(block_size=64, tau=-1.0e6, sink_start=0, sink_tokens=0),
    )
    mx.eval(dense, approx.output, all_exact.output)

    approx_errors = _error_metrics(approx.output, dense, prefix=lengths.prefix_tokens)
    all_exact_errors = _error_metrics(all_exact.output, dense, prefix=0)
    elapsed = time.perf_counter() - started
    after = _metrics()

    tolerances = {
        "all_exact_max_abs": args.all_exact_tolerance,
        "h3_prefix_max_abs": args.prefix_tolerance,
    }
    checks = {
        "all_exact_matches_dense": all_exact_errors["max_abs"] <= args.all_exact_tolerance,
        "h3_prefix_queries_match_dense": approx_errors["prefix_max_abs"] <= args.prefix_tolerance,
        "has_approximate_blocks_for_statistics": approx.stats["approximate_block_pairs"] > 0,
        "sink_covers_prefix_blocks": approx.stats["rounded_sink_token_range"][0] == 0
        and approx.stats["rounded_sink_token_range"][1] >= lengths.prefix_tokens,
    }
    ok = all(checks.values())
    return {
        "ok": ok,
        "recorded_at": _now(),
        "command": sys.argv,
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "parameters": {
            "lengths": lengths.as_dict(),
            "sequence_length": lengths.sequence_length,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "tau": args.tau,
            "block_size": 64,
            "seed": args.seed,
        },
        "tolerances": tolerances,
        "checks": checks,
        "approx_case": {
            "errors_vs_dense": approx_errors,
            "routing_stats": approx.stats,
        },
        "all_exact_case": {
            "errors_vs_dense": all_exact_errors,
            "routing_stats": all_exact.stats,
        },
        "elapsed_seconds": elapsed,
        "metrics_before": before,
        "metrics_after": after,
        "metrics_delta": _delta(before, after),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, help="JSON output path")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--text-tokens", type=int, default=77)
    parser.add_argument("--conditioning-video-tokens", type=int, default=64)
    parser.add_argument("--audio-tokens", type=int, default=80)
    parser.add_argument("--target-video-tokens", type=int, default=256)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--all-exact-tolerance", type=float, default=2e-5)
    parser.add_argument("--prefix-tolerance", type=float, default=2e-5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = run_probe(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2 if args.pretty else None, sort_keys=True) + "\n")
    print(
        "MacSol reference probe "
        f"ok={record['ok']} seq={record['parameters']['sequence_length']} "
        f"exact_ratio={record['approx_case']['routing_stats']['exact_block_ratio']:.4f} "
        f"target_max_abs={record['approx_case']['errors_vs_dense']['target_max_abs']:.3e} "
        f"all_exact_max_abs={record['all_exact_case']['errors_vs_dense']['max_abs']:.3e} "
        f"out={out}",
        flush=True,
    )
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
