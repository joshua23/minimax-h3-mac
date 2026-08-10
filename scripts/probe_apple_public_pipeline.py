#!/usr/bin/env python3
"""Audit public Apple/Metal/MLX mechanisms for the MiniMax-H3 M4 Pro route.

This is a bounded probe, not a generation kernel. It records which public surfaces are
available through the local SDK/MLX installation and runs tiny MLX custom-Metal
microbenchmarks for the mechanisms that could plausibly matter to the current H3
hotpaths: SIMD-group reductions and threadgroup scratch memory. The decision is
intentionally H3-specific and weighs the microbenchmarks against existing captured
MacSol v2/v5 evidence rather than treating a generic primitive win as a deployment win.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return None


def _cmd(argv: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        out = subprocess.check_output(argv, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return {"ok": True, "stdout": out.strip()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _vm_stat_value(label: str) -> int | None:
    try:
        out = subprocess.check_output(["vm_stat"], text=True, timeout=5)
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
        return subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True, timeout=5).strip()
    except Exception:
        return None


def _mlx_call(name: str) -> int | None:
    for owner in (mx, getattr(mx, "metal", None)):
        func = getattr(owner, name, None) if owner is not None else None
        if func is None:
            continue
        try:
            return int(func())
        except Exception:
            return None
    return None


def _reset_peak() -> None:
    for owner in (mx, getattr(mx, "metal", None)):
        func = getattr(owner, "reset_peak_memory", None) if owner is not None else None
        if func is None:
            continue
        try:
            func()
            return
        except Exception:
            continue


def _memory_snapshot() -> dict[str, Any]:
    return {
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


def _read_text(path: Path, *, max_bytes: int = 1_000_000) -> str | None:
    try:
        data = path.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return None


def _sdk_audit() -> dict[str, Any]:
    sdk = _cmd(["xcrun", "--sdk", "macosx", "--show-sdk-path"])
    sdk_path = Path(sdk["stdout"]) if sdk.get("ok") else None
    metal_headers = None
    resource_header = None
    resource_symbols: dict[str, bool] = {}
    io_symbols: dict[str, bool] = {}
    if sdk_path is not None:
        metal_headers = sdk_path / "System/Library/Frameworks/Metal.framework/Headers"
        resource_header = metal_headers / "MTLResource.h"
        resource_text = _read_text(resource_header)
        if resource_text is not None:
            for token in (
                "MTLCPUCacheModeWriteCombined",
                "MTLResourceStorageModeShared",
                "MTLResourceStorageModePrivate",
                "MTLHazardTrackingMode",
            ):
                resource_symbols[token] = token in resource_text
        io_header = metal_headers / "MTLIOCommandQueue.h"
        io_text = _read_text(io_header)
        if io_text is not None:
            for token in ("MTLIOCommandQueue", "MTLIOCommandBuffer", "loadBytes"):
                io_symbols[token] = token in io_text
    return {
        "xcrun_sdk": sdk,
        "xcrun_metal": _cmd(["xcrun", "--find", "metal"]),
        "xcrun_metallib": _cmd(["xcrun", "--find", "metallib"]),
        "metal_headers_path": str(metal_headers) if metal_headers is not None else None,
        "metal_headers_exist": bool(metal_headers and metal_headers.exists()),
        "resource_header": str(resource_header) if resource_header is not None else None,
        "resource_symbols": resource_symbols,
        "metal_io_symbols": io_symbols,
    }


def _mlx_surface_audit() -> dict[str, Any]:
    metal = getattr(mx, "metal", None)
    fast = getattr(mx, "fast", None)
    metal_names = sorted(dir(metal)) if metal is not None else []
    memory_names = [n for n in metal_names if any(t in n.lower() for t in ("memory", "cache", "wired", "limit"))]
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx_version": _version("mlx"),
        "numpy_version": _version("numpy"),
        "default_device": str(mx.default_device()),
        "has_mx_fast_metal_kernel": hasattr(fast, "metal_kernel"),
        "mx_metal_memory_cache_wired_symbols": memory_names,
        "has_clear_cache": hasattr(mx, "clear_cache"),
    }


@lru_cache(maxsize=1)
def _scalar_dot_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_public_scalar_d128_dot",
        input_names=["a", "b"],
        output_names=["out"],
        source=r"""
            uint row = thread_position_in_grid.x;
            if (row >= ROWS) { return; }
            uint base = row * uint(D);
            float acc = 0.0f;
            for (uint d = 0; d < D; ++d) {
                acc += static_cast<float>(a[base + d]) * static_cast<float>(b[base + d]);
            }
            out[row] = acc;
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _simd_dot_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_public_simd_sum_d128_dot",
        input_names=["a", "b"],
        output_names=["out"],
        source=r"""
            uint row = threadgroup_position_in_grid.y;
            uint lane = thread_index_in_simdgroup;
            if (row >= ROWS) { return; }
            uint base = row * uint(D);
            float partial = 0.0f;
            for (uint d = lane; d < D; d += 32) {
                partial += static_cast<float>(a[base + d]) * static_cast<float>(b[base + d]);
            }
            float acc = simd_sum(partial);
            if (lane == 0) { out[row] = acc; }
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _threadgroup_dot_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_public_threadgroup_d128_dot",
        input_names=["a", "b"],
        output_names=["out"],
        source=r"""
            uint row = threadgroup_position_in_grid.y;
            uint lid = thread_index_in_threadgroup;
            threadgroup float scratch[256];
            float partial = 0.0f;
            if (row < ROWS && lid < D) {
                uint idx = row * uint(D) + lid;
                partial = static_cast<float>(a[idx]) * static_cast<float>(b[idx]);
            }
            scratch[lid] = partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 128; stride > 0; stride >>= 1) {
                if (lid < stride) { scratch[lid] += scratch[lid + stride]; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0 && row < ROWS) { out[row] = scratch[0]; }
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _simdgroup_matrix_decl_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_public_simdgroup_matrix_decl_probe",
        input_names=["x"],
        output_names=["out"],
        source=r"""
            uint tid = thread_position_in_grid.x;
            simdgroup_matrix<float, 8, 8> tile;
            out[tid] = x[tid] + 0.0f;
        """,
        compile_options={"math_mode": "safe"},
    )


def _call_scalar(a: mx.array, b: mx.array, rows: int, dim: int) -> mx.array:
    kernel = _scalar_dot_kernel()
    if kernel is None:
        raise RuntimeError("mx.fast.metal_kernel unavailable")
    out = kernel(
        inputs=[a, b],
        template=[("ROWS", rows), ("D", dim), ("T", a.dtype)],
        grid=(rows, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(out)
    return out


def _call_simd(a: mx.array, b: mx.array, rows: int, dim: int) -> mx.array:
    kernel = _simd_dot_kernel()
    if kernel is None:
        raise RuntimeError("mx.fast.metal_kernel unavailable")
    out = kernel(
        inputs=[a, b],
        template=[("ROWS", rows), ("D", dim), ("T", a.dtype)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(out)
    return out


def _call_threadgroup(a: mx.array, b: mx.array, rows: int, dim: int) -> mx.array:
    kernel = _threadgroup_dot_kernel()
    if kernel is None:
        raise RuntimeError("mx.fast.metal_kernel unavailable")
    out = kernel(
        inputs=[a, b],
        template=[("ROWS", rows), ("D", dim), ("T", a.dtype)],
        grid=(256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(out)
    return out


def _try_simdgroup_matrix_decl() -> dict[str, Any]:
    kernel = _simdgroup_matrix_decl_kernel()
    if kernel is None:
        return {"ok": False, "reason": "mx.fast.metal_kernel unavailable"}
    x = mx.zeros((1,), dtype=mx.float32)
    try:
        out = kernel(
            inputs=[x],
            template=[("T", x.dtype)],
            grid=(1, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.float32],
        )[0]
        mx.eval(out)
        return {"ok": True, "output0": float(np.asarray(out)[0])}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _error(got: mx.array, ref: mx.array) -> dict[str, float]:
    diff = got.astype(mx.float32) - ref.astype(mx.float32)
    mx.eval(diff, ref)
    arr = np.asarray(diff, dtype=np.float32)
    ref_arr = np.asarray(ref.astype(mx.float32), dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    ref_rms = float(np.sqrt(np.mean(np.square(ref_arr)))) if ref_arr.size else 0.0
    return {
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "relative_rms": rms / max(ref_rms, 1e-12),
    }


def _time_fn(label: str, fn: Callable[[], mx.array], *, warmups: int, runs: int) -> dict[str, Any]:
    for _ in range(warmups):
        fn()
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        "label": label,
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def _microbench(rows: int, dim: int, runs: int, warmups: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    a = mx.array(rng.standard_normal((rows, dim), dtype=np.float32)).astype(mx.bfloat16)
    b = mx.array(rng.standard_normal((rows, dim), dtype=np.float32)).astype(mx.bfloat16)
    ref = mx.sum(a.astype(mx.float32) * b.astype(mx.float32), axis=1)
    mx.eval(a, b, ref)

    _reset_peak()
    before = _memory_snapshot()
    timings: dict[str, Any] = {}
    outputs: dict[str, mx.array] = {}
    errors: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for label, fn in (
        ("scalar_one_thread_per_row", lambda: _call_scalar(a, b, rows, dim)),
        ("simdgroup_simd_sum_one_group_per_row", lambda: _call_simd(a, b, rows, dim)),
        ("threadgroup_scratch_reduction_one_group_per_row", lambda: _call_threadgroup(a, b, rows, dim)),
    ):
        try:
            out = fn()
            outputs[label] = out
            errors[label] = _error(out, ref)
            timings[label] = _time_fn(label, fn, warmups=warmups, runs=runs)
        except Exception as exc:
            failures[label] = f"{type(exc).__name__}: {exc}"
    after = _memory_snapshot()

    med = {k: v["median_seconds"] for k, v in timings.items()}
    speedups: dict[str, float] = {}
    scalar = med.get("scalar_one_thread_per_row")
    if scalar and scalar > 0:
        for k, v in med.items():
            speedups[f"scalar_over_{k}"] = scalar / v if v > 0 else math.inf
    return {
        "shape": {"rows": rows, "dim": dim, "dtype": "bfloat16_inputs_float32_accumulation"},
        "timings": timings,
        "errors_vs_mlx_sum": errors,
        "failures": failures,
        "speedups": speedups,
        "memory_before": before,
        "memory_after": after,
        "memory_delta": _delta(before, after),
    }


def _load_prior(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _prior_h3_evidence() -> dict[str, Any]:
    v2 = _load_prior(ROOT / "experiments/macsol_metal_v2_20260810T115649Z/result.json")
    v5 = _load_prior(ROOT / "experiments/macsol_metal_v5_20260810T130615Z/result.json")
    dense_ab = _load_prior(ROOT / "experiments/dense_dequant_combo_20260810T133038Z/result.json")
    text_cache = _load_prior(ROOT / "experiments/text_conditioning_cache_ab_20260810T151018Z/result.json")
    out: dict[str, Any] = {}
    if v2:
        out["macsol_v2_simdgroup_summary"] = {
            "path": "experiments/macsol_metal_v2_20260810T115649Z/result.json",
            "status": v2.get("status") or v2.get("decision", {}).get("verdict"),
            "decision": v2.get("decision"),
            "real_timing": v2.get("real", {}).get("timing") or v2.get("real_timing"),
        }
    if v5:
        out["macsol_v5_exact_tile_summary"] = {
            "path": "experiments/macsol_metal_v5_20260810T130615Z/result.json",
            "status": v5.get("status") or v5.get("decision", {}).get("verdict"),
            "decision": v5.get("decision"),
            "real_timing": v5.get("real", {}).get("timing") or v5.get("real_timing"),
            "exact_tile_lower_bound": v5.get("real", {}).get("exact_tile_lower_bound") or v5.get("exact_tile_lower_bound"),
        }
    if dense_ab:
        out["dense_dequant_generation_ab"] = {
            "path": "experiments/dense_dequant_combo_20260810T133038Z/result.json",
            "verdict": dense_ab.get("decision", {}).get("verdict"),
            "reason": dense_ab.get("decision", {}).get("reason"),
        }
    if text_cache:
        out["text_conditioning_cache_generation_ab"] = {
            "path": "experiments/text_conditioning_cache_ab_20260810T151018Z/result.json",
            "verdict": text_cache.get("decision", {}).get("verdict"),
            "reason": text_cache.get("decision", {}).get("reason"),
        }
    return out


def _decide(audit: dict[str, Any], microbench: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    timings = microbench.get("timings", {})
    failures = microbench.get("failures", {})
    memory_delta = microbench.get("memory_delta", {})
    simd_ok = "simdgroup_simd_sum_one_group_per_row" in timings
    tg_ok = "threadgroup_scratch_reduction_one_group_per_row" in timings
    matrix_ok = bool(audit.get("simdgroup_matrix_decl_probe", {}).get("ok"))
    pageouts = int(memory_delta.get("vm_pageouts", 0) or 0)
    swapouts = int(memory_delta.get("vm_swapouts", 0) or 0)
    scalar = timings.get("scalar_one_thread_per_row", {}).get("median_seconds")
    simd = timings.get("simdgroup_simd_sum_one_group_per_row", {}).get("median_seconds")
    simd_speedup = (scalar / simd) if scalar and simd else None

    reasons: list[str] = []
    if simd_ok:
        reasons.append(f"MLX custom Metal can launch simd_sum reductions; D=128 dot microbench scalar-to-simd median ratio={simd_speedup:.2f}x" if simd_speedup else "MLX custom Metal can launch simd_sum reductions")
    else:
        reasons.append(f"simd_sum probe failed: {failures.get('simdgroup_simd_sum_one_group_per_row')}")
    if tg_ok:
        reasons.append("threadgroup scratch memory/barrier probe launched")
    else:
        reasons.append(f"threadgroup scratch probe failed: {failures.get('threadgroup_scratch_reduction_one_group_per_row')}")
    if matrix_ok:
        reasons.append("simdgroup_matrix type parsed in MLX custom Metal, but this probe did not expose a packed 4-bit/H3 GEMM primitive")
    else:
        reasons.append("simdgroup_matrix was not established as a usable MLX route by the local declaration probe")
    if pageouts or swapouts:
        reasons.append(f"small probe saw pageout/swapout deltas {pageouts}/{swapouts}, so no memory-pressure improvement is established")
    else:
        reasons.append("small probe itself was pageout/swapout clean, but it does not reduce H3 model residency")
    if prior.get("macsol_v2_simdgroup_summary") or prior.get("macsol_v5_exact_tile_summary"):
        reasons.append("existing captured-real H3 MacSol v2/v5 evidence already tested these public simd/threadgroup ideas and rejected them on the exact-block cost model")

    return {
        "verdict": "unsupported_close_for_now",
        "promoted_followup_route": None,
        "recommended_next_direction": "Stop kernel-budget spending on public Apple SIMD/threadgroup/MLX custom-Metal MacSol variants for now; pivot to memory lifecycle and asset/quality-bounded routes such as fresh idle memory gating, component residency cleanup, valid source-asset quantization, or teacher-forced quality-bounded cache/distillation evidence.",
        "memory_promote_gate": pageouts == 0 and swapouts == 0,
        "public_mechanism_with_h3_specific_benefit": False,
        "reasons": reasons,
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    decision = result["decision"]
    mb = result["microbench"]
    timings = mb.get("timings", {})
    lines = [
        "# Apple public pipeline audit for MiniMax-H3 on M4 Pro",
        "",
        f"Created: {result['created_utc']}",
        "",
        "## Verdict",
        "",
        f"**{decision['verdict']}**. No follow-up kernel route is promoted from this audit.",
        "",
        decision["recommended_next_direction"],
        "",
        "## Public surfaces observed",
        "",
        f"- MLX version: `{result['mlx_surface'].get('mlx_version')}`; default device: `{result['mlx_surface'].get('default_device')}`.",
        f"- `mx.fast.metal_kernel` available: `{result['mlx_surface'].get('has_mx_fast_metal_kernel')}`.",
        f"- MLX memory/cache/wired symbols: `{', '.join(result['mlx_surface'].get('mx_metal_memory_cache_wired_symbols', []))}`.",
        f"- Metal headers present in SDK: `{result['sdk_audit'].get('metal_headers_exist')}`; `xcrun --find metal` ok: `{result['sdk_audit'].get('xcrun_metal', {}).get('ok')}`.",
        f"- Resource/header symbols: `{result['sdk_audit'].get('resource_symbols')}`.",
        "",
        "## Tiny local microbench",
        "",
        f"Shape: rows={mb['shape']['rows']}, dim={mb['shape']['dim']}, {mb['shape']['dtype']}.",
        "",
        "| route | median seconds | max abs error vs MLX sum | rel RMS |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "scalar_one_thread_per_row",
        "simdgroup_simd_sum_one_group_per_row",
        "threadgroup_scratch_reduction_one_group_per_row",
    ):
        t = timings.get(key)
        e = mb.get("errors_vs_mlx_sum", {}).get(key)
        if not t or not e:
            lines.append(f"| {key} | failed | - | - |")
        else:
            lines.append(f"| {key} | {t['median_seconds']:.6g} | {e['max_abs']:.6g} | {e['relative_rms']:.6g} |")
    lines.extend(
        [
            "",
            f"Memory delta: `{mb.get('memory_delta')}`.",
            f"SIMD-group matrix declaration probe: `{result.get('simdgroup_matrix_decl_probe')}`.",
            "",
            "## H3-specific interpretation",
            "",
        ]
    )
    for reason in decision["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "This does not change generation defaults and does not justify a 320x192+ run. The public Apple/MLX mechanisms are useful for bounded pointwise/layout/reduction kernels, but the current H3 blocker is still model residency plus the Sol-Attn exact-block cost model rather than absence of a small public reduction primitive.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = ROOT / "experiments" / f"apple_public_pipeline_probe_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "created_utc": _now(),
        "objective": "Audit public Apple/Metal/MLX mechanisms and run a tiny local microbench before selecting the next MiniMax-H3 optimization direction.",
        "run_dir": str(run_dir.relative_to(ROOT)),
        "mlx_surface": _mlx_surface_audit(),
        "sdk_audit": _sdk_audit(),
    }
    result["simdgroup_matrix_decl_probe"] = _try_simdgroup_matrix_decl()
    result["microbench"] = _microbench(rows=args.rows, dim=args.dim, runs=args.runs, warmups=args.warmups, seed=args.seed)
    result["prior_h3_evidence"] = _prior_h3_evidence()
    audit_bundle = {
        "simdgroup_matrix_decl_probe": result["simdgroup_matrix_decl_probe"],
        "sdk_audit": result["sdk_audit"],
        "mlx_surface": result["mlx_surface"],
    }
    result["decision"] = _decide(audit_bundle, result["microbench"], result["prior_h3_evidence"])
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report_path = ROOT / "research" / "APPLE_PUBLIC_PIPELINE_AUDIT.md"
    _write_markdown(result, report_path)
    result["result_path"] = str(result_path.relative_to(ROOT))
    result["report_path"] = str(report_path.relative_to(ROOT))
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.dim != 128:
        raise SystemExit("This bounded probe is intentionally fixed to the H3 head dimension D=128")
    result = run(args)
    print(json.dumps({"result_path": result["result_path"], "report_path": result["report_path"], "decision": result["decision"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
