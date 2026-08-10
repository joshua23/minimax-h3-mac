#!/usr/bin/env python3
"""Archived bounded real 4-bit safetensors I/O and page-fault probe.

The default probe touches the local MiniMax-H3 4-bit DiT shards but materializes
only a small, explicit tensor set.  It compares the lossless safetensors
key-selective prototype against the current ``mx.load(shard)``-then-filter
pattern and records resource deltas.  It never downloads weights and never runs
generation.

The former provider-level ``selective_safetensors`` mode is rejected and is not
runnable from active streaming code; historical JSON artifacts preserve that A/B.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.selective_loading import (  # noqa: E402
    build_tensor_plan,
    fingerprint_tensors,
    load_selected_tensors,
    load_weight_map,
)

DEFAULT_KEYS = [
    # One real packed 4-bit DiT projection tensor (~55 MiB) from a multi-GB shard.
    "blocks.0.attn.qkv_proj.weight",
    # Two tiny fp32 static tensors in separate shards, to expose shard-open amplification.
    "audio_patch_proj.bias",
    "time_embedder.proj_in.bias",
]
DEFAULT_OUT = "experiments/apple_h3_4bit_io_probe.json"
KEY_MODES = ("safetensors_selective", "mlx_load")
PROVIDER_MODES = ("provider_mx_load",)
MODES = KEY_MODES + PROVIDER_MODES


def _current_rss_kib() -> int | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(out.strip())
    except Exception:
        return None


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
    }


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(after_value, (int, float)) and isinstance(before_value, (int, float)):
            out[key] = after_value - before_value
    return out


def _fingerprint_mx_array(array: Any) -> dict[str, Any]:
    import numpy as np
    import mlx.core as mx

    if array.dtype == mx.bfloat16:
        raw = np.ascontiguousarray(np.asarray(array.view(mx.uint16)))
        dtype = "bfloat16"
    else:
        raw = np.ascontiguousarray(np.asarray(array))
        dtype = str(raw.dtype)
    return {
        "shape": [int(dim) for dim in array.shape],
        "dtype": dtype,
        "nbytes": int(raw.nbytes),
        "sha256": hashlib.sha256(raw.tobytes(order="C")).hexdigest(),
    }


def _fingerprint_mx_tensors(tensors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: _fingerprint_mx_array(value) for key, value in tensors.items()}


def _block_source_keys(
    model_dir: Path,
    block_index: int,
    *,
    include_adaln: bool = True,
    adaln_only: bool = False,
) -> list[str]:
    weight_map = load_weight_map(model_dir)
    source_prefix = f"blocks.{block_index}."
    keys = [key for key in weight_map if key.startswith(source_prefix)]
    if adaln_only:
        return [key for key in keys if ".adaln_proj." in key]
    if not include_adaln:
        return [key for key in keys if ".adaln_proj." not in key]
    return keys


def _load_with_mx(model_dir: Path, keys: list[str]) -> dict[str, Any]:
    import mlx.core as mx

    weight_map = load_weight_map(model_dir)
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)

    arrays: dict[str, Any] = {}
    shard_events: list[dict[str, Any]] = []
    for shard, shard_keys in by_shard.items():
        started = time.perf_counter()
        loaded = mx.load(str(model_dir / shard))
        open_seconds = time.perf_counter() - started
        materialized: list[dict[str, Any]] = []
        for key in shard_keys:
            t0 = time.perf_counter()
            arrays[key] = loaded[key]
            materialized.append({"key": key, "seconds": time.perf_counter() - t0})
        shard_events.append(
            {
                "shard": shard,
                "returned_tensor_count": len(loaded),
                "selected_tensor_count": len(shard_keys),
                "mx_load_seconds": open_seconds,
                "materialized": materialized,
            }
        )
        del loaded
        gc.collect()
    try:
        mx.synchronize()
        mx.clear_cache()
    except Exception:
        pass
    return {"fingerprints": _fingerprint_mx_tensors({key: arrays[key] for key in keys}), "shard_events": shard_events}


def _load_with_provider(
    model_dir: Path,
    keys: list[str],
    *,
    block_index: int,
    block_load_mode: str,
    include_adaln: bool,
    adaln_only: bool,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from minimax_h3_mlx.streaming import QuantizedBlockProvider

    provider = QuantizedBlockProvider(model_dir, block_load_mode=block_load_mode)
    started = time.perf_counter()
    provider.load_block(block_index, include_adaln=include_adaln, adaln_only=adaln_only)
    block_load_seconds = time.perf_counter() - started
    flat = dict(tree_flatten(provider.slot.parameters()))
    mx.eval(*(flat.values()))

    source_prefix = f"blocks.{block_index}."
    arrays: dict[str, Any] = {}
    for source_key in keys:
        target_key = "blocks.0." + source_key[len(source_prefix) :]
        arrays[source_key] = flat[target_key]
    return {
        "fingerprints": _fingerprint_mx_tensors({key: arrays[key] for key in keys}),
        "shard_events": [
            {
                "block_index": block_index,
                "block_load_mode": block_load_mode,
                "selected_tensor_count": len(keys),
                "provider_load_block_seconds": block_load_seconds,
                "logical_bytes_loaded": provider.logical_bytes_loaded,
            }
        ],
    }


def _child_payload(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir)
    keys = json.loads(args.keys_json)
    before = _metrics()
    started = time.perf_counter()
    if args.mode == "safetensors_selective":
        arrays = load_selected_tensors(model_dir, keys)
        loader = {
            "fingerprints": fingerprint_tensors(arrays),
            "shard_events": [],
        }
    elif args.mode == "mlx_load":
        loader = _load_with_mx(model_dir, keys)
    elif args.mode in PROVIDER_MODES:
        if args.block_index is None:
            raise ValueError("provider modes require --block-index")
        loader = _load_with_provider(
            model_dir,
            keys,
            block_index=args.block_index,
            block_load_mode="mlx",
            include_adaln=args.include_adaln,
            adaln_only=args.adaln_only,
        )
    else:
        raise ValueError(f"unknown child mode {args.mode!r}")
    gc.collect()
    after = _metrics()
    return {
        "mode": args.mode,
        "ok": True,
        "pid": os.getpid(),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics_before": before,
        "metrics_after": after,
        "metrics_delta": _delta(before, after),
        **loader,
    }


def _run_child(mode: str, model_dir: Path, keys: list[str], args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child-mode",
        mode,
        "--model-dir",
        str(model_dir),
        "--keys-json",
        json.dumps(keys),
    ]
    if args.block_index is not None:
        cmd.extend(["--block-index", str(args.block_index)])
    if not args.include_adaln:
        cmd.append("--no-include-adaln")
    if args.adaln_only:
        cmd.append("--adaln-only")
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    record: dict[str, Any] = {
        "mode": mode,
        "command": cmd,
        "process_elapsed_seconds": time.perf_counter() - started,
        "returncode": proc.returncode,
        "stderr": proc.stderr,
    }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        record.update({"ok": False, "stdout": proc.stdout, "error": "child_stdout_not_json"})
        return record
    payload.update(record)
    payload["ok"] = bool(payload.get("ok")) and proc.returncode == 0
    return payload


def _compare_fingerprints(mode_records: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    usable = [record for record in mode_records if record.get("ok") and isinstance(record.get("fingerprints"), dict)]
    if len(usable) < 2:
        return {"ok": False, "reason": "fewer_than_two_successful_modes"}
    reference = usable[0]
    mismatches: list[dict[str, Any]] = []
    for record in usable[1:]:
        for key in keys:
            want = reference["fingerprints"].get(key)
            got = record["fingerprints"].get(key)
            if want != got:
                mismatches.append({"key": key, "reference_mode": reference["mode"], "mode": record["mode"], "reference": want, "observed": got})
    return {
        "ok": not mismatches,
        "reference_mode": reference["mode"],
        "compared_modes": [record["mode"] for record in usable],
        "mismatches": mismatches,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe exact-key safetensors loading against mx.load on local 4-bit DiT shards.")
    parser.add_argument("--model-dir", default="models/MiniMax-H3-MLX-4bit", help="4-bit DiT directory with model.safetensors.index.json")
    parser.add_argument("--key", action="append", default=[], help="tensor key to materialize; repeatable; defaults to a bounded 4-bit probe set")
    parser.add_argument("--max-requested-mib", type=float, default=128.0, help="abort before loading if selected tensor bytes exceed this bound")
    parser.add_argument("--mode", action="append", choices=MODES, default=[], help="loader mode; defaults to key modes, or provider modes when --block-index is set")
    parser.add_argument("--block-index", type=int, default=None, help="derive a real block tensor set and compare provider.load_block modes")
    parser.add_argument("--no-include-adaln", action="store_false", dest="include_adaln", help="with --block-index, exclude AdaLN tensors like the modulation-cache path")
    parser.add_argument("--adaln-only", action="store_true", help="with --block-index, load only AdaLN tensors")
    parser.add_argument("--out", default=DEFAULT_OUT, help="JSON output path")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    parser.add_argument("--_child-mode", dest="child_mode", choices=MODES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--keys-json", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.child_mode:
        args.mode = args.child_mode
        payload = _child_payload(args)
        print(json.dumps(payload, sort_keys=True))
        return 0

    model_dir = Path(args.model_dir)
    if args.block_index is None:
        keys = args.key or list(DEFAULT_KEYS)
        modes = args.mode or list(KEY_MODES)
    else:
        keys = args.key or _block_source_keys(
            model_dir,
            args.block_index,
            include_adaln=args.include_adaln,
            adaln_only=args.adaln_only,
        )
        modes = args.mode or list(PROVIDER_MODES)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        plan = build_tensor_plan(model_dir, keys, cwd=ROOT)
        max_bytes = int(args.max_requested_mib * 1024 * 1024)
        if int(plan["requested_tensor_bytes"]) > max_bytes:
            raise ValueError(
                f"selected tensors request {plan['requested_tensor_bytes']} bytes, above bound {max_bytes}"
            )
        mode_records = [_run_child(mode, model_dir, keys, args) for mode in modes]
        comparison = _compare_fingerprints(mode_records, keys)
        ok = all(record.get("ok") for record in mode_records) and comparison.get("ok") is True
        payload = {
            "schema_version": 1,
            "tool": "probe_selective_loading",
            "ok": ok,
            "exit_code": 0 if ok else 2,
            "started_at_utc": started_at,
            "model_dir": str(model_dir),
            "selected_keys": keys,
            "runs_generation": False,
            "downloads_models": False,
            "bounded_probe": {
                "max_requested_mib": args.max_requested_mib,
                "requested_tensor_bytes": plan["requested_tensor_bytes"],
                "materializes_only_selected_keys": True,
                "drops_os_page_cache": False,
                "page_cache_limitation": "macOS page cache is not dropped; ru_inblock/page-fault deltas are real for this process/order but may be warm-cache lower bounds.",
            },
            "plan": plan,
            "modes": mode_records,
            "lossless_comparison": comparison,
        }
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "tool": "probe_selective_loading",
            "ok": False,
            "exit_code": 2,
            "started_at_utc": started_at,
            "model_dir": str(model_dir),
            "selected_keys": keys,
            "runs_generation": False,
            "downloads_models": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    text = json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    sys.stdout.write(text)
    return int(payload.get("exit_code", 2))


if __name__ == "__main__":
    raise SystemExit(main())
