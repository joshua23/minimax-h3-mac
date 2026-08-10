#!/usr/bin/env python3
"""Build and benchmark a lossless block-local DiT safetensors layout.

This is a provider-level gate only: it rewrites selected real 4-bit block tensors
into project-local block shards, compares ``QuantizedBlockProvider(..., mx.load)``
against the original indexed layout, records fingerprints/resources, and never
runs generation or downloads assets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.block_local_reshard import (  # noqa: E402
    block_tensor_keys,
    create_block_local_layout,
)
from minimax_h3_mlx.selective_loading import load_weight_map  # noqa: E402

PHASES = ("full", "core", "adaln")
DEFAULT_BLOCKS = (0, 24)


def _current_rss_kib() -> int | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(out.strip())
    except Exception:
        return None


def _vm_pageouts() -> int | None:
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return None
    for line in out.splitlines():
        if line.strip().startswith("Pageouts:"):
            value = line.split(":", 1)[1].strip().rstrip(".").replace(".", "")
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _mlx_call(name: str) -> int | None:
    try:
        import mlx.core as mx
    except Exception:
        return None
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
    try:
        import mlx.core as mx
    except Exception:
        return
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
        "vm_pageouts": _vm_pageouts(),
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


def _fingerprint_mx_array(array: Any) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np

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


def _phase_kwargs(phase: str) -> dict[str, bool]:
    if phase == "full":
        return {"include_adaln": True, "adaln_only": False}
    if phase == "core":
        return {"include_adaln": False, "adaln_only": False}
    if phase == "adaln":
        return {"include_adaln": True, "adaln_only": True}
    raise ValueError(f"unknown phase {phase!r}")


def _physical_shard_bytes(model_dir: Path, keys: list[str]) -> dict[str, Any]:
    weight_map = load_weight_map(model_dir)
    shards = sorted({weight_map[key] for key in keys})
    records = []
    total = 0
    for shard in shards:
        size = (model_dir / shard).stat().st_size
        total += size
        records.append({"shard": shard, "size_bytes": size})
    return {"shard_count": len(shards), "total_bytes": total, "shards": records}


def _child_payload(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from minimax_h3_mlx.streaming import QuantizedBlockProvider

    model_dir = Path(args.model_dir)
    keys = json.loads(args.keys_json)
    phase = args.phase[0] if isinstance(args.phase, list) else args.phase
    block_index = args.block_index[0] if isinstance(args.block_index, list) else args.block_index
    phase_kwargs = _phase_kwargs(phase)
    _reset_mlx_peak()
    before = _metrics()
    started = time.perf_counter()
    provider = QuantizedBlockProvider(model_dir, block_load_mode="mlx")
    load_started = time.perf_counter()
    provider.load_block(block_index, **phase_kwargs)
    provider_load_block_seconds = time.perf_counter() - load_started
    flat = dict(tree_flatten(provider.slot.parameters()))
    mx.eval(*(flat.values()))
    source_prefix = f"blocks.{block_index}."
    arrays: dict[str, Any] = {}
    for source_key in keys:
        target_key = "blocks.0." + source_key[len(source_prefix) :]
        arrays[source_key] = flat[target_key]
    mx.synchronize()
    fingerprints = {key: _fingerprint_mx_array(arrays[key]) for key in keys}
    gc.collect()
    after = _metrics()
    return {
        "label": args.label,
        "ok": True,
        "pid": os.getpid(),
        "model_dir": str(model_dir),
        "block_index": block_index,
        "phase": phase,
        "key_count": len(keys),
        "logical_bytes_loaded": provider.logical_bytes_loaded,
        "physical_shards": _physical_shard_bytes(model_dir, keys),
        "provider_load_block_seconds": provider_load_block_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics_before": before,
        "metrics_after": after,
        "metrics_delta": _delta(before, after),
        "fingerprints": fingerprints,
    }


def _run_child(label: str, model_dir: Path, block_index: int, phase: str, keys: list[str]) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        "--label",
        label,
        "--model-dir",
        str(model_dir),
        "--block-index",
        str(block_index),
        "--phase",
        phase,
        "--keys-json",
        json.dumps(keys),
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    record: dict[str, Any] = {
        "label": label,
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


def _compare_pair(original: dict[str, Any], block_local: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    mismatches = []
    if not original.get("ok") or not block_local.get("ok"):
        return {
            "ok": False,
            "reason": "child_load_failed",
            "original_ok": original.get("ok"),
            "block_local_ok": block_local.get("ok"),
            "mismatches": mismatches,
        }
    for key in keys:
        want = original.get("fingerprints", {}).get(key)
        got = block_local.get("fingerprints", {}).get(key)
        if want != got:
            mismatches.append({"key": key, "original": want, "block_local": got})
    return {"ok": not mismatches, "mismatches": mismatches}


def _numeric(record: dict[str, Any], path: tuple[str, ...]) -> int | float | None:
    value: Any = record
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value if isinstance(value, (int, float)) else None


def _provider_gate(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    for item in comparisons:
        ident = f"block {item['block_index']} phase {item['phase']}"
        if not item["fingerprints_match"]:
            reasons.append(f"{ident}: fingerprint mismatch")
            continue
        original = item["original"]
        block_local = item["block_local"]
        original_elapsed = _numeric(original, ("process_elapsed_seconds",))
        block_elapsed = _numeric(block_local, ("process_elapsed_seconds",))
        if original_elapsed is None or block_elapsed is None or block_elapsed >= original_elapsed:
            reasons.append(f"{ident}: block-local process elapsed {block_elapsed} is not lower than original {original_elapsed}")
        original_rss = _numeric(original, ("metrics_delta", "current_rss_kib"))
        block_rss = _numeric(block_local, ("metrics_delta", "current_rss_kib"))
        if original_rss is not None and block_rss is not None and block_rss > original_rss:
            reasons.append(f"{ident}: block-local RSS delta {block_rss} KiB exceeds original {original_rss} KiB")
        original_maxrss = _numeric(original, ("metrics_delta", "ru_maxrss_raw"))
        block_maxrss = _numeric(block_local, ("metrics_delta", "ru_maxrss_raw"))
        if original_maxrss is not None and block_maxrss is not None and block_maxrss > original_maxrss:
            reasons.append(f"{ident}: block-local maxrss delta {block_maxrss} exceeds original {original_maxrss}")
        original_pageouts = _numeric(original, ("metrics_delta", "vm_pageouts"))
        block_pageouts = _numeric(block_local, ("metrics_delta", "vm_pageouts"))
        if original_pageouts is not None and block_pageouts is not None and block_pageouts > original_pageouts:
            reasons.append(f"{ident}: block-local pageouts delta {block_pageouts} exceeds original {original_pageouts}")
    return {
        "passed": not reasons,
        "rule": "all block/phase pairs must have exact fingerprints, lower block-local process elapsed, and no worse available RSS/maxrss/page-outs",
        "reasons": reasons,
        "candidate_e2e_allowed_next": not reasons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/MiniMax-H3-MLX-4bit", help="4-bit DiT transformer directory")
    parser.add_argument("--block-index", type=int, action="append", default=[], help="block index to benchmark; repeatable; defaults to 0 and 24")
    parser.add_argument("--phase", choices=PHASES, action="append", default=[], help="load phase to benchmark; defaults to full and core")
    parser.add_argument("--work-dir", default=None, help="project-local output directory for the derived partial layout and report")
    parser.add_argument("--out", default=None, help="JSON report path; defaults inside --work-dir")
    parser.add_argument("--reserve-gib", type=float, default=100.0, help="abort if free disk before reshard is below this reserve")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--keys-json", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args._child:
        if args.label is None or args.keys_json is None:
            raise SystemExit("child mode requires --label and --keys-json")
        payload = _child_payload(args)
        print(json.dumps(payload, sort_keys=True))
        return 0

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    model_dir = Path(args.model_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    work_dir = Path(args.work_dir) if args.work_dir else Path("experiments") / f"block_local_reshard_{stamp}"
    out = Path(args.out) if args.out else work_dir / "provider_ab.json"
    blocks = tuple(args.block_index or DEFAULT_BLOCKS)
    phases = tuple(args.phase or ("full", "core"))

    payload: dict[str, Any]
    try:
        free_bytes = shutil.disk_usage(Path.cwd()).free
        reserve_bytes = int(args.reserve_gib * 1024**3)
        if free_bytes < reserve_bytes:
            raise RuntimeError(
                f"free disk {free_bytes / 1024**3:.1f} GiB is below requested reserve {args.reserve_gib:.1f} GiB"
            )
        weight_map = load_weight_map(model_dir)
        layout_dir = work_dir / "transformer-block-local-partial"
        layout = create_block_local_layout(model_dir, layout_dir, block_indices=blocks, include_static=False)
        comparisons: list[dict[str, Any]] = []
        for block_index in blocks:
            for phase in phases:
                phase_kwargs = _phase_kwargs(phase)
                keys = list(block_tensor_keys(weight_map, block_index, **phase_kwargs))
                if not keys:
                    raise KeyError(f"no keys selected for block {block_index} phase {phase}")
                original = _run_child("original_mx_load", model_dir, block_index, phase, keys)
                block_local = _run_child("block_local_mx_load", layout_dir, block_index, phase, keys)
                comparison = _compare_pair(original, block_local, keys)
                comparisons.append(
                    {
                        "block_index": block_index,
                        "phase": phase,
                        "key_count": len(keys),
                        "fingerprints_match": comparison.get("ok") is True,
                        "fingerprint_comparison": comparison,
                        "original": original,
                        "block_local": block_local,
                    }
                )
        gate = _provider_gate(comparisons)
        payload = {
            "schema_version": 1,
            "tool": "probe_block_local_reshard",
            "ok": all(item["fingerprints_match"] for item in comparisons),
            "started_at_utc": started_at,
            "runs_generation": False,
            "downloads_models": False,
            "model_dir": str(model_dir),
            "work_dir": str(work_dir),
            "layout_dir": str(layout_dir),
            "selected_blocks": list(blocks),
            "phases": list(phases),
            "disk_free_before_bytes": free_bytes,
            "reserve_gib": args.reserve_gib,
            "reshard_layout": layout.to_dict(),
            "comparisons": comparisons,
            "provider_gate": gate,
        }
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "tool": "probe_block_local_reshard",
            "ok": False,
            "started_at_utc": started_at,
            "runs_generation": False,
            "downloads_models": False,
            "model_dir": str(model_dir),
            "work_dir": str(work_dir),
            "selected_blocks": list(blocks),
            "phases": list(phases),
            "error": f"{type(exc).__name__}: {exc}",
            "provider_gate": {"passed": False, "candidate_e2e_allowed_next": False},
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    out.write_text(text)
    sys.stdout.write(text)
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
