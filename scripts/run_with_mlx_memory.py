#!/usr/bin/env python3
"""Run a Python script in-process and report MLX allocator memory to stderr.

This helper lets orchestration scripts preserve a target script's normal CLI while
adding a stable machine-readable memory line for long MLX steps. It is intended
for repo-local deployment commands such as text-encoder quantization and real
video generation; it does not download or modify model assets by itself.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any


def _exit_code(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return 1


def _memory_snapshot() -> dict[str, int | None]:
    try:
        import mlx.core as mx
    except Exception:
        return {"peak_bytes": None, "active_bytes": None, "cache_bytes": None}

    def call(name: str) -> int | None:
        func = getattr(mx, name, None)
        if func is None and hasattr(mx, "metal"):
            func = getattr(mx.metal, name, None)
        if func is None:
            return None
        try:
            return int(func())
        except Exception:
            return None

    return {
        "peak_bytes": call("get_peak_memory"),
        "active_bytes": call("get_active_memory"),
        "cache_bytes": call("get_cache_memory"),
    }


def _reset_peak_memory() -> None:
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: run_with_mlx_memory.py TARGET_SCRIPT [ARGS...]", file=sys.stderr)
        return 64

    target = Path(argv[0])
    target_args = argv[1:]
    old_argv = sys.argv[:]
    _reset_peak_memory()
    try:
        sys.argv = [str(target), *target_args]
        try:
            runpy.run_path(str(target), run_name="__main__")
            return 0
        except SystemExit as exc:
            return _exit_code(exc.code)
    finally:
        sys.argv = old_argv
        snapshot = _memory_snapshot()
        print(
            "ARGUS_MLX_MEMORY "
            f"peak_bytes={snapshot['peak_bytes']} "
            f"active_bytes={snapshot['active_bytes']} "
            f"cache_bytes={snapshot['cache_bytes']}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
