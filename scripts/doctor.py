#!/usr/bin/env python3
"""CLI entry point for the lightweight MiniMax-H3 release doctor."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.doctor import main


if __name__ == "__main__":
    raise SystemExit(main())
