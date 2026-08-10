#!/usr/bin/env python3
"""CLI entry point for the MiniMax-H3 local asset preflight."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.asset_preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
