"""Unit tests for the paired INT8 projection-fidelity comparison."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_int8_linear_quality import paired_delta_ci


def main() -> int:
    result = paired_delta_ci([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], 1000, 7)
    assert result["mean"] < 0
    assert result["ci95"][1] < 0
    try:
        paired_delta_ci([0.1], [0.1, 0.2], 10, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched paired observations were accepted")
    print("INT8 quality helper tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
