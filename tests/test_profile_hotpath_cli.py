"""Parser-only checks for the DiT hotpath profiler cleanup surface.

These tests do not load model weights.  They keep rejected/tiny-only route names
out of the public --candidate choices while preserving only the default baseline
and retained individual real-block dense-dequant helper names.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_profile_module():
    path = ROOT / "scripts" / "profile_dit_block_hotpath.py"
    spec = importlib.util.spec_from_file_location("profile_dit_block_hotpath_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_choices(module) -> set[str]:
    # argparse stores choices on the action; using parse_args alone would not
    # catch accidentally re-exposed names that happen not to be exercised below.
    return set(module.ACTIVE_HOTPATH_CANDIDATES)


def test_profile_candidate_surface_preserves_only_default_and_retained_dense_helpers() -> None:
    module = load_profile_module()
    choices = _candidate_choices(module)

    retained = {
        "none",
        "attention_qkv_tiled_dense_dequant",
        "ffn_fc2_tiled_dense_dequant",
        "attention_out_dense_dequant",
        "attention_out_tiled_dense_dequant",
    }
    assert choices == retained

    quarantined = set(module.QUARANTINED_HOTPATH_CANDIDATES)
    archived = set(module.ARCHIVED_HOTPATH_CANDIDATES)
    assert {"ffn_hidden_tile_stream", "promoted_dense_dequant_combo"} <= quarantined
    assert quarantined <= archived
    assert archived.isdisjoint(choices)
    for name in sorted(archived):
        try:
            module.parse_args(["--candidate", name])
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover - failure path
            raise AssertionError(f"archived candidate {name!r} remained parseable")

    args = module.parse_args(["--candidate", "attention_qkv_tiled_dense_dequant"])
    assert args.candidate == "attention_qkv_tiled_dense_dequant"


def test_profile_help_does_not_leak_archived_candidate_names() -> None:
    module = load_profile_module()
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        try:
            module.parse_args(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
        else:  # pragma: no cover - argparse always exits for --help
            raise AssertionError("--help did not exit")
    help_text = stdout.getvalue()
    for name in module.ACTIVE_HOTPATH_CANDIDATES:
        assert name in help_text
    for name in module.ARCHIVED_HOTPATH_CANDIDATES:
        assert name not in help_text


def main() -> int:
    test_profile_candidate_surface_preserves_only_default_and_retained_dense_helpers()
    test_profile_help_does_not_leak_archived_candidate_names()
    print("profile hotpath CLI cleanup tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
