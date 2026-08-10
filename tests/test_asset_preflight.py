"""Focused tests for the header-only MiniMax-H3 asset preflight.

Run with:
    ./.venv/bin/python tests/test_asset_preflight.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.asset_preflight import ASSET_FAILURE, OK, run_preflight


DIT_CONFIG = {
    "_class_name": "MiniMaxH3DiTModel",
    "hidden_size": 5376,
    "num_layers": 50,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "text_dim": 5120,
    "patch_size": [1, 2, 2],
    "token_refiner_num_layers": 2,
    "ffn_hidden_size": 14336,
    "timestep_input_dim": 256,
    "time_embed_hidden_size": 5376,
    "time_embed_dim": 2688,
    "adaln_out_features": 96768,
    "final_adaln_out_features": 10752,
    "rope_inv_freq_len": 16,
}


def write_indexed_dit(root: Path, *, shard: str = "ok", hidden_size: int = 5376) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = dict(DIT_CONFIG)
    config["hidden_size"] = hidden_size
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 4},
        "weight_map": {"video_patch_proj.weight": "model-00001-of-00001.safetensors"},
    }))
    shard_path = root / "model-00001-of-00001.safetensors"
    if shard == "ok":
        save_file({"video_patch_proj.weight": np.zeros((1,), dtype=np.float32)}, str(shard_path))
    elif shard == "empty":
        shard_path.write_bytes(b"")
    elif shard == "bad_header":
        shard_path.write_bytes(b"not a safetensors file")
    elif shard == "missing":
        pass
    else:
        raise ValueError(shard)
    return root


def issue_codes(result: dict) -> set[str]:
    return {issue["code"] for issue in result["issues"]}


def assert_case(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} failed{(': ' + detail) if detail else ''}")
    print(f"ok  {name}{(' — ' + detail) if detail else ''}")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)

        good = write_indexed_dit(base / "good")
        result = run_preflight([good], cwd=ROOT)
        assert_case("valid indexed safetensors passes", result["exit_code"] == OK and result["ok"])
        assert_case("valid shard header counted", result["indexes"][0]["shards"][0]["tensor_count"] == 1)

        missing = write_indexed_dit(base / "missing", shard="missing")
        result = run_preflight([missing], cwd=ROOT)
        assert_case("missing shard is an asset failure", result["exit_code"] == ASSET_FAILURE and not result["ok"])
        assert_case("missing shard has stable issue code", "missing_shard_safetensors" in issue_codes(result))

        empty = write_indexed_dit(base / "empty", shard="empty")
        result = run_preflight([empty], cwd=ROOT)
        assert_case("empty shard is an asset failure", result["exit_code"] == ASSET_FAILURE)
        assert_case("empty shard has stable issue code", "empty_shard_safetensors" in issue_codes(result))

        corrupt = write_indexed_dit(base / "corrupt", shard="bad_header")
        result = run_preflight([corrupt], cwd=ROOT)
        assert_case("bad header is an asset failure", result["exit_code"] == ASSET_FAILURE)
        assert_case("bad header has stable issue code", "bad_shard_safetensors_header" in issue_codes(result))

        wrong_family = write_indexed_dit(base / "wrong-family", hidden_size=1)
        result = run_preflight([wrong_family], cwd=ROOT)
        assert_case("wrong MiniMax-H3 DiT family is an asset failure", result["exit_code"] == ASSET_FAILURE)
        assert_case("wrong family has stable issue code", "config_family_mismatch" in issue_codes(result))

        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "preflight_assets.py"), str(good)],
            check=False,
            text=True,
            capture_output=True,
        )
        payload = json.loads(proc.stdout)
        assert_case("CLI emits machine-readable JSON", payload["ok"] is True and payload["exit_code"] == OK)
        assert_case("CLI success exit code matches JSON", proc.returncode == payload["exit_code"] == OK)

        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "preflight_assets.py"), str(missing)],
            check=False,
            text=True,
            capture_output=True,
        )
        payload = json.loads(proc.stdout)
        assert_case("CLI failure exit code matches JSON", proc.returncode == payload["exit_code"] == ASSET_FAILURE)
        assert_case("CLI failure names missing asset", "missing_shard_safetensors" in issue_codes(payload))

    print("asset preflight focused tests passed")


if __name__ == "__main__":
    main()
