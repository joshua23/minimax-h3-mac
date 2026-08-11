"""Header-only tests for scripts/audit_quant.py; no MLX runtime is required."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_quant import audit


def write_artifact(root: Path, weight_dtype: str = "U32") -> None:
    specs = {
        "blocks.0.attn.qkv_proj.weight": {
            "dtype": weight_dtype,
            "shape": [16, 16],
            "data_offsets": [0, 1024],
        },
        "blocks.0.attn.qkv_proj.scales": {
            "dtype": "BF16",
            "shape": [16, 2],
            "data_offsets": [1024, 1088],
        },
        "blocks.0.attn.qkv_proj.biases": {
            "dtype": "BF16",
            "shape": [16, 2],
            "data_offsets": [1088, 1152],
        },
    }
    header = json.dumps(specs, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 8)
    shard = root / "model-00001-of-00001.safetensors"
    with shard.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.write(b"\0" * 1152)
    weight_map = {key: shard.name for key in specs}
    (root / "config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / "quant_config.json").write_text(
        json.dumps(
            {
                "bits": 8,
                "group_size": 32,
                "mode": "affine",
                "quantized_layers": {"8": 1},
            }
        )
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_artifact(root)
        report = audit(root)
        assert report["quantized_layer_count"] == 1
        assert report["format"]["packed_weight_dtype"] == "U32"
        assert report["layers"][0]["scales"]["shape"] == [16, 2]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_artifact(root, weight_dtype="F32")
        try:
            audit(root)
        except ValueError as exc:
            assert "packed weight dtype" in str(exc)
        else:
            raise AssertionError("audit accepted a non-U32 packed weight")
    print("quant audit tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
