"""Canonical BF16 Turbo adapters stream one component at a time with mixed ranks."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.native_turbo_lora import NativeTurboLoRAProvider


def pair(prefix: str, input_dims: int, output_dims: int, rank: int):
    return {
        f"{prefix}.lora_A.weight": mx.random.normal((rank, input_dims)).astype(mx.bfloat16),
        f"{prefix}.lora_B.weight": mx.random.normal((output_dims, rank)).astype(mx.bfloat16),
    }


def main() -> None:
    mx.random.seed(7)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        shards = root / "shards"
        shards.mkdir()
        groups = {
            ("block", 0): {
                **pair("blocks.0.attn.qkv_proj", 8, 12, 4),
                **pair("blocks.0.attn.out_proj", 4, 8, 4),
                **pair("blocks.0.mlp.fc1", 8, 16, 4),
                **pair("blocks.0.mlp.fc2", 8, 8, 4),
                **pair("blocks.0.adaln_proj.linear", 6, 24, 2),
            },
            ("refiner", 0): {
                **pair("token_refiner.blocks.0.attn.qkv_proj", 8, 12, 4),
                **pair("token_refiner.blocks.0.attn.out_proj", 4, 8, 4),
                **pair("token_refiner.blocks.0.mlp.fc1", 8, 16, 4),
                **pair("token_refiner.blocks.0.mlp.fc2", 8, 8, 4),
            },
            ("final", 0): pair("final_layer.adaln_proj.linear", 6, 16, 2),
        }
        weight_map = {}
        mapping = []
        for (family, index), arrays in groups.items():
            relative = f"shards/adapter-{family}-{index:03d}.safetensors"
            mx.save_safetensors(str(root / relative), arrays)
            for key, value in arrays.items():
                weight_map[key] = relative
                side = "A" if ".lora_A." in key else "B"
                mapping.append({
                    "mlx_key": key,
                    "shape": list(value.shape),
                    "dtype": "BF16",
                    "rank": int(value.shape[0] if side == "A" else value.shape[-1]),
                })
        (root / "adapter.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (root / "conversion_manifest.json").write_text(json.dumps({
            "format": "minimax-h3-mlx-native-turbo-lora-v1",
            "contract": {
                "alpha_policy": "alpha=rank",
                "runtime_multiplier": 1.0,
                "all_source_tensors_consumed": True,
                "post_write_tensor_values_exact": True,
            },
            "mapping": mapping,
        }))

        provider = NativeTurboLoRAProvider(root, num_blocks=1, num_refiner_blocks=1)
        block = provider.load_block(0)
        refiners = provider.load_refiners()
        final = provider.load_final()
        assert block.multiplier == 1.0
        assert block.tensors["attn.qkv_proj"][0].shape[0] == 4
        assert block.tensors["adaln_proj.linear"][0].shape[0] == 2
        assert len(refiners) == 1
        assert final.tensors["final_layer.adaln_proj.linear"][0].shape[0] == 2
        result = block.apply("attn.qkv_proj", mx.ones((1, 2, 8), dtype=mx.bfloat16))
        mx.eval(result)
        assert result.shape == (1, 2, 12)
        assert bool(mx.all(mx.isfinite(result)).item())
    print("native Turbo LoRA provider: mixed-rank component streaming PASS")


if __name__ == "__main__":
    main()
