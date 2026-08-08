"""The text quantizer streams selected layers and preserves checkpoint shape."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from quantize_text_encoder import convert


def main() -> None:
    mx.random.seed(0)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, output = root / "source", root / "output"
        source.mkdir()
        state = {
            "model.language_model.embed_tokens.weight": mx.random.normal((128, 64)),
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.random.normal((64, 64)),
            "model.language_model.layers.0.input_layernorm.weight": mx.ones((64,)),
            "model.language_model.layers.1.self_attn.q_proj.weight": mx.random.normal((64, 64)),
            "model.language_model.norm.weight": mx.ones((64,)),
            "model.visual.blocks.0.weight": mx.random.normal((64, 64)),
            "lm_head.weight": mx.random.normal((128, 64)),
        }
        mx.save_safetensors(str(source / "model.safetensors"), state)
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "model.safetensors" for key in state},
        }))
        (source / "config.json").write_text('{"model_type":"qwen3_vl"}')

        result = convert(source, output, num_layers=1)
        converted = mx.load(str(output / "model.safetensors"))
        assert "model.language_model.layers.0.self_attn.q_proj.scales" in converted
        assert "model.language_model.layers.0.self_attn.q_proj.biases" in converted
        assert "model.language_model.layers.1.self_attn.q_proj.weight" not in converted
        assert "model.visual.blocks.0.weight" not in converted
        assert "lm_head.weight" not in converted
        assert converted["model.language_model.layers.0.input_layernorm.weight"].dtype == mx.float32
        assert result["compression"] > 2.0
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()