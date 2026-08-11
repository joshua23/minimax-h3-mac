"""A converted H3 text checkpoint loads as quantized MLX and runs forward."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import transformers
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen3_vl.config import TextConfig
from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from minimax_h3_mlx.config import TAG_TEXT
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
from quantize_text_encoder import convert


def config() -> TextConfig:
    return TextConfig(
        model_type="qwen3_vl_text",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_theta=5_000_000.0,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [4, 2, 2],
            "mrope_interleaved": True,
        },
    )


def write_source(root: Path, model: Qwen3VLModel) -> None:
    state = {
        f"model.language_model.{key}": value
        for key, value in tree_flatten(model.parameters())
    }
    mx.save_safetensors(str(root / "model.safetensors"), state)
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "model.safetensors" for key in state},
    }))
    text = config()
    raw = {
        "model_type": "qwen3_vl",
        "image_token_id": 201,
        "video_token_id": 202,
        "vision_start_token_id": 203,
        "vision_end_token_id": 204,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rope_theta": 5_000_000.0,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [4, 2, 2],
                "mrope_interleaved": True,
            },
        },
        "vision_config": {
            "model_type": "qwen3_vl",
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [0],
            "hidden_act": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
        },
    }
    (root / "config.json").write_text(json.dumps(raw))


def main() -> None:
    mx.random.seed(0)
    source_model = Qwen3VLModel(config())
    mx.eval(source_model.parameters())
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    expected = source_model(input_ids)
    mx.eval(expected)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, output = root / "source", root / "output"
        tokenizer_assets = root / "tokenizer-assets"
        source.mkdir()
        tokenizer_assets.mkdir()
        write_source(source, source_model)
        convert(source, output, num_layers=1)
        original_quantize = mx.quantize
        mx.quantize = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("quantized checkpoint loading must not quantize random weights")
        )
        try:
            encoder = MiniMaxH3TextEncoder(output, num_layers=1, load_vision=False)
        finally:
            mx.quantize = original_quantize
        actual = encoder.language(input_ids)
        mx.eval(actual)

        seen_tokenizer_paths: list[Path] = []

        class FakeTokenizer:
            def __len__(self):
                return 256

            def __call__(self, prompt, add_special_tokens=False):
                assert prompt == "real prompt"
                assert add_special_tokens is False
                return {"input_ids": [5, 7, 11]}

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, path):
                seen_tokenizer_paths.append(Path(path))
                return FakeTokenizer()

        original_from_pretrained = transformers.AutoTokenizer.from_pretrained
        transformers.AutoTokenizer.from_pretrained = FakeAutoTokenizer.from_pretrained
        try:
            routed = MiniMaxH3TextEncoder(
                output,
                num_layers=1,
                load_vision=False,
                tokenizer_dir=tokenizer_assets,
            )
            routed_ids, routed_tags, _ = routed.build_request("real prompt")
        finally:
            transformers.AutoTokenizer.from_pretrained = original_from_pretrained

        class IncompleteTokenizer(FakeTokenizer):
            def __len__(self):
                return 1

        transformers.AutoTokenizer.from_pretrained = lambda path: IncompleteTokenizer()
        routed._tokenizer = None
        try:
            routed.build_request("real prompt")
        except ValueError as exc:
            incomplete_error = str(exc)
        else:
            raise AssertionError("an incomplete tokenizer vocabulary must fail before embedding")
        finally:
            transformers.AutoTokenizer.from_pretrained = original_from_pretrained

        class EmptyTokenizer(FakeTokenizer):
            def __call__(self, prompt, add_special_tokens=False):
                return {"input_ids": []}

        routed._tokenizer = EmptyTokenizer()
        try:
            routed.build_request("real prompt")
        except ValueError as exc:
            empty_error = str(exc)
        else:
            raise AssertionError("empty tokenization must fail before QuantizedEmbedding")

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    relative_l2 = float(
        mx.sqrt(mx.sum((actual - expected) ** 2))
        / mx.maximum(mx.sqrt(mx.sum(expected**2)), mx.array(1e-12))
    )
    assert relative_l2 < 0.5, relative_l2
    packed = [value for key, value in tree_flatten(encoder.language.parameters()) if key.endswith("weight")]
    assert any(value.dtype == mx.uint32 for value in packed)
    assert seen_tokenizer_paths == [tokenizer_assets]
    assert routed_ids.shape == (1, 3)
    assert routed_tags.tolist() == [TAG_TEXT, TAG_TEXT, TAG_TEXT]
    assert "has only 1 tokens" in incomplete_error
    assert "no input token IDs" in empty_error
    print(f"quantized text forward: shape={actual.shape}, relative_l2={relative_l2:.4f}")


if __name__ == "__main__":
    main()
