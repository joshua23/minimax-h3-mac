"""Parity of the streamed-with-vision text encoder against the resident one.

The low-memory path gained image support by streaming the BF16 vision tower in for one encode
call, materializing its outputs (patch embeddings + deepstack embeds), releasing the tower, and
splicing the vision rows into the streamed embedding before the decoder layers stream. None of
that can be compared against torch here; what this checks is that the streamed *with vision*
forward is numerically the same request as the resident forward on identical weights.

A tiny random-weight model is written out in the release's own on-disk shape (``config.json`` with
``text_config``/``vision_config`` plus ``model.language_model.*`` and ``model.visual.*``
safetensors). The tokenizer and processor come from the real FL2VA checkpoint so the request
presentation — ``"<Picture 1>: "`` labels, vision token ids, the processor's patch grids — is the
production one, while the weights stay tiny.

    ./.venv/bin/python tests/test_text_encoder_stream_vision.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

FAILURES: list[str] = []

HIDDEN = 64
HEAD_DIM = 16
NUM_LAYERS = 6
READ_LAYER = 4  # stand-in for the release's 50-of-64
VOCAB = 152_000  # must cover the release's special-token ids (image token 151655)

# The tokenizer/processor ship with the real checkpoint; the tiny weights never see raw token ids
# beyond what these produce.
TOKENIZER_DIR = ROOT / "models" / "MiniMax-H3" / "FL2VA" / "tokenizer"
PROCESSOR_DIR = ROOT / "models" / "MiniMax-H3" / "FL2VA" / "processor"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def build_checkpoint(path: Path) -> None:
    """Write a tiny random-weight release layout with both towers, in MLX only."""
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen3_vl.config import TextConfig, VisionConfig
    from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel
    from mlx_vlm.models.qwen3_vl.vision import VisionModel

    text_block = {
        "model_type": "qwen3_vl_text",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 128,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "rope_theta": 5000000.0,
        "rope_scaling": {
            "rope_type": "default",
            # Must sum to head_dim / 2, as [24, 20, 20] does for the release's head_dim 128.
            "mrope_section": [4, 2, 2],
            "mrope_interleaved": True,
        },
    }
    vision_block = {
        "model_type": "qwen3_vl",
        "depth": 4,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "in_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": HIDDEN,
        "num_position_embeddings": 64,
        # Two deepstack levels, both inside the read window: injected after language layers 0, 1.
        "deepstack_visual_indexes": [0, 2],
        "hidden_act": "gelu_pytorch_tanh",
        "initializer_range": 0.02,
    }
    config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "tie_word_embeddings": False,
        "text_config": text_block,
        "vision_config": vision_block,
    }
    (path / "config.json").write_text(json.dumps(config))

    state: dict[str, mx.array] = {}
    language = Qwen3VLModel(TextConfig.from_dict(text_block))
    for key, tensor in tree_flatten(language.parameters()):
        state[f"model.language_model.{key}"] = tensor.astype(mx.float32)
    vision = VisionModel(VisionConfig.from_dict(vision_block))
    for key, tensor in tree_flatten(vision.parameters()):
        state[f"model.visual.{key}"] = tensor.astype(mx.float32)
    # The release also carries a head the conditioner never evaluates; include it so the loader is
    # seen to skip it.
    state["lm_head.weight"] = mx.zeros((VOCAB, HIDDEN))

    mx.save_safetensors(str(path / "model.safetensors"), state)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in state}})
    )


def max_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def main() -> int:
    if not (TOKENIZER_DIR / "tokenizer_config.json").is_file():
        print("skip  FL2VA tokenizer/processor not present; nothing to compare against")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "text_encoder"
        model_dir.mkdir()
        build_checkpoint(model_dir)

        def build(stream_layers: bool) -> MiniMaxH3TextEncoder:
            return MiniMaxH3TextEncoder(
                model_dir,
                num_layers=READ_LAYER,
                load_vision=True,
                tokenizer_dir=TOKENIZER_DIR,
                processor_dir=PROCESSOR_DIR,
                stream_layers=stream_layers,
            )

        resident = build(stream_layers=False)
        streamed = build(stream_layers=True)

        rng = np.random.default_rng(7)
        image = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB")
        prompt = "a red fox"

        # -- with a keyframe image ------------------------------------------------------------
        ref_hidden, ref_tags = resident.encode(prompt, [image])
        stream_hidden, stream_tags = streamed.encode(prompt, [image])

        check("streamed vision released the tower", streamed.vision is None)
        check(
            "token tags match",
            np.array_equal(np.asarray(ref_tags), np.asarray(stream_tags)),
        )
        check("hidden shape matches", ref_hidden.shape == stream_hidden.shape,
              f"{ref_hidden.shape} vs {stream_hidden.shape}")

        scale = float(mx.max(mx.abs(ref_hidden.astype(mx.float32))))
        diff = max_abs_diff(ref_hidden, stream_hidden)
        # bf16 compute on identical weights: the two paths run the same ops in the same order, so
        # the tolerance covers rounding, not semantics.
        check("streamed-with-image parity", diff <= 2e-2 * max(scale, 1e-6),
              f"max_abs {diff:.3e} vs scale {scale:.3e}")

        # -- text-only regression -------------------------------------------------------------
        ref_text, _ = resident.encode(prompt)
        stream_text, _ = streamed.encode(prompt)
        text_diff = max_abs_diff(ref_text, stream_text)
        text_scale = float(mx.max(mx.abs(ref_text.astype(mx.float32))))
        check("streamed text-only parity", text_diff <= 2e-2 * max(text_scale, 1e-6),
              f"max_abs {text_diff:.3e} vs scale {text_scale:.3e}")

        # -- the released tower can encode again after release --------------------------------
        again_hidden, _ = resident.encode(prompt, [image])
        check("resident encoder reusable", max_abs_diff(ref_hidden, again_hidden) == 0.0)

        # -- an interleaved image + video reference request -----------------------------------
        # The tower emits image rows then video rows, but the vision blocks appear in the sequence
        # in request order — this is the case that exercises the feature-row reorder.
        from minimax_h3_mlx.ref2va import preprocess_reference_video

        frames = rng.integers(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        video_pixels, video_grid = preprocess_reference_video(frames)
        video_pad = streamed.video_token_id
        label_ids = streamed.tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
        video_label_ids = streamed.tokenizer("<Video 1>: ", add_special_tokens=False)["input_ids"]
        n_video_tokens = int(video_grid[0].prod()) // (streamed.merge_size**2)
        image_vision = streamed.processor.image_processor(images=[image], return_tensors="np")
        image_pixels = np.asarray(image_vision["pixel_values"])
        image_grid = np.asarray(image_vision["image_grid_thw"])
        n_image_tokens = int(image_grid[0].prod()) // (streamed.merge_size**2)
        ids = (
            label_ids
            + [streamed.vision_start_token_id]
            + [streamed.image_token_id] * n_image_tokens
            + [streamed.vision_end_token_id]
            + video_label_ids
            + [streamed.vision_start_token_id]
            + [video_pad] * n_video_tokens
            + [streamed.vision_end_token_id]
            + streamed.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        input_ids = mx.array(np.array([ids], dtype=np.int32))
        ref_hidden, ref_tags = resident.encode_presentation(
            input_ids, np.zeros(len(ids), dtype=np.int64),
            np.asarray(image_pixels), image_grid, video_pixels, video_grid,
        )
        stream_hidden, stream_tags = streamed.encode_presentation(
            input_ids, np.zeros(len(ids), dtype=np.int64),
            np.asarray(image_pixels), image_grid, video_pixels, video_grid,
        )
        mixed_scale = float(mx.max(mx.abs(ref_hidden.astype(mx.float32))))
        mixed_diff = max_abs_diff(ref_hidden, stream_hidden)
        check(
            "streamed image+video parity",
            ref_hidden.shape == stream_hidden.shape and mixed_diff <= 2e-2 * max(mixed_scale, 1e-6),
            f"max_abs {mixed_diff:.3e} vs scale {mixed_scale:.3e}",
        )

    print()
    if FAILURES:
        print(f"SOME CHECKS FAILED: {FAILURES}")
        return 1
    print("all streamed-vision checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
