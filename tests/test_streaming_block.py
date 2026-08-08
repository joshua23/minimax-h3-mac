"""Eager and one-block-at-a-time quantized DiT must agree exactly."""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
from minimax_h3_mlx.adaln import ModulationCache
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.quantize import QuantConfig, quantize_dit
from minimax_h3_mlx.streaming import load_streaming_dit


def config() -> DiTConfig:
    return DiTConfig(
        hidden_size=256,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=64,
        ffn_hidden_size=128,
        latents_dim=4,
        audio_latents_dim=8,
        text_dim=128,
        timestep_input_dim=16,
        time_embed_hidden_size=256,
        time_embed_dim=64,
        adaln_out_features=6 * 3 * 256,
        final_adaln_out_features=2 * 256,
        rope_inv_freq_len=4,
    )


def inputs(cfg: DiTConfig):
    n_text, n_video, n_audio = 5, 9, 3
    seq = n_text + n_video + n_audio
    text_i = mx.arange(n_text)
    video_i = mx.arange(n_text, n_text + n_video)
    audio_i = mx.arange(n_text + n_video, seq)
    tags = mx.concatenate([
        mx.full((n_text,), TAG_TEXT, dtype=mx.int32),
        mx.full((n_video,), TAG_VIDEO, dtype=mx.int32),
        mx.full((n_audio,), TAG_AUDIO, dtype=mx.int32),
    ])
    timestep_indices = mx.concatenate([
        mx.zeros((n_text,), dtype=mx.int32),
        mx.ones((n_video,), dtype=mx.int32),
        mx.zeros((n_audio,), dtype=mx.int32),
    ])
    positions = mx.stack(
        [mx.arange(seq) % 3, mx.arange(seq) % 5, mx.arange(seq) % 7],
        axis=-1,
    ).astype(mx.int32)
    return (
        mx.random.normal((1, n_video, cfg.video_patch_dim)),
        mx.random.normal((1, n_audio, cfg.audio_latents_dim)),
        mx.random.normal((1, n_text, cfg.text_dim)),
        mx.array([0.0, 0.7]),
        timestep_indices,
        tags,
        positions,
        video_i,
        audio_i,
        text_i,
    )


def write_checkpoint(root: Path, model: MiniMaxH3DiT, cfg: DiTConfig) -> None:
    state = dict(tree_flatten(model.parameters()))
    shard = "model.safetensors"
    mx.save_safetensors(str(root / shard), state)
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: shard for key in state},
    }))
    raw_config = asdict(cfg)
    raw_config["patch_size"] = list(cfg.patch_size)
    (root / "config.json").write_text(json.dumps(raw_config))
    (root / "quant_config.json").write_text(json.dumps({
        "bits": 4,
        "group_size": 64,
        "quantize_adaln": True,
        "adaln_bits": 8,
    }))


def main() -> None:
    mx.random.seed(0)
    cfg = config()
    eager = MiniMaxH3DiT(cfg)
    quantize_dit(
        eager,
        QuantConfig(bits=4, group_size=64, quantize_adaln=True, adaln_bits=8),
    )
    args = inputs(cfg)
    want_v, want_a = eager(*args)
    mx.eval(want_v, want_a)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_checkpoint(root, eager, cfg)
        streamed, provider = load_streaming_dit(root)
        cache = ModulationCache.build_streaming(streamed, provider, args[3], dtype=mx.float32)
        got_v, got_a = streamed(
            *args,
            modulation_cache=cache,
            block_provider=provider,
        )
        mx.eval(got_v, got_a)

    video_delta = float(mx.max(mx.abs(want_v - got_v)).item())
    audio_delta = float(mx.max(mx.abs(want_a - got_a)).item())
    assert video_delta == 0.0, video_delta
    assert audio_delta == 0.0, audio_delta
    assert provider.current_index == cfg.num_layers - 1
    print(
        f"streaming block exact: video={video_delta} audio={audio_delta}; "
        f"logical bytes loaded={provider.logical_bytes_loaded}"
    )


if __name__ == "__main__":
    main()