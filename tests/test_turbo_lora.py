"""Disk-streamed Turbo LoRA must match eager block application exactly."""
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

from minimax_h3_mlx.adaln import ModulationCache
from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.quantize import QuantConfig, quantize_dit
from minimax_h3_mlx.streaming import load_streaming_dit
from minimax_h3_mlx.turbo_lora import TARGETS, TurboLoRAProvider


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
    sequence = n_text + n_video + n_audio
    return (
        mx.random.normal((1, n_video, cfg.video_patch_dim)),
        mx.random.normal((1, n_audio, cfg.audio_latents_dim)),
        mx.random.normal((1, n_text, cfg.text_dim)),
        mx.array([0.0, 0.7]),
        mx.concatenate([
            mx.zeros((n_text,), dtype=mx.int32),
            mx.ones((n_video,), dtype=mx.int32),
            mx.zeros((n_audio,), dtype=mx.int32),
        ]),
        mx.concatenate([
            mx.full((n_text,), TAG_TEXT, dtype=mx.int32),
            mx.full((n_video,), TAG_VIDEO, dtype=mx.int32),
            mx.full((n_audio,), TAG_AUDIO, dtype=mx.int32),
        ]),
        mx.stack([
            mx.arange(sequence) % 3,
            mx.arange(sequence) % 5,
            mx.arange(sequence) % 7,
        ], axis=-1).astype(mx.int32),
        mx.arange(n_text, n_text + n_video),
        mx.arange(n_text + n_video, sequence),
        mx.arange(n_text),
    )


def write_base(root: Path, model: MiniMaxH3DiT, cfg: DiTConfig) -> None:
    state = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(root / "model.safetensors"), state)
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "model.safetensors" for key in state},
    }))
    raw = asdict(cfg)
    raw["patch_size"] = list(cfg.patch_size)
    (root / "config.json").write_text(json.dumps(raw))
    (root / "quant_config.json").write_text(json.dumps({
        "bits": 4,
        "group_size": 64,
        "quantize_adaln": True,
        "adaln_bits": 8,
    }))


def write_lora(path: Path, cfg: DiTConfig, rank: int = 4) -> None:
    shapes = {
        "attn.to_q": (cfg.inner_dim, cfg.hidden_size),
        "attn.to_k": (cfg.inner_dim, cfg.hidden_size),
        "attn.to_v": (cfg.inner_dim, cfg.hidden_size),
        "attn.to_out.0": (cfg.hidden_size, cfg.inner_dim),
        "ff.net.0.proj": (2 * cfg.ffn_hidden_size, cfg.hidden_size),
        "ff.net.2": (cfg.hidden_size, cfg.ffn_hidden_size),
    }
    state = {}
    groups = (
        ("transformer_blocks", cfg.num_layers),
        ("token_refiner.refiner_blocks", cfg.token_refiner_num_layers),
    )
    for family, count in groups:
        for index in range(count):
            for target in TARGETS:
                output_dims, input_dims = shapes[target]
                prefix = f"base_model.model.{family}.{index}.{target}"
                state[f"{prefix}.lora_A.default.weight"] = (
                    mx.random.normal((rank, input_dims)) * 0.01
                ).astype(mx.bfloat16)
                state[f"{prefix}.lora_B.default.weight"] = (
                    mx.random.normal((output_dims, rank)) * 0.01
                ).astype(mx.bfloat16)
    mx.save_safetensors(str(path), state)


class EagerProvider:
    def __init__(self, model: MiniMaxH3DiT, adapter: TurboLoRAProvider):
        self.model = model
        self.adapter = adapter
        self.block_count = len(model.blocks)
        self.current_lora = None
        self.refiner_loras = adapter.load_refiners()

    def load_block(self, index: int, **kwargs):
        self.current_lora = self.adapter.load_block(index)
        return self.model.blocks[index]


def main() -> None:
    mx.random.seed(0)
    cfg = config()
    eager = MiniMaxH3DiT(cfg)
    quantize_dit(
        eager,
        QuantConfig(bits=4, group_size=64, quantize_adaln=True, adaln_bits=8),
    )
    args = inputs(cfg)
    cache = ModulationCache.build(eager, args[3], dtype=mx.float32)
    baseline_v, baseline_a = eager(*args, modulation_cache=cache)
    mx.eval(baseline_v, baseline_a)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_base(root, eager, cfg)
        lora_path = root / "turbo.safetensors"
        write_lora(lora_path, cfg)

        adapter = TurboLoRAProvider(
            lora_path,
            num_blocks=cfg.num_layers,
            num_refiner_blocks=cfg.token_refiner_num_layers,
            hidden_size=cfg.hidden_size,
            inner_dim=cfg.inner_dim,
            ffn_hidden_size=cfg.ffn_hidden_size,
            alpha=8.0,
        )
        assert adapter.rank == 4
        assert adapter.multiplier == 2.0
        eager_provider = EagerProvider(eager, adapter)
        expected_v, expected_a = eager(
            *args,
            modulation_cache=cache,
            block_provider=eager_provider,
        )
        mx.eval(expected_v, expected_a)

        streamed, provider = load_streaming_dit(root, turbo_lora_path=lora_path)
        streamed_cache = ModulationCache.build_streaming(
            streamed,
            provider,
            args[3],
            dtype=mx.float32,
        )
        actual_v, actual_a = streamed(
            *args,
            modulation_cache=streamed_cache,
            block_provider=provider,
        )
        mx.eval(actual_v, actual_a)

    video_delta = float(mx.max(mx.abs(expected_v - actual_v)).item())
    audio_delta = float(mx.max(mx.abs(expected_a - actual_a)).item())
    baseline_delta = float(mx.max(mx.abs(expected_v - baseline_v)).item())
    assert video_delta == 0.0, video_delta
    assert audio_delta == 0.0, audio_delta
    assert baseline_delta > 0.0
    print(
        f"Turbo LoRA streamed exact: video={video_delta}, audio={audio_delta}, "
        f"base_delta={baseline_delta:.6f}"
    )


if __name__ == "__main__":
    main()
