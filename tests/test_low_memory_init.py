"""Low-memory construction must allocate no model components."""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline, detach_bfloat16


def write_metadata(root: Path) -> tuple[Path, Path]:
    (root / "model_index.json").write_text("{}")
    text = root / "text-4bit"
    dit = root / "dit-4bit"
    text.mkdir()
    dit.mkdir()
    (text / "quant_config.json").write_text('{"bits":4,"group_size":64}')
    cfg = asdict(DiTConfig(num_layers=2))
    cfg["patch_size"] = list(cfg["patch_size"])
    (dit / "config.json").write_text(json.dumps(cfg))
    (dit / "quant_config.json").write_text('{"bits":4,"group_size":64}')

    video = root / "video_vae"
    (video / "source").mkdir(parents=True)
    (video / "config.json").write_text(json.dumps({
        "vae_clip_length": 17,
        "vae_token_drop": 3,
        "latents_mean": [0.0] * 24,
        "latents_std": [1.0] * 24,
    }))
    (video / "source" / "config.json").write_text(json.dumps({
        "ch": 32,
        "ch_mult": [1, 2],
        "in_channels": 3,
        "out_ch": 3,
        "z_channels": 24,
        "num_res_blocks": 1,
        "space_down": [2, 2],
        "time_down": [1, 2],
        "vit_decoder_kwargs": {
            "num_layers": 1,
            "heads": 2,
            "dim_head": 16,
            "rope_theta": 100.0,
            "rope_dim_ratio": 0.75,
        },
    }))

    audio = root / "audio_vae"
    audio.mkdir()
    (audio / "config.json").write_text(json.dumps({
        "latents_mean": [0.0] * 32,
        "latents_std": [1.0] * 32,
    }))
    (audio / "metadata.json").write_text(json.dumps({
        "metadata": {"kwargs": {
            "encoder_dim": 16,
            "encoder_rates": [2, 2],
            "latent_dim": 32,
            "vae_latent_channels": 32,
            "decoder_dim": 32,
            "decoder_rates": [2, 2],
            "sample_rate": 32000,
        }}
    }))
    return text, dit


def main() -> None:
    source = mx.arange(6).reshape(2, 3).astype(mx.bfloat16)
    detached = detach_bfloat16(source)
    assert detached.dtype == mx.bfloat16
    assert mx.array_equal(detached, source).item()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        text, dit = write_metadata(root)
        pipe = MiniMaxH3Pipeline.from_pretrained(
            root,
            transformer_dir=dit,
            text_encoder_dir=text,
            low_memory=True,
            memory_limit_gb=1.0,
        )
        assert pipe._low_memory is True
        assert pipe.text_encoder is None
        assert pipe.dit is None
        assert pipe.video_vae is None
        assert pipe.audio_vae is None
        assert pipe._video_config.latent_channels == 24
        assert pipe._audio_config.latent_channels == 32
        assert pipe._dit_config.num_layers == 2
        print("low-memory init holds metadata only")


if __name__ == "__main__":
    main()
