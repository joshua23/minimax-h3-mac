"""Video VAE memory boundaries must change execution timing, not values."""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.video_vae import VideoVAE, VideoVAEConfig


def config() -> VideoVAEConfig:
    return VideoVAEConfig(
        latent_channels=4,
        block_out_channels=(8,),
        layers_per_block=1,
        spatial_downsample_factors=(2,),
        temporal_downsample_factors=(1,),
        norm_num_groups=1,
        decoder_num_layers=2,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=12,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=2,
        decoder_rope_dim_ratio=0.5,
        clip_length=5,
        token_drop=1,
    )


def without_internal_eval(fn):
    original_eval = mx.eval
    mx.eval = lambda *args, **kwargs: None
    try:
        output = fn()
    finally:
        mx.eval = original_eval
    mx.eval(output)
    return output


def main() -> None:
    mx.random.seed(0)
    model = VideoVAE(config())
    mx.eval(model.parameters())
    model.tile_sample_min_height = model.tile_sample_min_width = 8
    model.tile_sample_min_overlap_height = model.tile_sample_min_overlap_width = 4
    latents = mx.random.normal((1, 2, 8, 8, 4))
    mx.eval(latents)

    eager_boundaries = model._decode_clip(latents)
    mx.eval(eager_boundaries)
    lazy_graph = without_internal_eval(lambda: model._decode_clip(latents))

    delta = float(mx.max(mx.abs(eager_boundaries - lazy_graph)).item())
    assert delta == 0.0, delta
    assert eager_boundaries.shape == (1, 2, 16, 16, 3)
    pixels = mx.random.normal((1, 3, 7, 16, 16))
    mx.eval(pixels)
    encoded = model.encode(pixels)
    mx.eval(encoded)
    lazy_encoded = without_internal_eval(lambda: model.encode(pixels))
    encode_delta = float(mx.max(mx.abs(encoded - lazy_encoded)).item())
    assert encode_delta == 0.0, encode_delta
    assert encoded.shape == (1, 8, 9, 8, 8), encoded.shape
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
    from minimax_h3_mlx.packing import PIXEL_MEAN, PIXEL_STD
    frames = np.random.default_rng(0).integers(0, 256, (7, 16, 16, 3), dtype=np.uint8)
    pixels = frames.astype(np.float32).transpose(3,0,1,2)[None]
    pixels = (pixels / 255.0 - np.array(PIXEL_MEAN,np.float32).reshape(1,3,1,1,1)) / np.array(PIXEL_STD,np.float32).reshape(1,3,1,1,1)
    full = model.encode(mx.array(pixels))
    pipeline = MiniMaxH3Pipeline(None,None,model,None)
    chunked = pipeline._encode_reference_video_frames(frames)
    mx.eval(full,chunked)
    chunk_delta=float(mx.max(mx.abs(full-chunked)).item())
    assert chunk_delta == 0.0, chunk_delta
    print(f"reference-video chunk normalization exact: delta={chunk_delta}")
    print(f"video VAE encode boundaries exact: shape={encoded.shape}, delta={encode_delta}")
    print(f"video VAE eval boundaries exact: shape={eager_boundaries.shape}, delta={delta}")


if __name__ == "__main__":
    main()
