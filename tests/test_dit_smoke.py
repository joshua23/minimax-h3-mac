"""Tiny-config forward test for the MiniMax-H3 DiT — no weights download required."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.adaln import ModulationCache, drop_adaln_weights, schedule_timesteps
from minimax_h3_mlx.block_cache import BlockCacheConfig, BlockResidualCache
from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT


def tiny_config() -> DiTConfig:
    hidden = 64
    return DiTConfig(
        hidden_size=hidden,
        num_layers=2,
        token_refiner_num_layers=2,
        num_attention_heads=4,
        attention_head_dim=16,
        ffn_hidden_size=32,
        latents_dim=4,
        audio_latents_dim=8,
        patch_size=(1, 2, 2),
        text_dim=32,
        timestep_input_dim=16,
        time_embed_hidden_size=hidden,
        time_embed_dim=32,
        adaln_out_features=6 * 3 * hidden,
        final_adaln_out_features=2 * hidden,
        rope_inv_freq_len=2,
    )


def build_packed_layout(n_text: int, n_video: int, n_audio: int):
    """Text rows first, then video, then audio — one contiguous packed sequence."""
    seq = n_text + n_video + n_audio
    text_indices = mx.arange(n_text)
    video_indices = mx.arange(n_text, n_text + n_video)
    audio_indices = mx.arange(n_text + n_video, seq)

    tags = mx.concatenate(
        [
            mx.full((n_text,), TAG_TEXT, dtype=mx.int32),
            mx.full((n_video,), TAG_VIDEO, dtype=mx.int32),
            mx.full((n_audio,), TAG_AUDIO, dtype=mx.int32),
        ]
    )
    # Text and audio sit at the clean level (index 0); video rows at the noisy level (index 1).
    timestep_indices = mx.concatenate(
        [
            mx.zeros((n_text,), dtype=mx.int32),
            mx.ones((n_video,), dtype=mx.int32),
            mx.zeros((n_audio,), dtype=mx.int32),
        ]
    )
    position_ids = mx.stack(
        [mx.arange(seq) % 3, mx.arange(seq) % 5, mx.arange(seq) % 7], axis=-1
    ).astype(mx.int32)
    return text_indices, video_indices, audio_indices, tags, timestep_indices, position_ids


def test_forward_shapes():
    cfg = tiny_config()
    mx.random.seed(0)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())

    n_text, n_video, n_audio = 5, 9, 3
    text_i, video_i, audio_i, tags, ts_i, pos = build_packed_layout(n_text, n_video, n_audio)

    video = mx.random.normal((1, n_video, cfg.video_patch_dim))
    audio = mx.random.normal((1, n_audio, cfg.audio_latents_dim))
    text = mx.random.normal((1, n_text, cfg.text_dim))
    timestep = mx.array([0.0, 0.7])

    v_out, a_out = dit(
        video, audio, text, timestep, ts_i, tags, pos, video_i, audio_i, text_i
    )
    mx.eval(v_out, a_out)

    assert v_out.shape == (1, n_video, cfg.video_patch_dim), v_out.shape
    assert a_out.shape == (1, n_audio, cfg.audio_latents_dim), a_out.shape
    assert not mx.any(mx.isnan(v_out)).item()
    assert not mx.any(mx.isnan(a_out)).item()
    print(f"forward ok: video {v_out.shape}, audio {a_out.shape}")
    return dit, cfg, (video, audio, text, timestep, ts_i, tags, pos, video_i, audio_i, text_i)


def test_modulation_cache_matches_live_projection():
    """The precomputed AdaLN table must reproduce the live projection bit-for-bit in float32."""
    dit, cfg, args = test_forward_shapes()
    timestep = args[3]

    live_v, live_a = dit(*args)
    mx.eval(live_v, live_a)

    cache = ModulationCache.build(dit, timestep, dtype=mx.float32)
    cached_v, cached_a = dit(*args, modulation_cache=cache)
    mx.eval(cached_v, cached_a)

    dv = float(mx.max(mx.abs(live_v - cached_v)).item())
    da = float(mx.max(mx.abs(live_a - cached_a)).item())
    assert dv == 0.0 and da == 0.0, f"cache mismatch: video {dv}, audio {da}"
    print(f"modulation cache exact: video delta {dv}, audio delta {da}")

    # Once cached, the projections can be dropped and the model still runs.
    freed = drop_adaln_weights(dit)
    after_v, after_a = dit(*args, modulation_cache=cache)
    mx.eval(after_v, after_a)
    assert float(mx.max(mx.abs(after_v - cached_v)).item()) == 0.0
    total = sum(p.size for _, p in _flatten(dit.parameters()))
    print(f"dropped adaln, freeing {freed / 1024:.1f} KB; {total:,} params remain")


def test_schedule_timesteps():
    sigmas = mx.array([1.0, 0.75, 0.5, 0.25])
    ts = schedule_timesteps(sigmas)
    assert ts.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0], ts.tolist()
    print(f"schedule timesteps: {ts.tolist()}")


def test_block_cache_disabled_is_exact_and_enabled_skips_tail():
    dit, _cfg, args = test_forward_shapes()
    baseline_v, baseline_a = dit(*args)
    disabled = BlockResidualCache(BlockCacheConfig(sigma_threshold=0.0))
    disabled_v, disabled_a = dit(*args, block_cache=disabled)
    mx.eval(baseline_v, baseline_a, disabled_v, disabled_a)
    assert float(mx.max(mx.abs(baseline_v - disabled_v)).item()) == 0.0
    assert float(mx.max(mx.abs(baseline_a - disabled_a)).item()) == 0.0

    cache = BlockResidualCache(
        BlockCacheConfig(
            sigma_threshold=1.0,
            start_percent=0.0,
            end_percent=1.0,
            max_consecutive=2,
            cache_depth=0.5,
        )
    )
    for step, sigma in enumerate((1.0, 0.9, 0.8, 0.0)):
        video, audio = dit(
            *args,
            block_cache=cache,
            block_cache_sigma=sigma,
            block_cache_step=step,
            block_cache_total_steps=4,
        )
        mx.eval(video, audio)
        assert not mx.any(mx.isnan(video)).item()
        assert not mx.any(mx.isnan(audio)).item()

    stats = cache.stats()
    assert stats["full_steps"] == 2, stats
    assert stats["cache_steps"] == 2, stats
    assert stats["skipped_blocks"] == 2, stats
    print(f"block cache: {stats}")


def _flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flatten(v, f"{prefix}.{i}")
    elif isinstance(tree, mx.array):
        yield prefix, tree


if __name__ == "__main__":
    test_schedule_timesteps()
    test_modulation_cache_matches_live_projection()
    test_block_cache_disabled_is_exact_and_enabled_skips_tail()
    print("\nall smoke tests passed")
