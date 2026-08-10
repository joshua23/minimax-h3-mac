"""Bounded tests for the default-off MacSol/Sol-Attn reference path."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.macsol_reference import (  # noqa: E402
    H3PackedLengths,
    MacSolReferenceConfig,
    block_ranges,
    build_macsol_routing,
    dense_attention_reference,
    h3_macsol_config,
    h3_macsol_reference_attention,
    macsol_reference_attention,
    rounded_sink_blocks,
)


def assert_case(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} failed{(': ' + detail) if detail else ''}")
    print(f"ok  {name}{(' — ' + detail) if detail else ''}")


def _rng_qkv(sequence: int, heads: int = 2, head_dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((1, heads, sequence, head_dim), dtype=np.float32))
    k = mx.array(rng.standard_normal((1, heads, sequence, head_dim), dtype=np.float32))
    v = mx.array(rng.standard_normal((1, heads, sequence, head_dim), dtype=np.float32))
    return q, k, v


def _max_abs(a: mx.array, b: mx.array) -> float:
    mx.eval(a, b)
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def test_block_partitioning_and_sink_rounding() -> None:
    ranges = block_ranges(130, 64)
    assert_case("64-token block partitioning keeps a short tail", ranges == ((0, 64), (64, 128), (128, 130)))

    token_range, block_range = rounded_sink_blocks(260, sink_start=7, sink_tokens=66, block_size=64)
    assert_case("sink start rounds down to the containing block", token_range[0] == 0 and block_range[0] == 0)
    assert_case("sink stop rounds up to the covering block", token_range[1] == 128 and block_range[1] == 2)

    lengths = H3PackedLengths(text_tokens=33, conditioning_video_tokens=31, audio_tokens=1, target_video_tokens=195)
    cfg = h3_macsol_config(lengths, tau=1.0)
    token_range, block_range = rounded_sink_blocks(
        lengths.sequence_length,
        sink_start=cfg.sink_start,
        sink_tokens=cfg.sink_tokens,
        block_size=cfg.block_size,
    )
    assert_case("H3 prefix sink includes text+conditioning+audio", cfg.sink_tokens == 65)
    assert_case("H3 prefix sink is rounded outward to full 64-token blocks", token_range == (0, 128) and block_range == (0, 2))


def test_neighbor_sink_and_strict_threshold_routing() -> None:
    sequence = 256
    head_dim = 2
    q = np.zeros((1, 1, sequence, head_dim), dtype=np.float32)
    k = np.zeros_like(q)
    v = np.zeros_like(q)
    # Query block 3 points at KV block 1. With tau=0, block 1 is strictly above the row mean,
    # while block 0 remains equal/below threshold and is not a neighbor of block 3.
    q[:, :, 192:256, 0] = 1.0
    k[:, :, 64:128, 0] = 10.0
    routing = build_macsol_routing(
        mx.array(q),
        mx.array(k),
        MacSolReferenceConfig(block_size=64, tau=0.0, sink_start=0, sink_tokens=64),
        scale=1.0,
    )

    assert_case("neighbor block q3->kv2 is exact", bool(routing.neighbor_block_mask[0, 0, 3, 2]))
    assert_case("self block q3->kv3 is exact", bool(routing.neighbor_block_mask[0, 0, 3, 3]))
    assert_case("sink block 0 is exact for every query block", bool(routing.sink_block_mask[0, 0, 3, 0]))
    assert_case("threshold selects a non-neighbor high-proxy block", bool(routing.threshold_block_mask[0, 0, 3, 1]))
    assert_case("strict threshold does not select score equal to threshold", not bool(routing.threshold_block_mask[0, 0, 3, 0]))
    assert_case("union exact mask preserves threshold selection", 1 in routing.selected_kv_blocks(0, 0, 3))


def test_rejected_routing_variants_are_not_active_choices() -> None:
    q, k, _v = _rng_qkv(128, heads=1, head_dim=4, seed=31)
    for mode in ("budget_topk", "threshold_h3_structure", "spatial_tube"):
        try:
            build_macsol_routing(
                q,
                k,
                MacSolReferenceConfig(block_size=64, sink_start=0, sink_tokens=64, routing_mode=mode),
            )
        except ValueError as exc:
            message = str(exc)
            assert_case(f"{mode} is rejected as archived provenance only", mode in message and "threshold" in message)
        else:
            raise AssertionError(f"{mode} remained runnable in the active MacSol reference")


def test_centroid_value_fallback_is_not_drop_only() -> None:
    sequence = 256
    head_dim = 4
    q = mx.zeros((1, 1, sequence, head_dim), dtype=mx.float32)
    k = mx.zeros((1, 1, sequence, head_dim), dtype=mx.float32)
    v = mx.zeros((1, 1, sequence, head_dim), dtype=mx.float32)
    v[:, :, 0:64, :] = 1.0  # Far from q block 3 and therefore approximate, not exact.

    cfg = MacSolReferenceConfig(block_size=64, tau=0.0, sink_start=0, sink_tokens=0, include_neighbors=True)
    result = macsol_reference_attention(q, k, v, cfg, scale=1.0)
    dense = dense_attention_reference(q, k, v, scale=1.0)
    mx.eval(result.output, dense)
    q3 = np.asarray(result.output[:, :, 192:256, :])
    dense_q3 = np.asarray(dense[:, :, 192:256, :])
    assert_case("far unselected block contributes through V-sum fallback", float(q3.mean()) > 0.20)
    assert_case("centroid fallback matches dense when all K inside every block equal K-mean", np.max(np.abs(q3 - dense_q3)) < 1e-6)
    assert_case("some block pairs remain approximate", result.stats["approximate_block_pairs"] > 0)


def test_dense_equivalence_when_every_block_is_exact() -> None:
    q, k, v = _rng_qkv(192, heads=2, head_dim=8, seed=12)
    cfg = MacSolReferenceConfig(block_size=64, tau=-1.0e6, sink_start=0, sink_tokens=0)
    got = macsol_reference_attention(q, k, v, cfg).output
    dense = dense_attention_reference(q, k, v)
    delta = _max_abs(got, dense)
    assert_case("all-exact MacSol reference matches dense attention", delta < 2e-5, f"max_abs={delta:.3e}")


def test_h3_prefix_queries_are_dense_and_prefix_kv_sink_is_exact() -> None:
    lengths = H3PackedLengths(text_tokens=37, conditioning_video_tokens=20, audio_tokens=13, target_video_tokens=186)
    q, k, v = _rng_qkv(lengths.sequence_length, heads=1, head_dim=8, seed=23)
    result = h3_macsol_reference_attention(q, k, v, lengths, tau=1.0)
    dense = dense_attention_reference(q, k, v)
    prefix = lengths.prefix_tokens
    prefix_delta = _max_abs(result.output[:, :, :prefix, :], dense[:, :, :prefix, :])
    assert_case("H3 prefix query rows are exactly dense", prefix_delta < 2e-5, f"max_abs={prefix_delta:.3e}")

    summary = result.stats
    assert_case("H3 exact sink starts at token zero", summary["sink_request"] == {"start": 0, "tokens": prefix})
    assert_case("non-multiple H3 prefix sink rounds into block 1", summary["rounded_sink_block_range"] == [0, 2])
    assert_case("rounded sink block 1 is exact for a target query block", bool(result.routing.sink_block_mask[0, 0, 3, 1]))
    assert_case("reference path stays non-generation", summary["reference_only_not_generation_path"] is True)


def main() -> int:
    test_block_partitioning_and_sink_rounding()
    test_neighbor_sink_and_strict_threshold_routing()
    test_rejected_routing_variants_are_not_active_choices()
    test_centroid_value_fallback_is_not_drop_only()
    test_dense_equivalence_when_every_block_is_exact()
    test_h3_prefix_queries_are_dense_and_prefix_kv_sink_is_exact()
    print("PASS: MacSol reference semantics hold on bounded synthetic layouts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
