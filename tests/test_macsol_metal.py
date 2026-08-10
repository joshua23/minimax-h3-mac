"""Bounded parity tests for the rejected/internal MLX/Metal MacSol probe."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.macsol_metal import (  # noqa: E402
    MACSOL_METAL_STATUS,
    MACSOL_METAL_V2_STATUS,
    MACSOL_METAL_V3_STATUS,
    MACSOL_METAL_V4_STATUS,
    MACSOL_METAL_V5_STATUS,
    has_macsol_metal,
    has_macsol_metal_v2,
    has_macsol_metal_v3,
    has_macsol_metal_v4,
    has_macsol_metal_v5,
    macsol_attention_metal_subset,
    macsol_attention_metal_subset_v2,
    macsol_attention_metal_subset_v3,
    macsol_attention_metal_subset_v4,
    macsol_exact_tile_metal_v5,
    macsol_prepare_metal,
)
from minimax_h3_mlx.macsol_reference import (  # noqa: E402
    H3PackedLengths,
    MacSolReferenceConfig,
    build_macsol_routing,
    dense_attention_reference,
    h3_macsol_config,
    h3_macsol_reference_attention,
)


def assert_case(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} failed{(': ' + detail) if detail else ''}")
    print(f"ok  {name}{(' — ' + detail) if detail else ''}")


def _qkv(lengths: H3PackedLengths, *, heads: int = 2, head_dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    shape = (1, heads, lengths.sequence_length, head_dim)
    q = mx.array(rng.standard_normal(shape, dtype=np.float32)).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal(shape, dtype=np.float32)).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal(shape, dtype=np.float32)).astype(mx.bfloat16)
    return q, k, v


def _max_abs(a: mx.array, b: mx.array) -> float:
    mx.eval(a, b)
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def test_macsol_metal_statuses_are_rejected_provenance_only() -> None:
    statuses = [MACSOL_METAL_STATUS, MACSOL_METAL_V2_STATUS, MACSOL_METAL_V3_STATUS, MACSOL_METAL_V4_STATUS]
    for index, status in enumerate(statuses, start=1):
        assert_case(
            f"Metal v{index} status is rejected/provenance-only",
            "rejected" in status and "provenance_only" in status,
            status,
        )
    assert_case(
        "Metal v5 exact-tile status is rejected/provenance-only",
        "rejected" in MACSOL_METAL_V5_STATUS and "provenance_only" in MACSOL_METAL_V5_STATUS,
        MACSOL_METAL_V5_STATUS,
    )


def test_macsol_metal_threshold_and_forward_match_reference() -> None:
    if not has_macsol_metal():
        print("MacSol Metal rejected v1 provenance probe skipped: mx.fast.metal_kernel unavailable")
        return
    lengths = H3PackedLengths(text_tokens=33, conditioning_video_tokens=31, audio_tokens=1, target_video_tokens=195)
    q, k, v = _qkv(lengths, seed=17)
    cfg = h3_macsol_config(lengths, tau=0.0, block_size=64)
    prep = macsol_prepare_metal(q, k, v, block_size=64, tau=0.0)
    routing = build_macsol_routing(q, k, cfg)
    threshold_delta = _max_abs(prep.thresholds, mx.array(routing.thresholds[..., 0], dtype=mx.float32))
    assert_case("Metal threshold preparation matches reference", threshold_delta < 5e-5, f"max_abs={threshold_delta:.3e}")

    got = macsol_attention_metal_subset(q, k, v, cfg, preparation=prep).output
    ref = h3_macsol_reference_attention(q, k, v, lengths, tau=0.0, block_size=64).output
    delta = _max_abs(got, ref)
    assert_case("Rejected Metal v1 archive sparse online-softmax matches MacSol reference", delta < 5e-5, f"max_abs={delta:.3e}")


def test_macsol_metal_all_exact_sink_matches_dense() -> None:
    if not has_macsol_metal():
        print("MacSol Metal rejected v1 provenance probe skipped: mx.fast.metal_kernel unavailable")
        return
    lengths = H3PackedLengths(text_tokens=8, conditioning_video_tokens=8, audio_tokens=8, target_video_tokens=40)
    q, k, v = _qkv(lengths, heads=1, head_dim=8, seed=23)
    cfg = MacSolReferenceConfig(
        block_size=16,
        tau=0.0,
        sink_start=0,
        sink_tokens=lengths.sequence_length,
        prefix_query_tokens=0,
        force_prefix_queries_dense=False,
    )
    got = macsol_attention_metal_subset(q, k, v, cfg).output
    dense = dense_attention_reference(q, k, v)
    delta = _max_abs(got, dense)
    assert_case("all-exact rejected Metal v1 archive sink matches dense attention", delta < 5e-5, f"max_abs={delta:.3e}")


def test_macsol_metal_v2_threshold_and_forward_match_reference() -> None:
    if not has_macsol_metal_v2():
        print("MacSol Metal rejected v2 provenance probe skipped: mx.fast.metal_kernel unavailable")
        return
    lengths = H3PackedLengths(text_tokens=33, conditioning_video_tokens=31, audio_tokens=1, target_video_tokens=195)
    q, k, v = _qkv(lengths, seed=31)
    cfg = h3_macsol_config(lengths, tau=0.0, block_size=64)
    prep = macsol_prepare_metal(q, k, v, block_size=64, tau=0.0)
    got = macsol_attention_metal_subset_v2(q, k, v, cfg, preparation=prep).output
    ref = h3_macsol_reference_attention(q, k, v, lengths, tau=0.0, block_size=64).output
    delta = _max_abs(got, ref)
    assert_case("Rejected Metal v2 cooperative archive sparse online-softmax matches MacSol reference", delta < 5e-5, f"max_abs={delta:.3e}")


def test_macsol_metal_v2_all_exact_sink_matches_dense() -> None:
    if not has_macsol_metal_v2():
        print("MacSol Metal rejected v2 provenance probe skipped: mx.fast.metal_kernel unavailable")
        return
    lengths = H3PackedLengths(text_tokens=8, conditioning_video_tokens=8, audio_tokens=8, target_video_tokens=40)
    q, k, v = _qkv(lengths, heads=1, head_dim=8, seed=37)
    cfg = MacSolReferenceConfig(
        block_size=16,
        tau=0.0,
        sink_start=0,
        sink_tokens=lengths.sequence_length,
        prefix_query_tokens=0,
        force_prefix_queries_dense=False,
    )
    got = macsol_attention_metal_subset_v2(q, k, v, cfg).output
    dense = dense_attention_reference(q, k, v)
    delta = _max_abs(got, dense)
    assert_case("all-exact rejected Metal v2 archive sink matches dense attention", delta < 5e-5, f"max_abs={delta:.3e}")


def test_macsol_metal_v3_threshold_and_forward_match_reference() -> None:
    if not has_macsol_metal_v3():
        print("MacSol Metal rejected v3 provenance probe skipped: mx.fast.metal_kernel unavailable")
        return
    lengths = H3PackedLengths(text_tokens=33, conditioning_video_tokens=31, audio_tokens=1, target_video_tokens=195)
    q, k, v = _qkv(lengths, seed=41)
    cfg = h3_macsol_config(lengths, tau=0.0, block_size=64)
    prep = macsol_prepare_metal(q, k, v, block_size=64, tau=0.0)
    got = macsol_attention_metal_subset_v3(q, k, v, cfg, preparation=prep).output
    ref = h3_macsol_reference_attention(q, k, v, lengths, tau=0.0, block_size=64).output
    delta = _max_abs(got, ref)
    assert_case("Rejected Metal v3 q-block tiled archive sparse online-softmax matches MacSol reference", delta < 5e-5, f"max_abs={delta:.3e}")


def test_macsol_metal_v3_all_exact_sink_matches_dense() -> None:
    if not has_macsol_metal_v3():
        print("MacSol Metal rejected v3 provenance probe skipped: mx.fast.metal_kernel unavailable")
        return
    lengths = H3PackedLengths(text_tokens=8, conditioning_video_tokens=8, audio_tokens=8, target_video_tokens=40)
    q, k, v = _qkv(lengths, heads=1, head_dim=8, seed=43)
    cfg = MacSolReferenceConfig(
        block_size=16,
        tau=0.0,
        sink_start=0,
        sink_tokens=lengths.sequence_length,
        prefix_query_tokens=0,
        force_prefix_queries_dense=False,
    )
    got = macsol_attention_metal_subset_v3(q, k, v, cfg).output
    dense = dense_attention_reference(q, k, v)
    delta = _max_abs(got, dense)
    assert_case("all-exact rejected Metal v3 archive sink matches dense attention", delta < 5e-5, f"max_abs={delta:.3e}")


def test_macsol_metal_v4_threshold_and_forward_match_reference() -> None:
    if not has_macsol_metal_v4():
        print("MacSol Metal rejected v4 provenance probe skipped: mx.fast.metal_kernel unavailable")
        return
    lengths = H3PackedLengths(text_tokens=33, conditioning_video_tokens=31, audio_tokens=1, target_video_tokens=195)
    q, k, v = _qkv(lengths, seed=47)
    cfg = h3_macsol_config(lengths, tau=0.0, block_size=64)
    prep = macsol_prepare_metal(q, k, v, block_size=64, tau=0.0)
    got = macsol_attention_metal_subset_v4(q, k, v, cfg, preparation=prep).output
    ref = h3_macsol_reference_attention(q, k, v, lengths, tau=0.0, block_size=64).output
    delta = _max_abs(got, ref)
    assert_case("Rejected Metal v4 q-block/SIMD archive sparse online-softmax matches MacSol reference", delta < 5e-5, f"max_abs={delta:.3e}")


def test_macsol_metal_v4_all_exact_sink_matches_dense() -> None:
    if not has_macsol_metal_v4():
        print("MacSol Metal rejected v4 provenance probe skipped: mx.fast.metal_kernel unavailable")
        return
    lengths = H3PackedLengths(text_tokens=8, conditioning_video_tokens=8, audio_tokens=8, target_video_tokens=40)
    q, k, v = _qkv(lengths, heads=1, head_dim=8, seed=53)
    cfg = MacSolReferenceConfig(
        block_size=16,
        tau=0.0,
        sink_start=0,
        sink_tokens=lengths.sequence_length,
        prefix_query_tokens=0,
        force_prefix_queries_dense=False,
    )
    got = macsol_attention_metal_subset_v4(q, k, v, cfg).output
    dense = dense_attention_reference(q, k, v)
    delta = _max_abs(got, dense)
    assert_case("all-exact rejected Metal v4 archive sink matches dense attention", delta < 5e-5, f"max_abs={delta:.3e}")


def _exact_tile_reference(q: mx.array, k: mx.array, v: mx.array, *, q_block: int, kv_block: int, head: int = 0, block_size: int = 64) -> mx.array:
    scale = float(q.shape[-1] ** -0.5)
    qlo = q_block * block_size
    klo = kv_block * block_size
    q_block_mx = q[0, head, qlo : qlo + block_size].astype(mx.float32)
    k_block_mx = k[0, head, klo : klo + block_size].astype(mx.float32)
    v_block_mx = v[0, head, klo : klo + block_size].astype(mx.float32)
    scores = mx.matmul(q_block_mx, k_block_mx.T) * scale
    return mx.matmul(mx.softmax(scores, axis=-1), v_block_mx)


def test_macsol_metal_v5_exact_tile_matches_local_dense_bf16_d128() -> None:
    if not has_macsol_metal_v5():
        print("MacSol Metal v5 exact-tile probe skipped: mx.fast.metal_kernel unavailable")
        return
    lengths = H3PackedLengths(text_tokens=64, conditioning_video_tokens=0, audio_tokens=0, target_video_tokens=64)
    q, k, v = _qkv(lengths, heads=1, head_dim=128, seed=59)
    got = macsol_exact_tile_metal_v5(q, k, v, q_block_start=1, kv_block_start=0, kv_block_count=1, head_count=1).output[0, 0, 0]
    ref = _exact_tile_reference(q, k, v, q_block=1, kv_block=0)
    delta = _max_abs(got, ref)
    assert_case("Metal v5 exact 64x64 BF16 D=128 tile matches local dense", delta < 2e-4, f"max_abs={delta:.3e}")


def test_macsol_metal_v5_exact_tile_batch_matches_each_kv_block() -> None:
    if not has_macsol_metal_v5():
        print("MacSol Metal v5 exact-tile probe skipped: mx.fast.metal_kernel unavailable")
        return
    lengths = H3PackedLengths(text_tokens=64, conditioning_video_tokens=0, audio_tokens=0, target_video_tokens=128)
    q, k, v = _qkv(lengths, heads=1, head_dim=128, seed=61)
    got = macsol_exact_tile_metal_v5(q, k, v, q_block_start=0, kv_block_start=1, kv_block_count=2, head_count=1).output
    ref1 = _exact_tile_reference(q, k, v, q_block=0, kv_block=1)
    ref2 = _exact_tile_reference(q, k, v, q_block=0, kv_block=2)
    delta = max(_max_abs(got[0, 0, 0], ref1), _max_abs(got[0, 0, 1], ref2))
    assert_case("Metal v5 exact-tile batched KV blocks match local dense", delta < 2e-4, f"max_abs={delta:.3e}")


def main() -> int:
    test_macsol_metal_statuses_are_rejected_provenance_only()
    test_macsol_metal_threshold_and_forward_match_reference()
    test_macsol_metal_all_exact_sink_matches_dense()
    test_macsol_metal_v2_threshold_and_forward_match_reference()
    test_macsol_metal_v2_all_exact_sink_matches_dense()
    test_macsol_metal_v3_threshold_and_forward_match_reference()
    test_macsol_metal_v3_all_exact_sink_matches_dense()
    test_macsol_metal_v4_threshold_and_forward_match_reference()
    test_macsol_metal_v4_all_exact_sink_matches_dense()
    test_macsol_metal_v5_exact_tile_matches_local_dense_bf16_d128()
    test_macsol_metal_v5_exact_tile_batch_matches_each_kv_block()
    print("PASS: MacSol Metal bounded semantics hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
