"""Tiny-config forward test for the MiniMax-H3 DiT — no weights download required."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.adaln import ModulationCache, drop_adaln_weights, schedule_timesteps
from minimax_h3_mlx.block_cache import BlockCacheConfig, BlockResidualCache
from minimax_h3_mlx.config import MODALITY_NUM, TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
# Rejected/tiny-only route probes are kept in this file as archive-only helpers by
# renaming them away from pytest's `test_*` collection prefix.  The public profiler
# candidate surface is checked in tests/test_profile_hotpath_cli.py.
from minimax_h3_mlx.dit import (
    DENSE_DEQUANT_PROFILE_OFF,
    DENSE_DEQUANT_PROFILE_QKV_ONLY_TILED,
    DENSE_DEQUANT_PROFILE_FFN_FC2_TILED,
    DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
    DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED,
    MiniMaxH3DiT,
    apply_dense_dequant_profile_to_block,
    apply_rotary,
    apply_rotary_qk_metal,
    gather_packed_modulation_rows,
    has_indexed_gated_residual_metal,
    has_qkv_rmsnorm_rotary_sdpa_metal,
    has_qkv_rmsnorm_sdpa_metal,
    has_rotary_qk_metal,
    has_sdpa_out_layout_metal,
    has_swiglu_from_fused_metal,
    indexed_gated_residual_metal,
    materialize_attention_input_contiguous,
    materialize_attention_output_contiguous,
    materialize_ffn_hidden_contiguous,
    materialize_ffn_input_contiguous,
    materialize_sdpa_inputs_contiguous,
    qkv_rmsnorm_rotary_sdpa_metal,
    qkv_rmsnorm_sdpa_metal,
    sdpa_head_batch_rank3,
    sdpa_headgroup_split_rank4,
    sdpa_out_to_bshd_metal,
    swiglu_from_fused_metal,
)


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


_ARCHIVED_HOTPATH_CANDIDATE_HELPER_SUFFIXES = {
    "ffn_projection_2d_qmm": ("ffn_fc1_fc2_2d_projection_candidate_exact",),
    "ffn_fc1_rank2_qmm": ("ffn_fc1_rank2_qmm_candidate_exact_and_fc2_unchanged",),
    "ffn_fc2_rank2_qmm": ("ffn_fc2_rank2_qmm_candidate_exact_and_fc1_unchanged",),
    "ffn_fc2_input_chunked_qmm": ("ffn_fc2_input_chunked_qmm_candidate_bounded_for_quantized_tiny_fc2",),
    "ffn_fc1_dense_dequant": ("ffn_fc1_dense_dequant_candidate_exact_for_quantized_tiny_fc1",),
    "ffn_fc1_tiled_dense_dequant": ("ffn_fc1_tiled_dense_dequant_candidate_exact_for_quantized_tiny_fc1",),
    "ffn_fc2_dense_dequant": ("ffn_fc2_dense_dequant_candidate_exact_for_dense_tiny_fc2",),
    "ffn_hidden_tile_stream": ("ffn_hidden_tile_stream_candidate_bounded_for_quantized_tiny_ffn",),
    "ffn_subgraph_compile": ("ffn_subgraph_compile_candidate_exact_and_lora_fallback",),
    "indexed_gated_residual_metal": ("indexed_gated_residual_metal_candidate_bounded",),
    "attention_projection_2d_qmm": ("attention_qkv_2d_projection_candidate_exact",),
    "attention_qkv_input_chunked_qmm": (
        "attention_qkv_input_chunked_qmm_candidate_bounded_for_quantized_tiny_qkv",
    ),
    "attention_sdpa_headgroup_split_rank4": ("attention_sdpa_headgroup_split_rank4_candidate_exact",),
    "attention_sdpa_head_batch_rank3": ("attention_sdpa_head_batch_rank3_candidate_exact",),
    "ffn_mx_split_swiglu": ("ffn_mx_split_swiglu_candidate_exact",),
    "ffn_metal_swiglu": ("ffn_fused_swiglu_metal_candidate_bounded",),
    "ffn_fused_swiglu_metal": ("ffn_fused_swiglu_metal_candidate_bounded",),
    "ffn_fc1_split_gate_value_quantized_qmm": (
        "ffn_fc1_split_gate_value_quantized_qmm_candidate_exact_for_quantized_tiny_fc1",
    ),
    "ffn_pre_fc1_contiguous": ("ffn_pre_fc1_contiguous_candidate_exact",),
    "ffn_pre_fc2_contiguous": ("ffn_pre_fc2_contiguous_candidate_exact",),
    "ffn_sequence_chunked": ("ffn_sequence_chunk_candidate_exact",),
    "adaln_packed_gather": ("adaln_packed_gather_candidate_exact_and_projection_already_unique",),
    "attention_pre_qkv_contiguous": ("attention_pre_qkv_contiguous_candidate_exact",),
    "attention_qkv_pretranspose_layout": ("attention_qkv_pretranspose_layout_candidate_exact",),
    "attention_pre_sdpa_contiguous": ("attention_pre_sdpa_contiguous_candidate_exact",),
    "attention_pre_out_proj_contiguous": ("attention_pre_out_proj_contiguous_candidate_exact",),
    "attention_qkv_rmsnorm_sdpa_metal": ("attention_qkv_rmsnorm_sdpa_metal_candidate_bounded",),
    "attention_qkv_rmsnorm_rotary_sdpa_metal": (
        "attention_qkv_rmsnorm_rotary_sdpa_metal_candidate_bounded",
    ),
    "attention_sdpa_out_layout_metal": ("attention_sdpa_out_layout_metal_candidate_exact",),
    "attention_rotary_qk_metal": ("attention_rotary_qk_metal_candidate_bounded",),
}

_PROMOTED_HOTPATH_CANDIDATE_TESTS = {
    "test_ffn_fc2_tiled_dense_dequant_candidate_exact_for_quantized_tiny_fc2",
    "test_attention_out_dense_dequant_candidate_exact_for_quantized_tiny_out_proj",
    "test_attention_out_tiled_dense_dequant_candidate_exact_for_quantized_tiny_out_proj",
    "test_attention_qkv_tiled_dense_dequant_candidate_exact_for_quantized_tiny_qkv",
}

_DENSE_CORE_MANUAL_TESTS = {
    "test_archived_hotpath_candidates_are_decollected_and_not_manually_invoked",
    "test_schedule_timesteps",
    "test_refined_text_cache_candidate_exact_and_ignores_text_embeds",
    "test_modulation_cache_matches_live_projection",
    "test_block_cache_disabled_is_exact_and_enabled_skips_tail",
}


def _profile_archived_hotpath_candidate_names() -> set[str]:
    profile_path = Path(__file__).resolve().parents[1] / "scripts" / "profile_dit_block_hotpath.py"
    tree = ast.parse(profile_path.read_text())
    literal_tuples: dict[str, tuple[str, ...]] = {}
    archived_assignment_seen = False
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in {"QUARANTINED_HOTPATH_CANDIDATES", "ARCHIVED_NONPROMOTED_HOTPATH_CANDIDATES"}:
            literal_tuples[target.id] = tuple(ast.literal_eval(node.value))
        elif target.id == "ARCHIVED_HOTPATH_CANDIDATES":
            archived_assignment_seen = True
    assert archived_assignment_seen, "profile hotpath ARCHIVED_HOTPATH_CANDIDATES assignment moved"
    return set(literal_tuples["QUARANTINED_HOTPATH_CANDIDATES"]) | set(
        literal_tuples["ARCHIVED_NONPROMOTED_HOTPATH_CANDIDATES"]
    )


def _defined_functions_and_manual_runner_calls() -> tuple[list[str], list[str]]:
    source = Path(__file__).read_text()
    tree = ast.parse(source)
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    manual_calls: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        is_main_guard = (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
        )
        if not is_main_guard:
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id.startswith(("test_", "archived_")):
                    manual_calls.append(call.func.id)
    return functions, manual_calls


def test_archived_hotpath_candidates_are_decollected_and_not_manually_invoked():
    archived_candidates = _profile_archived_hotpath_candidate_names()
    functions, manual_calls = _defined_functions_and_manual_runner_calls()
    collected_tests = [name for name in functions if name.startswith("test_")]

    direct_collected_hits = {
        candidate: [name for name in collected_tests if candidate in name]
        for candidate in archived_candidates
    }
    direct_manual_hits = {
        candidate: [name for name in manual_calls if candidate in name]
        for candidate in archived_candidates
    }
    direct_hits = {
        candidate: {"pytest": direct_collected_hits[candidate], "manual": direct_manual_hits[candidate]}
        for candidate in archived_candidates
        if direct_collected_hits[candidate] or direct_manual_hits[candidate]
    }
    assert not direct_hits, f"archived candidate names still active: {direct_hits}"

    helper_candidates = set(_ARCHIVED_HOTPATH_CANDIDATE_HELPER_SUFFIXES)
    assert helper_candidates <= archived_candidates
    helper_suffixes = tuple(
        suffix for suffixes in _ARCHIVED_HOTPATH_CANDIDATE_HELPER_SUFFIXES.values() for suffix in suffixes
    )
    alias_collected_hits = [name for name in collected_tests if name.endswith(helper_suffixes)]
    alias_manual_hits = [name for name in manual_calls if name.endswith(helper_suffixes)]
    assert not alias_collected_hits, f"archived helper aliases still pytest-collected: {alias_collected_hits}"
    assert not alias_manual_hits, f"archived helper aliases still called by __main__: {alias_manual_hits}"

    expected_manual_calls = _DENSE_CORE_MANUAL_TESTS | _PROMOTED_HOTPATH_CANDIDATE_TESTS
    assert set(manual_calls) == expected_manual_calls


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


def archived_ffn_mx_split_swiglu_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(7)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 11, cfg.hidden_size))
    block = dit.blocks[0]

    block.mlp.use_mx_split_swiglu_candidate = False
    baseline = block.mlp(x)
    block.mlp.use_mx_split_swiglu_candidate = True
    candidate = block.mlp(x)
    block.mlp.use_mx_split_swiglu_candidate = False
    mx.eval(baseline, candidate)

    delta = float(mx.max(mx.abs(baseline - candidate)).item())
    assert delta == 0.0, f"mx.split SwiGLU candidate changed FFN output by {delta}"
    print(f"ffn mx.split candidate exact: max delta {delta}")


def archived_ffn_fc1_fc2_2d_projection_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(17)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 13, cfg.hidden_size))
    block = dit.blocks[0]

    assert block.mlp.use_ffn_2d_projection_candidate is False
    block.mlp.use_ffn_2d_projection_candidate = False
    baseline_fc1 = block.mlp._fc1_project(x)
    gate, value = baseline_fc1[..., : block.mlp._ffn], baseline_fc1[..., block.mlp._ffn :]
    hidden = mx.sigmoid(gate) * gate * value
    baseline_fc2 = block.mlp._fc2_project(hidden)
    baseline = block.mlp(x)

    block.mlp.use_ffn_2d_projection_candidate = True
    candidate_fc1 = block.mlp._fc1_project(x)
    candidate_fc2 = block.mlp._fc2_project(hidden)
    candidate = block.mlp(x)
    block.mlp.use_ffn_2d_projection_candidate = False
    mx.eval(baseline_fc1, candidate_fc1, baseline_fc2, candidate_fc2, baseline, candidate)

    deltas = {
        "fc1": float(mx.max(mx.abs(baseline_fc1 - candidate_fc1)).item()),
        "fc2": float(mx.max(mx.abs(baseline_fc2 - candidate_fc2)).item()),
        "mlp": float(mx.max(mx.abs(baseline - candidate)).item()),
    }
    assert all(delta == 0.0 for delta in deltas.values()), f"FFN 2D projection deltas: {deltas}"
    print(f"ffn fc1/fc2 2D projection candidate exact: {deltas}")


def archived_ffn_fc1_rank2_qmm_candidate_exact_and_fc2_unchanged():
    cfg = tiny_config()
    mx.random.seed(19)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 13, cfg.hidden_size))
    block = dit.blocks[0]

    assert block.mlp.use_ffn_fc1_rank2_qmm_candidate is False
    assert block.mlp.use_ffn_fc2_rank2_qmm_candidate is False
    assert block.mlp.use_ffn_2d_projection_candidate is False
    block.mlp.use_ffn_fc1_rank2_qmm_candidate = False
    baseline_fc1 = block.mlp._fc1_project(x)
    baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=x)
    baseline_fc2 = block.mlp._fc2_project(baseline_hidden)
    baseline = block.mlp(x)

    block.mlp.use_ffn_fc1_rank2_qmm_candidate = True
    candidate_fc1 = block.mlp._fc1_project(x)
    candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=x)
    candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
    candidate = block.mlp(x)
    block.mlp.use_ffn_fc1_rank2_qmm_candidate = False
    mx.eval(
        baseline_fc1,
        candidate_fc1,
        baseline_hidden,
        candidate_hidden,
        baseline_fc2,
        candidate_fc2,
        baseline,
        candidate,
    )

    deltas = {
        "fc1": float(mx.max(mx.abs(baseline_fc1 - candidate_fc1)).item()),
        "hidden": float(mx.max(mx.abs(baseline_hidden - candidate_hidden)).item()),
        "fc2": float(mx.max(mx.abs(baseline_fc2 - candidate_fc2)).item()),
        "mlp": float(mx.max(mx.abs(baseline - candidate)).item()),
    }
    assert candidate_fc1.shape == baseline_fc1.shape
    assert candidate_fc2.shape == baseline_fc2.shape
    assert candidate.shape == baseline.shape
    assert candidate_fc1.dtype == baseline_fc1.dtype
    assert candidate_fc2.dtype == baseline_fc2.dtype
    assert candidate.dtype == baseline.dtype
    assert all(delta == 0.0 for delta in deltas.values()), f"FFN fc1 rank-2 QMM deltas: {deltas}"
    print(f"ffn fc1 rank-2 QMM candidate exact and fc2 unchanged: {deltas}")


def archived_ffn_fc1_split_gate_value_quantized_qmm_candidate_exact_for_quantized_tiny_fc1():
    cfg = tiny_config()
    mx.random.seed(20)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    block = dit.blocks[0]
    nn.quantize(
        block.mlp,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear) and path.endswith("fc1"),
    )
    mx.eval(block.mlp.parameters())

    assert block.mlp.use_ffn_fc1_split_gate_value_quantized_qmm_candidate is False
    assert block.mlp.use_ffn_fc1_rank2_qmm_candidate is False
    block.mlp.use_ffn_fc1_split_gate_value_quantized_qmm_candidate = False
    baseline_fc1 = block.mlp._fc1_project(x)
    baseline_gate, baseline_value = baseline_fc1[..., : block.mlp._ffn], baseline_fc1[..., block.mlp._ffn :]
    baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=x)
    baseline_fc2 = block.mlp._fc2_project(baseline_hidden)
    baseline = block.mlp(x)

    block.mlp.use_ffn_fc1_split_gate_value_quantized_qmm_candidate = True
    fallback_fc1 = block.mlp._fc1_project(x, lora=object())
    candidate_gate, candidate_value = block.mlp._fc1_split_gate_value_project(x)
    candidate_fc1 = block.mlp._fc1_project(x)
    candidate_hidden_direct = nn.silu(candidate_gate) * candidate_value
    candidate_hidden_reassembled = block.mlp._swiglu_hidden(candidate_fc1, x=x)
    candidate_fc2 = block.mlp._fc2_project(candidate_hidden_direct)
    candidate = block.mlp(x)
    split_info = block.mlp.fc1_split_gate_value_quantized_qmm_info()
    block.mlp.use_ffn_fc1_split_gate_value_quantized_qmm_candidate = False
    mx.eval(
        baseline_fc1,
        fallback_fc1,
        baseline_gate,
        candidate_gate,
        baseline_value,
        candidate_value,
        candidate_fc1,
        baseline_hidden,
        candidate_hidden_direct,
        candidate_hidden_reassembled,
        baseline_fc2,
        candidate_fc2,
        baseline,
        candidate,
    )

    deltas = {
        "gate": float(mx.max(mx.abs(baseline_gate.astype(mx.float32) - candidate_gate.astype(mx.float32))).item()),
        "value": float(mx.max(mx.abs(baseline_value.astype(mx.float32) - candidate_value.astype(mx.float32))).item()),
        "fc1_reassembled": float(
            mx.max(mx.abs(baseline_fc1.astype(mx.float32) - candidate_fc1.astype(mx.float32))).item()
        ),
        "fallback_fc1": float(mx.max(mx.abs(baseline_fc1.astype(mx.float32) - fallback_fc1.astype(mx.float32))).item()),
        "hidden_direct": float(
            mx.max(mx.abs(baseline_hidden.astype(mx.float32) - candidate_hidden_direct.astype(mx.float32))).item()
        ),
        "hidden_reassembled": float(
            mx.max(mx.abs(baseline_hidden.astype(mx.float32) - candidate_hidden_reassembled.astype(mx.float32))).item()
        ),
        "fc2": float(mx.max(mx.abs(baseline_fc2.astype(mx.float32) - candidate_fc2.astype(mx.float32))).item()),
        "mlp": float(mx.max(mx.abs(baseline.astype(mx.float32) - candidate.astype(mx.float32))).item()),
    }
    assert candidate_gate.shape == baseline_gate.shape
    assert candidate_value.shape == baseline_value.shape
    assert candidate_fc1.shape == baseline_fc1.shape
    assert candidate_hidden_direct.shape == baseline_hidden.shape
    assert candidate_fc2.shape == baseline_fc2.shape
    assert candidate.shape == baseline.shape
    assert candidate_gate.dtype == baseline_gate.dtype
    assert candidate_value.dtype == baseline_value.dtype
    assert candidate_fc1.dtype == baseline_fc1.dtype
    assert candidate_hidden_direct.dtype == baseline_hidden.dtype
    assert candidate_fc2.dtype == baseline_fc2.dtype
    assert candidate.dtype == baseline.dtype
    assert split_info["source_is_quantized"] is True
    assert split_info["uses_quantized_matmul"] is True
    assert split_info["dense_dequantization"] is False
    assert split_info["gate_output_rows"] == [0, cfg.ffn_hidden_size]
    assert split_info["value_output_rows"] == [cfg.ffn_hidden_size, 2 * cfg.ffn_hidden_size]
    assert split_info["separate_projection_count"] == 2
    assert split_info["materializes_fused_fc1_in_forward"] is False
    assert deltas["fallback_fc1"] == 0.0, f"LoRA fallback path unexpectedly used split fc1 QMM: {deltas}"
    assert all(delta <= 1e-6 for delta in deltas.values()), f"FFN split gate/value fc1 QMM deltas: {deltas}"
    print(f"ffn fc1 split gate/value quantized QMM candidate within tiny quantized tolerance: {deltas}")



def archived_ffn_fc2_rank2_qmm_candidate_exact_and_fc1_unchanged():
    cfg = tiny_config()
    mx.random.seed(21)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 13, cfg.hidden_size))
    block = dit.blocks[0]

    assert block.mlp.use_ffn_fc2_rank2_qmm_candidate is False
    assert block.mlp.use_ffn_2d_projection_candidate is False
    block.mlp.use_ffn_fc2_rank2_qmm_candidate = False
    baseline_fc1 = block.mlp._fc1_project(x)
    baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=x)
    baseline_fc2 = block.mlp._fc2_project(baseline_hidden)
    baseline = block.mlp(x)

    block.mlp.use_ffn_fc2_rank2_qmm_candidate = True
    candidate_fc1 = block.mlp._fc1_project(x)
    candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=x)
    candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
    candidate = block.mlp(x)
    block.mlp.use_ffn_fc2_rank2_qmm_candidate = False
    mx.eval(
        baseline_fc1,
        candidate_fc1,
        baseline_hidden,
        candidate_hidden,
        baseline_fc2,
        candidate_fc2,
        baseline,
        candidate,
    )

    deltas = {
        "fc1": float(mx.max(mx.abs(baseline_fc1 - candidate_fc1)).item()),
        "hidden": float(mx.max(mx.abs(baseline_hidden - candidate_hidden)).item()),
        "fc2": float(mx.max(mx.abs(baseline_fc2 - candidate_fc2)).item()),
        "mlp": float(mx.max(mx.abs(baseline - candidate)).item()),
    }
    assert candidate_fc1.shape == baseline_fc1.shape
    assert candidate_fc2.shape == baseline_fc2.shape
    assert candidate.shape == baseline.shape
    assert candidate_fc1.dtype == baseline_fc1.dtype
    assert candidate_fc2.dtype == baseline_fc2.dtype
    assert candidate.dtype == baseline.dtype
    assert all(delta == 0.0 for delta in deltas.values()), f"FFN fc2 rank-2 QMM deltas: {deltas}"
    print(f"ffn fc2 rank-2 QMM candidate exact and fc1 unchanged: {deltas}")


def archived_ffn_fc1_dense_dequant_candidate_exact_for_quantized_tiny_fc1():
    cfg = tiny_config()
    mx.random.seed(22)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    nn.quantize(
        block.mlp,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear) and path.endswith("fc1"),
    )
    mx.eval(block.mlp.parameters())

    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)

    assert block.mlp.use_ffn_fc1_dense_dequant_candidate is False
    block.mlp.use_ffn_fc1_dense_dequant_candidate = False
    baseline_fc1 = block.mlp._fc1_project(x)
    baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=x)
    baseline_fc2 = block.mlp._fc2_project(baseline_hidden)
    baseline = block.mlp(x)

    block.mlp.use_ffn_fc1_dense_dequant_candidate = True
    fallback_fc1 = block.mlp._fc1_project(x, lora=object())
    fallback_cache_info = block.mlp.fc1_dense_dequant_cache_info()
    candidate_fc1 = block.mlp._fc1_project(x)
    candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=x)
    candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
    candidate = block.mlp(x)
    cache_info = block.mlp.fc1_dense_dequant_cache_info()
    block.mlp.use_ffn_fc1_dense_dequant_candidate = False
    block.mlp.clear_fc1_dense_dequant_cache()
    mx.eval(
        baseline_fc1,
        fallback_fc1,
        candidate_fc1,
        baseline_hidden,
        candidate_hidden,
        baseline_fc2,
        candidate_fc2,
        baseline,
        candidate,
    )

    deltas = {
        "fc1": float(mx.max(mx.abs(baseline_fc1.astype(mx.float32) - candidate_fc1.astype(mx.float32))).item()),
        "fallback_fc1": float(
            mx.max(mx.abs(baseline_fc1.astype(mx.float32) - fallback_fc1.astype(mx.float32))).item()
        ),
        "hidden": float(
            mx.max(mx.abs(baseline_hidden.astype(mx.float32) - candidate_hidden.astype(mx.float32))).item()
        ),
        "fc2": float(mx.max(mx.abs(baseline_fc2.astype(mx.float32) - candidate_fc2.astype(mx.float32))).item()),
        "mlp": float(mx.max(mx.abs(baseline.astype(mx.float32) - candidate.astype(mx.float32))).item()),
    }
    assert candidate_fc1.shape == baseline_fc1.shape
    assert candidate_hidden.shape == baseline_hidden.shape
    assert candidate_fc2.shape == baseline_fc2.shape
    assert candidate.shape == baseline.shape
    assert candidate_fc1.dtype == baseline_fc1.dtype
    assert candidate_hidden.dtype == baseline_hidden.dtype
    assert candidate_fc2.dtype == baseline_fc2.dtype
    assert candidate.dtype == baseline.dtype
    assert fallback_cache_info["source_is_quantized"] is True
    assert fallback_cache_info["cached"] is False
    assert cache_info["source_is_quantized"] is True
    assert cache_info["cached"] is True
    assert cache_info["dense_nbytes"] > cache_info["source_weight_nbytes"]
    assert deltas["fallback_fc1"] == 0.0, f"LoRA fallback path unexpectedly used dense fc1 cache: {deltas}"
    assert all(delta <= 1e-6 for delta in deltas.values()), f"FFN dense-fc1 tiny deltas: {deltas}"
    print(f"ffn fc1 dense-dequant candidate within tiny quantized tolerance: {deltas}")



def archived_ffn_fc1_tiled_dense_dequant_candidate_exact_for_quantized_tiny_fc1():
    cfg = tiny_config()
    mx.random.seed(25)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    block = dit.blocks[0]
    nn.quantize(
        block.mlp,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear) and path.endswith("fc1"),
    )
    mx.eval(block.mlp.parameters())

    assert block.mlp.use_ffn_fc1_tiled_dense_dequant_candidate is False
    assert block.mlp.use_ffn_fc1_dense_dequant_candidate is False
    block.mlp.ffn_fc1_tiled_output_channels = 17
    baseline_fc1 = block.mlp._fc1_project(x)
    baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=x)
    baseline_fc2 = block.mlp._fc2_project(baseline_hidden)
    baseline = block.mlp(x)

    block.mlp.use_ffn_fc1_tiled_dense_dequant_candidate = True
    fallback_fc1 = block.mlp._fc1_project(x, lora=object())
    candidate_fc1 = block.mlp._fc1_project(x)
    candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=x)
    candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
    candidate = block.mlp(x)
    tile_info = block.mlp.fc1_tiled_dense_dequant_info()
    resident_cache_info = block.mlp.fc1_dense_dequant_cache_info()
    block.mlp.use_ffn_fc1_tiled_dense_dequant_candidate = False
    mx.eval(
        baseline_fc1,
        fallback_fc1,
        candidate_fc1,
        baseline_hidden,
        candidate_hidden,
        baseline_fc2,
        candidate_fc2,
        baseline,
        candidate,
    )

    deltas = {
        "fc1": float(mx.max(mx.abs(baseline_fc1.astype(mx.float32) - candidate_fc1.astype(mx.float32))).item()),
        "fallback_fc1": float(
            mx.max(mx.abs(baseline_fc1.astype(mx.float32) - fallback_fc1.astype(mx.float32))).item()
        ),
        "hidden": float(
            mx.max(mx.abs(baseline_hidden.astype(mx.float32) - candidate_hidden.astype(mx.float32))).item()
        ),
        "fc2": float(mx.max(mx.abs(baseline_fc2.astype(mx.float32) - candidate_fc2.astype(mx.float32))).item()),
        "mlp": float(mx.max(mx.abs(baseline.astype(mx.float32) - candidate.astype(mx.float32))).item()),
    }
    assert candidate_fc1.shape == baseline_fc1.shape
    assert candidate_hidden.shape == baseline_hidden.shape
    assert candidate_fc2.shape == baseline_fc2.shape
    assert candidate.shape == baseline.shape
    assert candidate_fc1.dtype == baseline_fc1.dtype
    assert candidate_hidden.dtype == baseline_hidden.dtype
    assert candidate_fc2.dtype == baseline_fc2.dtype
    assert candidate.dtype == baseline.dtype
    assert tile_info["source_is_quantized"] is True
    assert tile_info["effective_tile_output_channels"] == 17
    assert tile_info["tile_count"] == 4
    assert tile_info["max_dense_tile_nbytes"] < tile_info["full_dense_nbytes_if_resident"]
    assert tile_info["persistent_full_dense_allocation"] is False
    assert tile_info["fused_gate_value_row_order_preserved"] is True
    assert resident_cache_info["cached"] is False
    assert deltas["fallback_fc1"] == 0.0, f"LoRA fallback path unexpectedly used tiled fc1: {deltas}"
    assert all(delta <= 1e-6 for delta in deltas.values()), f"FFN tiled dense-fc1 tiny deltas: {deltas}"
    print(f"ffn fc1 tiled dense-dequant candidate within tiny quantized tolerance: {deltas}")


def archived_ffn_fc2_input_chunked_qmm_candidate_bounded_for_quantized_tiny_fc2():
    hidden = 64
    cfg = DiTConfig(
        hidden_size=hidden,
        num_layers=2,
        token_refiner_num_layers=2,
        num_attention_heads=4,
        attention_head_dim=16,
        ffn_hidden_size=96,
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
    mx.random.seed(33)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    nn.quantize(
        block.mlp,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear) and path.endswith("fc2"),
    )
    # The real 4-bit H3 block stores quantization scales/biases in bf16; mirror that path so the
    # test constrains shape, fallback, and bounded accumulation drift rather than only fp32 scales.
    block.mlp.fc2.scales = block.mlp.fc2.scales.astype(mx.bfloat16)
    block.mlp.fc2.biases = block.mlp.fc2.biases.astype(mx.bfloat16)
    mx.eval(block.mlp.parameters())

    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    assert block.mlp.use_ffn_fc2_input_chunked_qmm_candidate is False
    block.mlp.ffn_fc2_input_chunk_groups = 1
    baseline_fc1 = block.mlp._fc1_project(x)
    baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=x)
    baseline_fc2 = block.mlp._fc2_project(baseline_hidden)
    baseline = block.mlp(x)

    block.mlp.use_ffn_fc2_input_chunked_qmm_candidate = True
    candidate_fc1 = block.mlp._fc1_project(x)
    candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=x)
    fallback_fc2 = block.mlp._fc2_project(candidate_hidden, lora=object())
    candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
    candidate = block.mlp(x)
    chunk_info = block.mlp.fc2_input_chunked_qmm_info()
    block.mlp.use_ffn_fc2_input_chunked_qmm_candidate = False
    mx.eval(
        baseline_fc1,
        candidate_fc1,
        baseline_hidden,
        candidate_hidden,
        fallback_fc2,
        baseline_fc2,
        candidate_fc2,
        baseline,
        candidate,
    )

    deltas = {
        "fc1": float(mx.max(mx.abs(baseline_fc1.astype(mx.float32) - candidate_fc1.astype(mx.float32))).item()),
        "hidden": float(
            mx.max(mx.abs(baseline_hidden.astype(mx.float32) - candidate_hidden.astype(mx.float32))).item()
        ),
        "fallback_fc2": float(
            mx.max(mx.abs(baseline_fc2.astype(mx.float32) - fallback_fc2.astype(mx.float32))).item()
        ),
        "fc2": float(mx.max(mx.abs(baseline_fc2.astype(mx.float32) - candidate_fc2.astype(mx.float32))).item()),
        "mlp": float(mx.max(mx.abs(baseline.astype(mx.float32) - candidate.astype(mx.float32))).item()),
    }
    assert candidate_fc1.shape == baseline_fc1.shape
    assert candidate_hidden.shape == baseline_hidden.shape
    assert candidate_fc2.shape == baseline_fc2.shape
    assert candidate.shape == baseline.shape
    assert candidate_fc1.dtype == baseline_fc1.dtype
    assert candidate_hidden.dtype == baseline_hidden.dtype
    assert candidate_fc2.dtype == baseline_fc2.dtype
    assert candidate.dtype == baseline.dtype
    assert chunk_info["source_is_quantized"] is True
    assert chunk_info["uses_quantized_matmul"] is True
    assert chunk_info["dense_dequantization"] is False
    assert chunk_info["total_input_groups"] == 3
    assert chunk_info["effective_chunk_groups"] == 1
    assert chunk_info["chunk_count"] == 3
    assert chunk_info["features_per_full_chunk"] == 32
    assert chunk_info["materializes_dense_weight"] is False
    assert chunk_info["learned_bias_added_once_after_partial_accumulation"] is True
    assert deltas["fc1"] == 0.0 and deltas["hidden"] == 0.0, f"chunked fc2 changed upstream: {deltas}"
    assert deltas["fallback_fc2"] == 0.0, f"LoRA fallback path unexpectedly used chunked fc2 QMM: {deltas}"
    assert deltas["fc2"] <= 4e-2 and deltas["mlp"] <= 4e-2, f"chunked fc2 bounded deltas: {deltas}"
    print(f"ffn fc2 input-chunked QMM candidate bounded for tiny quantized fc2: {deltas}")



def archived_ffn_fc2_dense_dequant_candidate_exact_for_dense_tiny_fc2():
    cfg = tiny_config()
    mx.random.seed(23)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    block = dit.blocks[0]

    assert block.mlp.use_ffn_fc2_dense_dequant_candidate is False
    baseline_fc1 = block.mlp._fc1_project(x)
    baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=x)
    baseline_fc2 = block.mlp._fc2_project(baseline_hidden)
    baseline = block.mlp(x)

    block.mlp.use_ffn_fc2_dense_dequant_candidate = True
    candidate_fc1 = block.mlp._fc1_project(x)
    candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=x)
    candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
    candidate = block.mlp(x)
    cache_info = block.mlp.fc2_dense_dequant_cache_info()
    block.mlp.use_ffn_fc2_dense_dequant_candidate = False
    block.mlp.clear_fc2_dense_dequant_cache()
    mx.eval(
        baseline_fc1,
        candidate_fc1,
        baseline_hidden,
        candidate_hidden,
        baseline_fc2,
        candidate_fc2,
        baseline,
        candidate,
    )

    deltas = {
        "fc1": float(mx.max(mx.abs(baseline_fc1.astype(mx.float32) - candidate_fc1.astype(mx.float32))).item()),
        "hidden": float(
            mx.max(mx.abs(baseline_hidden.astype(mx.float32) - candidate_hidden.astype(mx.float32))).item()
        ),
        "fc2": float(mx.max(mx.abs(baseline_fc2.astype(mx.float32) - candidate_fc2.astype(mx.float32))).item()),
        "mlp": float(mx.max(mx.abs(baseline.astype(mx.float32) - candidate.astype(mx.float32))).item()),
    }
    assert candidate_fc1.shape == baseline_fc1.shape
    assert candidate_hidden.shape == baseline_hidden.shape
    assert candidate_fc2.shape == baseline_fc2.shape
    assert candidate.shape == baseline.shape
    assert candidate_fc1.dtype == baseline_fc1.dtype
    assert candidate_hidden.dtype == baseline_hidden.dtype
    assert candidate_fc2.dtype == baseline_fc2.dtype
    assert candidate.dtype == baseline.dtype
    assert cache_info["source_is_quantized"] is False
    assert cache_info["cached"] is False
    assert all(delta == 0.0 for delta in deltas.values()), f"FFN dense-fc2 tiny deltas: {deltas}"
    print(f"ffn fc2 dense-dequant candidate exact for dense tiny fc2: {deltas}")


def test_ffn_fc2_tiled_dense_dequant_candidate_exact_for_quantized_tiny_fc2():
    cfg = tiny_config()
    mx.random.seed(24)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    block = dit.blocks[0]
    nn.quantize(
        block.mlp,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear) and path.endswith("fc2"),
    )
    mx.eval(block.mlp.parameters())

    assert block.mlp.use_ffn_fc2_tiled_dense_dequant_candidate is False
    assert block.mlp.use_ffn_fc2_dense_dequant_candidate is False
    block.mlp.ffn_fc2_tiled_output_channels = 17
    baseline_fc1 = block.mlp._fc1_project(x)
    baseline_hidden = block.mlp._swiglu_hidden(baseline_fc1, x=x)
    baseline_fc2 = block.mlp._fc2_project(baseline_hidden)
    baseline = block.mlp(x)

    block.mlp.use_ffn_fc2_tiled_dense_dequant_candidate = True
    candidate_fc1 = block.mlp._fc1_project(x)
    candidate_hidden = block.mlp._swiglu_hidden(candidate_fc1, x=x)
    candidate_fc2 = block.mlp._fc2_project(candidate_hidden)
    candidate = block.mlp(x)
    tile_info = block.mlp.fc2_tiled_dense_dequant_info()
    resident_cache_info = block.mlp.fc2_dense_dequant_cache_info()
    block.mlp.use_ffn_fc2_tiled_dense_dequant_candidate = False
    mx.eval(
        baseline_fc1,
        candidate_fc1,
        baseline_hidden,
        candidate_hidden,
        baseline_fc2,
        candidate_fc2,
        baseline,
        candidate,
    )

    deltas = {
        "fc1": float(mx.max(mx.abs(baseline_fc1.astype(mx.float32) - candidate_fc1.astype(mx.float32))).item()),
        "hidden": float(
            mx.max(mx.abs(baseline_hidden.astype(mx.float32) - candidate_hidden.astype(mx.float32))).item()
        ),
        "fc2": float(mx.max(mx.abs(baseline_fc2.astype(mx.float32) - candidate_fc2.astype(mx.float32))).item()),
        "mlp": float(mx.max(mx.abs(baseline.astype(mx.float32) - candidate.astype(mx.float32))).item()),
    }
    assert candidate_fc1.shape == baseline_fc1.shape
    assert candidate_hidden.shape == baseline_hidden.shape
    assert candidate_fc2.shape == baseline_fc2.shape
    assert candidate.shape == baseline.shape
    assert candidate_fc1.dtype == baseline_fc1.dtype
    assert candidate_hidden.dtype == baseline_hidden.dtype
    assert candidate_fc2.dtype == baseline_fc2.dtype
    assert candidate.dtype == baseline.dtype
    assert tile_info["source_is_quantized"] is True
    assert tile_info["effective_tile_output_channels"] == 17
    assert tile_info["tile_count"] == 4
    assert tile_info["max_dense_tile_nbytes"] < tile_info["full_dense_nbytes_if_resident"]
    assert tile_info["persistent_full_dense_allocation"] is False
    assert resident_cache_info["cached"] is False
    assert deltas["fc1"] == 0.0 and deltas["hidden"] == 0.0, f"FFN tiled dense-fc2 changed upstream: {deltas}"
    assert deltas["fc2"] <= 1e-6 and deltas["mlp"] <= 1e-6, f"FFN tiled dense-fc2 tiny deltas: {deltas}"
    print(f"ffn fc2 tiled dense-dequant candidate within tiny quantized tolerance: {deltas}")


def archived_ffn_hidden_tile_stream_candidate_bounded_for_quantized_tiny_ffn():
    cfg = tiny_config()
    cfg.ffn_hidden_size = 64
    mx.random.seed(25)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 7, cfg.hidden_size)).astype(mx.bfloat16)
    block = dit.blocks[0]
    nn.quantize(
        block.mlp,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear) and path in ("fc1", "fc2"),
    )
    mx.eval(block.mlp.parameters())

    assert block.mlp.use_ffn_hidden_tile_stream_candidate is False
    block.mlp.ffn_hidden_tile_groups = 1
    baseline = block.mlp(x)
    block.mlp.use_ffn_hidden_tile_stream_candidate = True
    candidate = block.mlp(x)
    info = block.mlp.hidden_tile_stream_info(1, batch_size=1, sequence_length=7)
    block.mlp.use_ffn_hidden_tile_stream_candidate = False
    mx.eval(baseline, candidate)

    diff = candidate.astype(mx.float32) - baseline.astype(mx.float32)
    max_abs = float(mx.max(mx.abs(diff)).item())
    rel_l2 = float(
        (mx.sqrt(mx.sum(diff * diff)) / mx.sqrt(mx.sum(baseline.astype(mx.float32) * baseline.astype(mx.float32)))).item()
    )
    assert candidate.shape == baseline.shape
    assert candidate.dtype == baseline.dtype
    assert info["source_is_quantized_fc2"] is True
    assert info["total_input_groups"] == 2
    assert info["effective_tile_groups"] == 1
    assert info["tile_count"] == 2
    assert info["materializes_full_fc1_gate_value"] is False
    assert info["materializes_full_swiglu_hidden"] is False
    assert info["max_tile_gate_value_nbytes"] < info["full_gate_value_nbytes_baseline"]
    assert info["max_tile_hidden_nbytes"] < info["full_hidden_nbytes_baseline"]
    assert max_abs <= 1e-4 and rel_l2 <= 1e-6, (max_abs, rel_l2)
    print(f"ffn hidden-tile stream candidate bounded for tiny quantized FFN: max_abs={max_abs} rel_l2={rel_l2}")


def archived_ffn_pre_fc1_contiguous_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(27)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    block = dit.blocks[0]

    assert block.mlp.use_ffn_pre_fc1_contiguous_candidate is False
    direct_input = materialize_ffn_input_contiguous(x, cfg.hidden_size)
    block.mlp.use_ffn_pre_fc1_contiguous_candidate = False
    baseline_fc1 = block.mlp._fc1_project(x)
    baseline = block.mlp(x)
    block.mlp.use_ffn_pre_fc1_contiguous_candidate = True
    candidate_input = block.mlp._pre_fc1_input(x)
    candidate_fc1 = block.mlp._fc1_project(x)
    candidate = block.mlp(x)
    block.mlp.use_ffn_pre_fc1_contiguous_candidate = False
    mx.eval(direct_input, candidate_input, baseline_fc1, candidate_fc1, baseline, candidate)

    deltas = {
        "input": float(mx.max(mx.abs(x.astype(mx.float32) - candidate_input.astype(mx.float32))).item()),
        "direct_input": float(mx.max(mx.abs(x.astype(mx.float32) - direct_input.astype(mx.float32))).item()),
        "fc1": float(mx.max(mx.abs(baseline_fc1.astype(mx.float32) - candidate_fc1.astype(mx.float32))).item()),
        "mlp": float(mx.max(mx.abs(baseline.astype(mx.float32) - candidate.astype(mx.float32))).item()),
    }
    assert candidate_input.shape == x.shape
    assert direct_input.shape == x.shape
    assert candidate_input.dtype == x.dtype
    assert direct_input.dtype == x.dtype
    assert candidate_fc1.shape == baseline_fc1.shape
    assert candidate_fc1.dtype == baseline_fc1.dtype
    assert candidate.shape == baseline.shape
    assert candidate.dtype == baseline.dtype
    assert all(delta == 0.0 for delta in deltas.values()), f"FFN pre-fc1 contiguous deltas: {deltas}"
    print(f"ffn pre-fc1 contiguous candidate exact: {deltas}")


def archived_ffn_pre_fc2_contiguous_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(29)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    block = dit.blocks[0]

    assert block.mlp.use_ffn_pre_fc2_contiguous_candidate is False
    fused = block.mlp._fc1_project(x)
    hidden = block.mlp._swiglu_hidden(fused)
    direct_hidden = materialize_ffn_hidden_contiguous(hidden, block.mlp._ffn)
    block.mlp.use_ffn_pre_fc2_contiguous_candidate = False
    baseline_fc2 = block.mlp._fc2_project(hidden)
    baseline = block.mlp(x)
    block.mlp.use_ffn_pre_fc2_contiguous_candidate = True
    candidate_fc2 = block.mlp._fc2_project(hidden)
    candidate = block.mlp(x)
    block.mlp.use_ffn_pre_fc2_contiguous_candidate = False
    mx.eval(hidden, direct_hidden, baseline_fc2, candidate_fc2, baseline, candidate)

    deltas = {
        "hidden": float(mx.max(mx.abs(hidden.astype(mx.float32) - direct_hidden.astype(mx.float32))).item()),
        "fc2": float(mx.max(mx.abs(baseline_fc2.astype(mx.float32) - candidate_fc2.astype(mx.float32))).item()),
        "mlp": float(mx.max(mx.abs(baseline.astype(mx.float32) - candidate.astype(mx.float32))).item()),
    }
    assert direct_hidden.shape == hidden.shape
    assert direct_hidden.dtype == hidden.dtype
    assert candidate_fc2.shape == baseline_fc2.shape
    assert candidate_fc2.dtype == baseline_fc2.dtype
    assert candidate.shape == baseline.shape
    assert candidate.dtype == baseline.dtype
    assert all(delta == 0.0 for delta in deltas.values()), f"FFN pre-fc2 contiguous deltas: {deltas}"
    print(f"ffn pre-fc2 contiguous candidate exact: {deltas}")



def archived_ffn_sequence_chunk_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(19)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 17, cfg.hidden_size))
    block = dit.blocks[0]

    assert block.mlp.use_ffn_sequence_chunk_candidate is False
    block.mlp.use_ffn_sequence_chunk_candidate = False
    baseline = block.mlp(x)
    block.mlp.ffn_sequence_chunk_size = 5
    block.mlp.use_ffn_sequence_chunk_candidate = True
    candidate = block.mlp(x)
    block.mlp.use_ffn_sequence_chunk_candidate = False
    mx.eval(baseline, candidate)

    delta = float(mx.max(mx.abs(baseline - candidate)).item())
    assert candidate.shape == baseline.shape
    assert candidate.dtype == baseline.dtype
    assert delta == 0.0, f"sequence-chunked FFN candidate changed output by {delta}"
    print(f"ffn sequence-chunked candidate exact: chunk rows 5, max delta {delta}")



def archived_ffn_subgraph_compile_candidate_exact_and_lora_fallback():
    compile_fn = getattr(mx, "compile", None)
    if compile_fn is None:
        print("ffn subgraph compile candidate skipped: mx.compile unavailable")
        return

    cfg = tiny_config()
    mx.random.seed(41)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    block = dit.blocks[0]

    class ZeroLora:
        def apply(self, name, z):
            if name == "ff.net.0.proj":
                return mx.zeros((*z.shape[:-1], 2 * cfg.ffn_hidden_size), dtype=z.dtype)
            if name == "ff.net.2":
                return mx.zeros((*z.shape[:-1], cfg.hidden_size), dtype=z.dtype)
            raise AssertionError(f"unexpected LoRA key {name}")

    assert block.mlp.use_ffn_subgraph_compile_candidate is False
    assert block.mlp.ffn_subgraph_compile_cache_info()["cached"] is False
    baseline_branch = block.mlp._ffn_baseline_subgraph(x)
    baseline = block.mlp(x)
    baseline_lora = block.mlp(x, lora=ZeroLora())

    block.mlp.use_ffn_subgraph_compile_candidate = True
    candidate = block.mlp(x)
    cache_info = block.mlp.ffn_subgraph_compile_cache_info()
    fallback_lora = block.mlp(x, lora=ZeroLora())
    block.mlp.use_ffn_subgraph_compile_candidate = False
    block.mlp.clear_ffn_subgraph_compile_cache()
    mx.eval(baseline_branch, baseline, baseline_lora, candidate, fallback_lora)

    deltas = {
        "branch_vs_default": float(mx.max(mx.abs(baseline_branch.astype(mx.float32) - baseline.astype(mx.float32))).item()),
        "compiled": float(mx.max(mx.abs(baseline.astype(mx.float32) - candidate.astype(mx.float32))).item()),
        "lora_fallback": float(mx.max(mx.abs(baseline_lora.astype(mx.float32) - fallback_lora.astype(mx.float32))).item()),
    }
    assert candidate.shape == baseline.shape
    assert candidate.dtype == baseline.dtype
    assert fallback_lora.shape == baseline_lora.shape
    assert fallback_lora.dtype == baseline_lora.dtype
    assert cache_info["mx_compile_available"] is True
    assert cache_info["cached"] is True
    assert block.mlp.ffn_subgraph_compile_cache_info()["cached"] is False
    assert all(delta <= 1e-5 for delta in deltas.values()), f"FFN subgraph compile deltas: {deltas}"
    print(f"ffn subgraph compile candidate exact within tiny tolerance: {deltas}")



def archived_ffn_fused_swiglu_metal_candidate_bounded():
    if not has_swiglu_from_fused_metal():
        print("ffn fused SwiGLU Metal candidate skipped: mx.fast.metal_kernel unavailable")
        return

    cfg = tiny_config()
    mx.random.seed(23)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    x = mx.random.normal((1, 13, cfg.hidden_size))
    block = dit.blocks[0]

    assert block.mlp.use_ffn_metal_swiglu_candidate is False
    fused = block.mlp._fc1_project(x)
    baseline_hidden = block.mlp._swiglu_hidden(fused)
    metal_hidden = swiglu_from_fused_metal(fused, block.mlp._ffn)
    block.mlp.use_ffn_metal_swiglu_candidate = False
    baseline = block.mlp(x)
    block.mlp.use_ffn_metal_swiglu_candidate = True
    candidate = block.mlp(x)
    block.mlp.use_ffn_metal_swiglu_candidate = False
    mx.eval(baseline_hidden, metal_hidden, baseline, candidate)

    deltas = {
        "hidden": float(mx.max(mx.abs(baseline_hidden - metal_hidden)).item()),
        "mlp": float(mx.max(mx.abs(baseline - candidate)).item()),
    }
    assert metal_hidden.shape == baseline_hidden.shape
    assert metal_hidden.dtype == baseline_hidden.dtype
    assert candidate.shape == baseline.shape
    assert candidate.dtype == baseline.dtype
    assert deltas["hidden"] == 0.0 and deltas["mlp"] == 0.0, f"Metal SwiGLU deltas: {deltas}"

    bf16_fused = mx.random.normal((1, 7, 2 * block.mlp._ffn)).astype(mx.bfloat16)
    bf16_gate, bf16_value = bf16_fused[..., : block.mlp._ffn], bf16_fused[..., block.mlp._ffn :]
    bf16_baseline = nn.silu(bf16_gate) * bf16_value
    bf16_candidate = swiglu_from_fused_metal(bf16_fused, block.mlp._ffn)
    mx.eval(bf16_baseline, bf16_candidate)
    bf16_delta = float(mx.max(mx.abs(bf16_baseline.astype(mx.float32) - bf16_candidate.astype(mx.float32))).item())
    assert bf16_delta == 0.0, f"bf16 Metal SwiGLU delta {bf16_delta}"
    print(f"ffn fused SwiGLU Metal candidate exact: {deltas}, bf16 hidden max delta {bf16_delta}")



def archived_adaln_packed_gather_candidate_exact_and_projection_already_unique():
    cfg = tiny_config()
    mx.random.seed(41)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    sequence = 17
    x = mx.random.normal((1, sequence, cfg.hidden_size))
    temb = mx.random.normal((3, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 3, 5, 7, 2, 4, 6], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(sequence) % 3, mx.arange(sequence) % 5, mx.arange(sequence) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)

    assert block.use_packed_adaln_gather_candidate is False
    # The projection has already been reduced to the unique timestep table expanded by modality;
    # repeated per-token determinant rows appear only at the gather/materialization boundary.
    assert modulation[0].shape == (int(temb.shape[0]) * MODALITY_NUM, cfg.hidden_size)
    assert modulation[0].shape[0] < adaln_indices.shape[0]

    baseline_rows = tuple(tensor[adaln_indices] for tensor in modulation)
    packed_rows = gather_packed_modulation_rows(modulation, adaln_indices)
    block.use_packed_adaln_gather_candidate = False
    baseline = block(x, modulation, adaln_indices, rotary)
    block.use_packed_adaln_gather_candidate = True
    candidate = block(x, modulation, adaln_indices, rotary)
    block.use_packed_adaln_gather_candidate = False
    mx.eval(*(baseline_rows + packed_rows), baseline, candidate)

    row_deltas = [
        float(mx.max(mx.abs(reference - gathered)).item())
        for reference, gathered in zip(baseline_rows, packed_rows)
    ]
    block_delta = float(mx.max(mx.abs(baseline - candidate)).item())
    assert all(delta == 0.0 for delta in row_deltas), f"packed AdaLN gather row deltas: {row_deltas}"
    assert block_delta == 0.0, f"packed AdaLN gather candidate changed block output by {block_delta}"
    print(
        "adaln packed gather candidate exact: "
        f"table rows {modulation[0].shape[0]}, sequence rows {adaln_indices.shape[0]}, "
        f"max row delta {max(row_deltas)}, block delta {block_delta}"
    )



def archived_indexed_gated_residual_metal_candidate_bounded():
    if not has_indexed_gated_residual_metal():
        print("indexed gated-residual Metal candidate skipped: mx.fast.metal_kernel unavailable")
        return

    cfg = tiny_config()
    mx.random.seed(43)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    sequence = 13
    x = mx.random.normal((1, sequence, cfg.hidden_size))
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(sequence) % 3, mx.arange(sequence) % 5, mx.arange(sequence) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

    assert block.use_indexed_gated_residual_metal_candidate is False
    h = block.norm1(x) * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
    attn_out = block.attn(h, rotary)
    baseline_msa = x + gate_msa[adaln_indices] * attn_out
    metal_msa = indexed_gated_residual_metal(x, gate_msa, adaln_indices, attn_out)
    h2 = block.norm2(baseline_msa) * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]
    mlp_out = block.mlp(h2)
    baseline_mlp = baseline_msa + gate_mlp[adaln_indices] * mlp_out
    metal_mlp = indexed_gated_residual_metal(baseline_msa, gate_mlp, adaln_indices, mlp_out)

    block.use_indexed_gated_residual_metal_candidate = False
    baseline_block = block(x, modulation, adaln_indices, rotary)
    block.use_indexed_gated_residual_metal_candidate = True
    candidate_block = block(x, modulation, adaln_indices, rotary)
    block.use_indexed_gated_residual_metal_candidate = False
    mx.eval(baseline_msa, metal_msa, baseline_mlp, metal_mlp, baseline_block, candidate_block)

    deltas = {
        "msa_residual": float(mx.max(mx.abs(baseline_msa - metal_msa)).item()),
        "mlp_residual": float(mx.max(mx.abs(baseline_mlp - metal_mlp)).item()),
        "block": float(mx.max(mx.abs(baseline_block - candidate_block)).item()),
    }
    assert all(delta <= 1e-5 for delta in deltas.values()), f"indexed gated residual deltas: {deltas}"

    base_bf16 = mx.random.normal((2, 9, cfg.hidden_size)).astype(mx.bfloat16)
    branch_bf16 = mx.random.normal((2, 9, cfg.hidden_size)).astype(mx.bfloat16)
    gate_bf16 = mx.random.normal((6, cfg.hidden_size)).astype(mx.bfloat16)
    idx_bf16 = mx.array([0, 1, 2, 3, 4, 5, 0, 2, 4], dtype=mx.int32)
    bf16_baseline = base_bf16 + gate_bf16[idx_bf16] * branch_bf16
    bf16_metal = indexed_gated_residual_metal(base_bf16, gate_bf16, idx_bf16, branch_bf16)
    mx.eval(bf16_baseline, bf16_metal)
    bf16_delta = float(mx.max(mx.abs(bf16_baseline.astype(mx.float32) - bf16_metal.astype(mx.float32))).item())
    assert bf16_delta <= 4e-2, f"bf16 indexed gated residual bounded delta {bf16_delta}"
    print(f"indexed gated-residual Metal candidate bounded: {deltas}, bf16 max delta {bf16_delta}")



def test_refined_text_cache_candidate_exact_and_ignores_text_embeds():
    dit, _cfg, args = test_forward_shapes()
    video, audio, text, timestep, ts_i, tags, pos, video_i, audio_i, text_i = args

    baseline_v, baseline_a = dit(*args)
    refined_text = dit.precompute_text_conditioning(text)
    cached_v, cached_a = dit(
        video,
        audio,
        None,
        timestep,
        ts_i,
        tags,
        pos,
        video_i,
        audio_i,
        text_i,
        refined_text=refined_text,
    )
    bogus_text_v, bogus_text_a = dit(
        video,
        audio,
        mx.zeros_like(text),
        timestep,
        ts_i,
        tags,
        pos,
        video_i,
        audio_i,
        text_i,
        refined_text=refined_text,
    )
    mx.eval(baseline_v, baseline_a, refined_text, cached_v, cached_a, bogus_text_v, bogus_text_a)

    dv = float(mx.max(mx.abs(baseline_v - cached_v)).item())
    da = float(mx.max(mx.abs(baseline_a - cached_a)).item())
    rel_v = 0.0 if dv == 0.0 else dv / float(mx.max(mx.abs(baseline_v)).item())
    rel_a = 0.0 if da == 0.0 else da / float(mx.max(mx.abs(baseline_a)).item())
    bogus_dv = float(mx.max(mx.abs(cached_v - bogus_text_v)).item())
    bogus_da = float(mx.max(mx.abs(cached_a - bogus_text_a)).item())
    assert dv == 0.0 and da == 0.0, f"refined text cache mismatch: video {dv}, audio {da}"
    assert bogus_dv == 0.0 and bogus_da == 0.0, "cached path unexpectedly read text_embeds"
    print(
        "refined text cache exact: "
        f"video max_abs {dv}, max_rel {rel_v}; audio max_abs {da}, max_rel {rel_a}"
    )


def archived_attention_qkv_2d_projection_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(11)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    x = mx.random.normal((1, 11, cfg.hidden_size))
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(11) % 3, mx.arange(11) % 5, mx.arange(11) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    block.attn.use_qkv_2d_projection_candidate = False
    baseline_qkv_projection = block.attn._qkv_project(h)
    baseline_qkv_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_qkv_2d_projection_candidate = True
    candidate_qkv_projection = block.attn._qkv_project(h)
    candidate_qkv_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_qkv_2d_projection_candidate = False
    mx.eval(baseline_qkv_projection, candidate_qkv_projection, baseline_qkv_block, candidate_qkv_block)

    assert candidate_qkv_projection.shape == baseline_qkv_projection.shape
    assert candidate_qkv_projection.dtype == baseline_qkv_projection.dtype
    qkv_proj_delta = float(mx.max(mx.abs(baseline_qkv_projection - candidate_qkv_projection)).item())
    qkv_block_delta = float(mx.max(mx.abs(baseline_qkv_block - candidate_qkv_block)).item())
    assert qkv_proj_delta == 0.0, f"2D qkv projection candidate changed projection by {qkv_proj_delta}"
    assert qkv_block_delta == 0.0, f"2D qkv projection candidate changed block by {qkv_block_delta}"

    out_input = mx.random.normal((1, 11, cfg.inner_dim)).astype(h.dtype)
    block.attn.use_out_2d_projection_candidate = False
    baseline_out_projection = block.attn._out_project(out_input)
    baseline_out_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_out_2d_projection_candidate = True
    candidate_out_projection = block.attn._out_project(out_input)
    candidate_out_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_out_2d_projection_candidate = False
    mx.eval(baseline_out_projection, candidate_out_projection, baseline_out_block, candidate_out_block)

    assert candidate_out_projection.shape == baseline_out_projection.shape
    assert candidate_out_projection.dtype == baseline_out_projection.dtype
    out_proj_delta = float(mx.max(mx.abs(baseline_out_projection - candidate_out_projection)).item())
    out_block_delta = float(mx.max(mx.abs(baseline_out_block - candidate_out_block)).item())
    assert out_proj_delta == 0.0, f"2D out projection candidate changed projection by {out_proj_delta}"
    assert out_block_delta == 0.0, f"2D out projection candidate changed block by {out_block_delta}"
    print(
        "attention 2D projection candidates exact: "
        f"qkv projection {qkv_proj_delta}, qkv block {qkv_block_delta}, "
        f"out projection {out_proj_delta}, out block {out_block_delta}"
    )


def test_attention_out_dense_dequant_candidate_exact_for_quantized_tiny_out_proj():
    cfg = tiny_config()
    mx.random.seed(29)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    nn.quantize(
        block.attn,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear) and path.endswith("out_proj"),
    )
    mx.eval(block.attn.parameters())

    x = mx.random.normal((1, 11, cfg.hidden_size)).astype(mx.bfloat16)
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(11) % 3, mx.arange(11) % 5, mx.arange(11) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    out_input = mx.random.normal((1, 11, cfg.inner_dim)).astype(mx.bfloat16)

    assert block.attn.use_out_dense_dequant_candidate is False
    block.attn.use_out_dense_dequant_candidate = False
    baseline_projection = block.attn._out_project(out_input)
    baseline_block = block(x, modulation, adaln_indices, rotary)

    block.attn.use_out_dense_dequant_candidate = True
    fallback_projection = block.attn._out_project(out_input, lora=object())
    fallback_cache_info = block.attn.out_dense_dequant_cache_info()
    candidate_projection = block.attn._out_project(out_input)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    cache_info = block.attn.out_dense_dequant_cache_info()
    block.attn.use_out_dense_dequant_candidate = False
    block.attn.clear_out_dense_dequant_cache()
    mx.eval(baseline_projection, fallback_projection, candidate_projection, baseline_block, candidate_block)

    deltas = {
        "projection": float(
            mx.max(mx.abs(baseline_projection.astype(mx.float32) - candidate_projection.astype(mx.float32))).item()
        ),
        "fallback_projection": float(
            mx.max(mx.abs(baseline_projection.astype(mx.float32) - fallback_projection.astype(mx.float32))).item()
        ),
        "block": float(mx.max(mx.abs(baseline_block.astype(mx.float32) - candidate_block.astype(mx.float32))).item()),
    }
    assert candidate_projection.shape == baseline_projection.shape
    assert candidate_projection.dtype == baseline_projection.dtype
    assert candidate_block.shape == baseline_block.shape
    assert candidate_block.dtype == baseline_block.dtype
    assert fallback_cache_info["source_is_quantized"] is True
    assert fallback_cache_info["cached"] is False
    assert cache_info["source_is_quantized"] is True
    assert cache_info["cached"] is True
    assert cache_info["dense_nbytes"] > cache_info["source_weight_nbytes"]
    assert deltas["fallback_projection"] == 0.0, f"LoRA fallback path unexpectedly used dense cache: {deltas}"
    assert deltas["projection"] <= 1e-6 and deltas["block"] <= 1e-6, (
        f"attention out dense-dequant tiny deltas: {deltas}"
    )
    print(f"attention out dense-dequant candidate within tiny quantized tolerance: {deltas}")


def test_attention_out_tiled_dense_dequant_candidate_exact_for_quantized_tiny_out_proj():
    cfg = tiny_config()
    mx.random.seed(32)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    nn.quantize(
        block.attn,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear) and path.endswith("out_proj"),
    )
    mx.eval(block.attn.parameters())

    x = mx.random.normal((1, 11, cfg.hidden_size)).astype(mx.bfloat16)
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(11) % 3, mx.arange(11) % 5, mx.arange(11) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    out_input = mx.random.normal((1, 11, cfg.inner_dim)).astype(mx.bfloat16)

    assert block.attn.use_out_tiled_dense_dequant_candidate is False
    assert block.attn.use_out_dense_dequant_candidate is False
    block.attn.out_tiled_output_channels = 17
    block.attn.use_out_tiled_dense_dequant_candidate = False
    baseline_projection = block.attn._out_project(out_input)
    baseline_block = block(x, modulation, adaln_indices, rotary)

    block.attn.use_out_tiled_dense_dequant_candidate = True
    fallback_projection = block.attn._out_project(out_input, lora=object())
    candidate_projection = block.attn._out_project(out_input)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    tile_info = block.attn.out_tiled_dense_dequant_info()
    block.attn.use_out_tiled_dense_dequant_candidate = False
    mx.eval(baseline_projection, fallback_projection, candidate_projection, baseline_block, candidate_block)

    deltas = {
        "projection": float(
            mx.max(mx.abs(baseline_projection.astype(mx.float32) - candidate_projection.astype(mx.float32))).item()
        ),
        "fallback_projection": float(
            mx.max(mx.abs(baseline_projection.astype(mx.float32) - fallback_projection.astype(mx.float32))).item()
        ),
        "block": float(mx.max(mx.abs(baseline_block.astype(mx.float32) - candidate_block.astype(mx.float32))).item()),
    }
    assert candidate_projection.shape == baseline_projection.shape
    assert candidate_projection.dtype == baseline_projection.dtype
    assert fallback_projection.shape == baseline_projection.shape
    assert fallback_projection.dtype == baseline_projection.dtype
    assert candidate_block.shape == baseline_block.shape
    assert candidate_block.dtype == baseline_block.dtype
    assert tile_info["source_is_quantized"] is True
    assert tile_info["effective_tile_output_channels"] == 17
    assert tile_info["tile_count"] == 4
    assert tile_info["max_dense_tile_nbytes"] < tile_info["full_dense_nbytes_if_resident"]
    assert tile_info["persistent_full_dense_allocation"] is False
    quantized_dense_tolerance = 2e-6
    assert deltas["fallback_projection"] == 0.0, f"LoRA fallback path unexpectedly used tiled out_proj: {deltas}"
    assert all(delta <= quantized_dense_tolerance for delta in deltas.values()), (
        f"attention out tiled dense-dequant deltas: {deltas}"
    )
    print(f"attention out tiled dense-dequant candidate within tiny quantized tolerance: {deltas}")


def test_attention_qkv_tiled_dense_dequant_candidate_exact_for_quantized_tiny_qkv():
    cfg = tiny_config()
    mx.random.seed(30)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    nn.quantize(
        block.attn,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear) and path.endswith("qkv_proj"),
    )
    mx.eval(block.attn.parameters())

    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    assert block.attn.use_qkv_tiled_dense_dequant_candidate is False
    assert block.attn.use_qkv_2d_projection_candidate is False
    block.attn.qkv_tiled_output_channels = 31
    block.attn.use_qkv_tiled_dense_dequant_candidate = False
    baseline_projection = block.attn._qkv_project(h)
    baseline_raw_qkv = baseline_projection.reshape(1, 13, cfg.num_attention_heads, 3, cfg.attention_head_dim)
    baseline_q, baseline_k, baseline_v = block.attn._qkv_sdpa_tensors(h)
    baseline_block = block(x, modulation, adaln_indices, rotary)

    block.attn.use_qkv_tiled_dense_dequant_candidate = True
    fallback_projection = block.attn._qkv_project(h, lora=object())
    candidate_projection = block.attn._qkv_project(h)
    candidate_raw_qkv = candidate_projection.reshape(1, 13, cfg.num_attention_heads, 3, cfg.attention_head_dim)
    candidate_q, candidate_k, candidate_v = block.attn._qkv_sdpa_tensors(h)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    tile_info = block.attn.qkv_tiled_dense_dequant_info()
    block.attn.use_qkv_tiled_dense_dequant_candidate = False
    mx.eval(
        baseline_projection,
        fallback_projection,
        candidate_projection,
        baseline_raw_qkv,
        candidate_raw_qkv,
        baseline_q,
        baseline_k,
        baseline_v,
        candidate_q,
        candidate_k,
        candidate_v,
        baseline_block,
        candidate_block,
    )

    raw_deltas = {
        "q": float(
            mx.max(mx.abs(baseline_raw_qkv[:, :, :, 0].astype(mx.float32) - candidate_raw_qkv[:, :, :, 0].astype(mx.float32))).item()
        ),
        "k": float(
            mx.max(mx.abs(baseline_raw_qkv[:, :, :, 1].astype(mx.float32) - candidate_raw_qkv[:, :, :, 1].astype(mx.float32))).item()
        ),
        "v": float(
            mx.max(mx.abs(baseline_raw_qkv[:, :, :, 2].astype(mx.float32) - candidate_raw_qkv[:, :, :, 2].astype(mx.float32))).item()
        ),
    }
    deltas = {
        "projection": float(
            mx.max(mx.abs(baseline_projection.astype(mx.float32) - candidate_projection.astype(mx.float32))).item()
        ),
        "fallback_projection": float(
            mx.max(mx.abs(baseline_projection.astype(mx.float32) - fallback_projection.astype(mx.float32))).item()
        ),
        "sdpa_q": float(mx.max(mx.abs(baseline_q.astype(mx.float32) - candidate_q.astype(mx.float32))).item()),
        "sdpa_k": float(mx.max(mx.abs(baseline_k.astype(mx.float32) - candidate_k.astype(mx.float32))).item()),
        "sdpa_v": float(mx.max(mx.abs(baseline_v.astype(mx.float32) - candidate_v.astype(mx.float32))).item()),
        "block": float(mx.max(mx.abs(baseline_block.astype(mx.float32) - candidate_block.astype(mx.float32))).item()),
    }
    assert candidate_projection.shape == baseline_projection.shape
    assert candidate_projection.dtype == baseline_projection.dtype
    assert fallback_projection.shape == baseline_projection.shape
    assert fallback_projection.dtype == baseline_projection.dtype
    assert candidate_raw_qkv.shape == baseline_raw_qkv.shape
    assert candidate_q.shape == baseline_q.shape
    assert candidate_k.shape == baseline_k.shape
    assert candidate_v.shape == baseline_v.shape
    assert candidate_q.dtype == baseline_q.dtype
    assert candidate_k.dtype == baseline_k.dtype
    assert candidate_v.dtype == baseline_v.dtype
    assert candidate_block.shape == baseline_block.shape
    assert candidate_block.dtype == baseline_block.dtype
    assert tile_info["source_is_quantized"] is True
    assert tile_info["effective_tile_output_channels"] == 31
    assert tile_info["tile_count"] == 7
    assert tile_info["max_dense_tile_nbytes"] < tile_info["full_dense_nbytes_if_resident"]
    assert tile_info["persistent_full_dense_allocation"] is False
    quantized_dense_tolerance = 2e-6
    assert deltas["fallback_projection"] == 0.0, f"LoRA fallback path unexpectedly used tiled qkv: {deltas}"
    assert all(delta <= quantized_dense_tolerance for delta in raw_deltas.values()), f"raw qkv row-order deltas: {raw_deltas}"
    assert all(delta <= quantized_dense_tolerance for delta in deltas.values()), f"attention qkv tiled dense-dequant deltas: {deltas}"
    print(f"attention qkv tiled dense-dequant candidate within tiny quantized tolerance: {deltas}, raw {raw_deltas}")



def test_dense_dequant_opt_in_profile_exact_for_quantized_tiny_block():
    cfg = tiny_config()
    mx.random.seed(41)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    nn.quantize(
        block,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear)
        and path.endswith(("attn.qkv_proj", "attn.out_proj", "mlp.fc2")),
    )
    mx.eval(block.parameters())

    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)

    baseline = block(x, modulation, adaln_indices, rotary)

    selected = apply_dense_dequant_profile_to_block(
        block,
        DENSE_DEQUANT_PROFILE_QKV_ONLY_TILED,
        attention_qkv_tile_size=31,
        ffn_fc2_tile_size=17,
        attention_out_tile_size=19,
    )
    qkv_only = block(x, modulation, adaln_indices, rotary)
    assert selected == DENSE_DEQUANT_PROFILE_QKV_ONLY_TILED
    assert block.attn.use_qkv_tiled_dense_dequant_candidate is True
    assert block.mlp.use_ffn_fc2_tiled_dense_dequant_candidate is False
    assert block.attn.use_out_dense_dequant_candidate is False
    assert block.attn.use_out_tiled_dense_dequant_candidate is False

    selected = apply_dense_dequant_profile_to_block(
        block,
        DENSE_DEQUANT_PROFILE_FFN_FC2_TILED,
        attention_qkv_tile_size=31,
        ffn_fc2_tile_size=17,
        attention_out_tile_size=19,
    )
    fc2_only = block(x, modulation, adaln_indices, rotary)
    fc2_tile_info = block.mlp.fc2_tiled_dense_dequant_info()
    assert selected == DENSE_DEQUANT_PROFILE_FFN_FC2_TILED
    assert block.attn.use_qkv_tiled_dense_dequant_candidate is False
    assert block.mlp.use_ffn_fc2_tiled_dense_dequant_candidate is True
    assert block.attn.use_out_dense_dequant_candidate is False
    assert block.attn.use_out_tiled_dense_dequant_candidate is False
    assert fc2_tile_info["effective_tile_output_channels"] == 17

    selected = apply_dense_dequant_profile_to_block(
        block,
        DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
        attention_qkv_tile_size=31,
        ffn_fc2_tile_size=17,
        attention_out_tile_size=19,
    )
    resident = block(x, modulation, adaln_indices, rotary)
    resident_cache = block.attn.out_dense_dequant_cache_info()
    assert selected == DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT
    assert block.attn.use_qkv_tiled_dense_dequant_candidate is True
    assert block.mlp.use_ffn_fc2_tiled_dense_dequant_candidate is True
    assert block.attn.use_out_dense_dequant_candidate is True
    assert block.attn.use_out_tiled_dense_dequant_candidate is False

    selected = apply_dense_dequant_profile_to_block(
        block,
        DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED,
        attention_qkv_tile_size=31,
        ffn_fc2_tile_size=17,
        attention_out_tile_size=19,
    )
    tiled = block(x, modulation, adaln_indices, rotary)
    assert selected == DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED
    assert block.attn.use_qkv_tiled_dense_dequant_candidate is True
    assert block.mlp.use_ffn_fc2_tiled_dense_dequant_candidate is True
    assert block.attn.use_out_dense_dequant_candidate is False
    assert block.attn.use_out_tiled_dense_dequant_candidate is True

    selected = apply_dense_dequant_profile_to_block(block, DENSE_DEQUANT_PROFILE_OFF)
    off = block(x, modulation, adaln_indices, rotary)
    assert selected == DENSE_DEQUANT_PROFILE_OFF
    assert block.attn.use_qkv_tiled_dense_dequant_candidate is False
    assert block.mlp.use_ffn_fc2_tiled_dense_dequant_candidate is False
    assert block.attn.use_out_dense_dequant_candidate is False
    assert block.attn.use_out_tiled_dense_dequant_candidate is False
    mx.eval(baseline, qkv_only, fc2_only, resident, tiled, off)

    deltas = {
        "qkv_only": float(mx.max(mx.abs(baseline.astype(mx.float32) - qkv_only.astype(mx.float32))).item()),
        "fc2_only": float(mx.max(mx.abs(baseline.astype(mx.float32) - fc2_only.astype(mx.float32))).item()),
        "resident": float(mx.max(mx.abs(baseline.astype(mx.float32) - resident.astype(mx.float32))).item()),
        "tiled": float(mx.max(mx.abs(baseline.astype(mx.float32) - tiled.astype(mx.float32))).item()),
        "off": float(mx.max(mx.abs(baseline.astype(mx.float32) - off.astype(mx.float32))).item()),
    }
    assert qkv_only.shape == baseline.shape and fc2_only.shape == baseline.shape and resident.shape == baseline.shape and tiled.shape == baseline.shape and off.shape == baseline.shape
    assert qkv_only.dtype == baseline.dtype and fc2_only.dtype == baseline.dtype and resident.dtype == baseline.dtype and tiled.dtype == baseline.dtype and off.dtype == baseline.dtype
    assert resident_cache["cached"] is True
    assert deltas["off"] == 0.0
    assert deltas["qkv_only"] <= 1e-5 and deltas["fc2_only"] <= 1e-5 and deltas["resident"] <= 1e-5 and deltas["tiled"] <= 1e-5, f"dense-dequant profile deltas: {deltas}"
    print(f"dense-dequant opt-in profiles exact for tiny quantized block: {deltas}")



def archived_attention_qkv_input_chunked_qmm_candidate_bounded_for_quantized_tiny_qkv():
    cfg = tiny_config()
    mx.random.seed(34)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    nn.quantize(
        block.attn,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear) and path.endswith("qkv_proj"),
    )
    block.attn.qkv_proj.scales = block.attn.qkv_proj.scales.astype(mx.bfloat16)
    block.attn.qkv_proj.biases = block.attn.qkv_proj.biases.astype(mx.bfloat16)
    mx.eval(block.attn.parameters())

    h = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    assert block.attn.use_qkv_input_chunked_qmm_candidate is False
    block.attn.qkv_input_chunk_groups = 1

    baseline_projection = block.attn._qkv_project(h)
    baseline_raw_qkv = baseline_projection.reshape(1, 13, cfg.num_attention_heads, 3, cfg.attention_head_dim)
    baseline_q, baseline_k, baseline_v = block.attn._qkv_sdpa_tensors(h)

    block.attn.use_qkv_input_chunked_qmm_candidate = True
    fallback_projection = block.attn._qkv_project(h, lora=object())
    candidate_projection = block.attn._qkv_project(h)
    candidate_raw_qkv = candidate_projection.reshape(1, 13, cfg.num_attention_heads, 3, cfg.attention_head_dim)
    candidate_q, candidate_k, candidate_v = block.attn._qkv_sdpa_tensors(h)
    chunk_info = block.attn.qkv_input_chunked_qmm_info()
    block.attn.use_qkv_input_chunked_qmm_candidate = False
    mx.eval(
        baseline_projection,
        fallback_projection,
        candidate_projection,
        baseline_raw_qkv,
        candidate_raw_qkv,
        baseline_q,
        baseline_k,
        baseline_v,
        candidate_q,
        candidate_k,
        candidate_v,
    )

    raw_deltas = {
        "q": float(
            mx.max(mx.abs(baseline_raw_qkv[:, :, :, 0].astype(mx.float32) - candidate_raw_qkv[:, :, :, 0].astype(mx.float32))).item()
        ),
        "k": float(
            mx.max(mx.abs(baseline_raw_qkv[:, :, :, 1].astype(mx.float32) - candidate_raw_qkv[:, :, :, 1].astype(mx.float32))).item()
        ),
        "v": float(
            mx.max(mx.abs(baseline_raw_qkv[:, :, :, 2].astype(mx.float32) - candidate_raw_qkv[:, :, :, 2].astype(mx.float32))).item()
        ),
    }
    deltas = {
        "projection": float(
            mx.max(mx.abs(baseline_projection.astype(mx.float32) - candidate_projection.astype(mx.float32))).item()
        ),
        "fallback_projection": float(
            mx.max(mx.abs(baseline_projection.astype(mx.float32) - fallback_projection.astype(mx.float32))).item()
        ),
        "sdpa_q": float(mx.max(mx.abs(baseline_q.astype(mx.float32) - candidate_q.astype(mx.float32))).item()),
        "sdpa_k": float(mx.max(mx.abs(baseline_k.astype(mx.float32) - candidate_k.astype(mx.float32))).item()),
        "sdpa_v": float(mx.max(mx.abs(baseline_v.astype(mx.float32) - candidate_v.astype(mx.float32))).item()),
    }
    assert candidate_projection.shape == baseline_projection.shape
    assert candidate_projection.dtype == baseline_projection.dtype
    assert fallback_projection.shape == baseline_projection.shape
    assert fallback_projection.dtype == baseline_projection.dtype
    assert candidate_raw_qkv.shape == baseline_raw_qkv.shape
    assert candidate_q.shape == baseline_q.shape
    assert candidate_k.shape == baseline_k.shape
    assert candidate_v.shape == baseline_v.shape
    assert candidate_q.dtype == baseline_q.dtype
    assert candidate_k.dtype == baseline_k.dtype
    assert candidate_v.dtype == baseline_v.dtype
    assert chunk_info["source_is_quantized"] is True
    assert chunk_info["uses_quantized_matmul"] is True
    assert chunk_info["dense_dequantization"] is False
    assert chunk_info["total_input_groups"] == 2
    assert chunk_info["effective_chunk_groups"] == 1
    assert chunk_info["chunk_count"] == 2
    assert chunk_info["features_per_full_chunk"] == 32
    assert chunk_info["materializes_dense_weight"] is False
    assert chunk_info["output_features_match_qkv_contract"] is True
    assert deltas["fallback_projection"] == 0.0, f"LoRA fallback path unexpectedly used chunked qkv QMM: {deltas}"
    assert raw_deltas["v"] == deltas["sdpa_v"], f"v layout should only transpose raw v rows: {raw_deltas}, {deltas}"
    assert deltas["projection"] <= 5e-2 and all(delta <= 5e-2 for delta in raw_deltas.values()), (
        f"qkv input-chunked QMM raw deltas out of bounded range: {deltas}, raw {raw_deltas}"
    )
    assert deltas["sdpa_q"] <= 8e-2 and deltas["sdpa_k"] <= 8e-2 and deltas["sdpa_v"] <= 5e-2, (
        f"qkv input-chunked QMM SDPA tensor deltas out of bounded range: {deltas}, raw {raw_deltas}"
    )
    print(f"attention qkv input-chunked QMM candidate bounded for tiny quantized qkv: {deltas}, raw {raw_deltas}")



def archived_attention_pre_qkv_contiguous_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(14)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    assert block.attn.use_pre_qkv_contiguous_candidate is False
    direct_input = materialize_attention_input_contiguous(h, cfg.hidden_size)
    block.attn.use_pre_qkv_contiguous_candidate = False
    baseline_qkv_projection = block.attn._qkv_project(h)
    baseline_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_pre_qkv_contiguous_candidate = True
    candidate_input = block.attn._pre_qkv_input(h)
    candidate_qkv_projection = block.attn._qkv_project(h)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_pre_qkv_contiguous_candidate = False
    mx.eval(direct_input, candidate_input, baseline_qkv_projection, candidate_qkv_projection, baseline_block, candidate_block)

    deltas = {
        "input": float(mx.max(mx.abs(h.astype(mx.float32) - candidate_input.astype(mx.float32))).item()),
        "direct_input": float(mx.max(mx.abs(h.astype(mx.float32) - direct_input.astype(mx.float32))).item()),
        "qkv": float(mx.max(mx.abs(baseline_qkv_projection - candidate_qkv_projection)).item()),
        "block": float(mx.max(mx.abs(baseline_block - candidate_block)).item()),
    }
    assert candidate_input.shape == h.shape
    assert direct_input.shape == h.shape
    assert candidate_input.dtype == h.dtype
    assert direct_input.dtype == h.dtype
    assert candidate_qkv_projection.shape == baseline_qkv_projection.shape
    assert candidate_qkv_projection.dtype == baseline_qkv_projection.dtype
    assert candidate_block.shape == baseline_block.shape
    assert candidate_block.dtype == baseline_block.dtype
    assert all(delta == 0.0 for delta in deltas.values()), f"pre-QKV contiguous deltas: {deltas}"
    print(f"attention pre-QKV contiguous candidate exact: {deltas}")



def archived_attention_qkv_pretranspose_layout_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(13)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    x = mx.random.normal((1, 13, cfg.hidden_size))
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    block.attn.use_qkv_pretranspose_layout_candidate = False
    baseline_q, baseline_k, baseline_v = block.attn._qkv_sdpa_tensors(h)
    baseline_attn = block.attn(h, rotary)
    baseline_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_qkv_pretranspose_layout_candidate = True
    candidate_q, candidate_k, candidate_v = block.attn._qkv_sdpa_tensors(h)
    candidate_attn = block.attn(h, rotary)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_qkv_pretranspose_layout_candidate = False
    mx.eval(
        baseline_q,
        baseline_k,
        baseline_v,
        baseline_attn,
        baseline_block,
        candidate_q,
        candidate_k,
        candidate_v,
        candidate_attn,
        candidate_block,
    )

    assert candidate_q.shape == baseline_q.shape
    assert candidate_k.shape == baseline_k.shape
    assert candidate_v.shape == baseline_v.shape
    assert candidate_q.dtype == baseline_q.dtype
    assert candidate_k.dtype == baseline_k.dtype
    assert candidate_v.dtype == baseline_v.dtype
    deltas = {
        "q": float(mx.max(mx.abs(baseline_q - candidate_q)).item()),
        "k": float(mx.max(mx.abs(baseline_k - candidate_k)).item()),
        "v": float(mx.max(mx.abs(baseline_v - candidate_v)).item()),
        "attn": float(mx.max(mx.abs(baseline_attn - candidate_attn)).item()),
        "block": float(mx.max(mx.abs(baseline_block - candidate_block)).item()),
    }
    assert all(delta == 0.0 for delta in deltas.values()), f"pretranspose deltas: {deltas}"
    print(f"attention qkv pretranspose layout candidate exact: {deltas}")


def archived_attention_pre_sdpa_contiguous_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(17)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    x = mx.random.normal((1, 13, cfg.hidden_size))
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    assert block.attn.use_pre_sdpa_contiguous_candidate is False
    block.attn.use_pre_sdpa_contiguous_candidate = False
    q, k, v = block.attn._qkv_sdpa_tensors(h)
    q = apply_rotary(q, *rotary)
    k = apply_rotary(k, *rotary)
    baseline_q, baseline_k, baseline_v = block.attn._pre_sdpa_inputs(q, k, v)
    direct_q, direct_k, direct_v = materialize_sdpa_inputs_contiguous(q, k, v)
    baseline_attn = block.attn(h, rotary)
    baseline_block = block(x, modulation, adaln_indices, rotary)

    block.attn.use_pre_sdpa_contiguous_candidate = True
    candidate_q, candidate_k, candidate_v = block.attn._pre_sdpa_inputs(q, k, v)
    candidate_attn = block.attn(h, rotary)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_pre_sdpa_contiguous_candidate = False
    mx.eval(
        baseline_q,
        baseline_k,
        baseline_v,
        direct_q,
        direct_k,
        direct_v,
        candidate_q,
        candidate_k,
        candidate_v,
        baseline_attn,
        candidate_attn,
        baseline_block,
        candidate_block,
    )

    assert candidate_q.shape == baseline_q.shape
    assert candidate_k.shape == baseline_k.shape
    assert candidate_v.shape == baseline_v.shape
    assert candidate_q.dtype == baseline_q.dtype
    assert candidate_k.dtype == baseline_k.dtype
    assert candidate_v.dtype == baseline_v.dtype
    deltas = {
        "direct_q": float(mx.max(mx.abs(baseline_q - direct_q)).item()),
        "direct_k": float(mx.max(mx.abs(baseline_k - direct_k)).item()),
        "direct_v": float(mx.max(mx.abs(baseline_v - direct_v)).item()),
        "candidate_q": float(mx.max(mx.abs(baseline_q - candidate_q)).item()),
        "candidate_k": float(mx.max(mx.abs(baseline_k - candidate_k)).item()),
        "candidate_v": float(mx.max(mx.abs(baseline_v - candidate_v)).item()),
        "attn": float(mx.max(mx.abs(baseline_attn - candidate_attn)).item()),
        "block": float(mx.max(mx.abs(baseline_block - candidate_block)).item()),
    }
    assert all(delta == 0.0 for delta in deltas.values()), f"pre-SDPA contiguous deltas: {deltas}"
    print(f"attention pre-SDPA contiguous candidate exact: {deltas}")


def archived_attention_sdpa_headgroup_split_rank4_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(18)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    assert block.attn.use_sdpa_headgroup_split_candidate is False
    q, k, v = block.attn._qkv_sdpa_tensors(h)
    q = apply_rotary(q, *rotary)
    k = apply_rotary(k, *rotary)
    baseline_sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=block.attn.scale, mask=None)
    candidate_sdpa = sdpa_headgroup_split_rank4(q, k, v, scale=block.attn.scale, mask=None, heads_per_group=2)

    block.attn.use_sdpa_headgroup_split_candidate = False
    baseline_attn = block.attn(h, rotary)
    baseline_block = block(x, modulation, adaln_indices, rotary)
    block.attn.sdpa_headgroup_heads_per_group = 2
    block.attn.use_sdpa_headgroup_split_candidate = True
    candidate_attn = block.attn(h, rotary)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_sdpa_headgroup_split_candidate = False
    mx.eval(baseline_sdpa, candidate_sdpa, baseline_attn, candidate_attn, baseline_block, candidate_block)

    assert candidate_sdpa.shape == baseline_sdpa.shape == q.shape
    assert candidate_sdpa.dtype == baseline_sdpa.dtype
    assert candidate_attn.shape == baseline_attn.shape
    assert candidate_attn.dtype == baseline_attn.dtype
    deltas = {
        "sdpa": float(mx.max(mx.abs(baseline_sdpa.astype(mx.float32) - candidate_sdpa.astype(mx.float32))).item()),
        "attn": float(mx.max(mx.abs(baseline_attn.astype(mx.float32) - candidate_attn.astype(mx.float32))).item()),
        "block": float(mx.max(mx.abs(baseline_block.astype(mx.float32) - candidate_block.astype(mx.float32))).item()),
    }
    tolerance = 1e-6
    assert all(delta <= tolerance for delta in deltas.values()), f"SDPA head-group rank4 deltas: {deltas}"
    print(f"attention SDPA head-group rank4 candidate within {tolerance}: {deltas}")



def archived_attention_sdpa_head_batch_rank3_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(19)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    assert block.attn.use_sdpa_head_batch_rank3_candidate is False
    q, k, v = block.attn._qkv_sdpa_tensors(h)
    q = apply_rotary(q, *rotary)
    k = apply_rotary(k, *rotary)
    baseline_sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=block.attn.scale, mask=None)
    try:
        candidate_sdpa = sdpa_head_batch_rank3(q, k, v, scale=block.attn.scale, mask=None)
    except RuntimeError as exc:
        assert "rank-3 [B*H,S,D]" in str(exc)
        block.attn.use_sdpa_head_batch_rank3_candidate = True
        try:
            candidate_attn = block.attn(h, rotary)
            mx.eval(candidate_attn)
        except RuntimeError as attn_exc:
            assert "rank-3 [B*H,S,D]" in str(attn_exc)
        else:
            raise AssertionError("SDPA head-batch rank3 flag should surface the same unsupported rank-3 error")
        finally:
            block.attn.use_sdpa_head_batch_rank3_candidate = False
        mx.eval(baseline_sdpa)
        print(f"attention SDPA head-batch rank3 candidate unsupported by local MLX: {exc}")
        return

    block.attn.use_sdpa_head_batch_rank3_candidate = False
    baseline_attn = block.attn(h, rotary)
    baseline_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_sdpa_head_batch_rank3_candidate = True
    candidate_attn = block.attn(h, rotary)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_sdpa_head_batch_rank3_candidate = False
    mx.eval(baseline_sdpa, candidate_sdpa, baseline_attn, candidate_attn, baseline_block, candidate_block)

    assert candidate_sdpa.shape == baseline_sdpa.shape == q.shape
    assert candidate_sdpa.dtype == baseline_sdpa.dtype
    assert candidate_attn.shape == baseline_attn.shape
    assert candidate_attn.dtype == baseline_attn.dtype
    deltas = {
        "sdpa": float(mx.max(mx.abs(baseline_sdpa.astype(mx.float32) - candidate_sdpa.astype(mx.float32))).item()),
        "attn": float(mx.max(mx.abs(baseline_attn.astype(mx.float32) - candidate_attn.astype(mx.float32))).item()),
        "block": float(mx.max(mx.abs(baseline_block.astype(mx.float32) - candidate_block.astype(mx.float32))).item()),
    }
    assert all(delta == 0.0 for delta in deltas.values()), f"SDPA head-batch rank3 deltas: {deltas}"
    print(f"attention SDPA head-batch rank3 candidate exact: {deltas}")



def archived_attention_pre_out_proj_contiguous_candidate_exact():
    cfg = tiny_config()
    mx.random.seed(23)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    x = mx.random.normal((1, 13, cfg.hidden_size)).astype(mx.bfloat16)
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    assert block.attn.use_pre_out_proj_contiguous_candidate is False
    q, k, v = block.attn._qkv_sdpa_tensors(h)
    q = apply_rotary(q, *rotary)
    k = apply_rotary(k, *rotary)
    sdpa_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=block.attn.scale, mask=None)
    out_input = sdpa_out.transpose(0, 2, 1, 3).reshape(1, 13, cfg.inner_dim).astype(h.dtype)
    direct_input = materialize_attention_output_contiguous(out_input, cfg.inner_dim)

    block.attn.use_pre_out_proj_contiguous_candidate = False
    baseline_input = block.attn._pre_out_project_input(out_input)
    baseline_projection = block.attn._out_project(out_input)
    baseline_attn = block.attn(h, rotary)
    baseline_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_pre_out_proj_contiguous_candidate = True
    candidate_input = block.attn._pre_out_project_input(out_input)
    candidate_projection = block.attn._out_project(out_input)
    candidate_attn = block.attn(h, rotary)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_pre_out_proj_contiguous_candidate = False
    mx.eval(
        direct_input,
        baseline_input,
        candidate_input,
        baseline_projection,
        candidate_projection,
        baseline_attn,
        candidate_attn,
        baseline_block,
        candidate_block,
    )

    assert candidate_input.shape == baseline_input.shape == out_input.shape
    assert direct_input.shape == out_input.shape
    assert candidate_input.dtype == baseline_input.dtype == out_input.dtype
    assert direct_input.dtype == out_input.dtype
    assert candidate_projection.shape == baseline_projection.shape
    assert candidate_projection.dtype == baseline_projection.dtype
    assert candidate_attn.shape == baseline_attn.shape
    assert candidate_attn.dtype == baseline_attn.dtype
    deltas = {
        "input": float(mx.max(mx.abs(baseline_input.astype(mx.float32) - candidate_input.astype(mx.float32))).item()),
        "direct_input": float(mx.max(mx.abs(out_input.astype(mx.float32) - direct_input.astype(mx.float32))).item()),
        "projection": float(mx.max(mx.abs(baseline_projection - candidate_projection)).item()),
        "attn": float(mx.max(mx.abs(baseline_attn - candidate_attn)).item()),
        "block": float(mx.max(mx.abs(baseline_block - candidate_block)).item()),
    }
    assert all(delta == 0.0 for delta in deltas.values()), f"pre-out_proj contiguous deltas: {deltas}"
    print(f"attention pre-out_proj contiguous candidate exact: {deltas}")



def archived_attention_qkv_rmsnorm_sdpa_metal_candidate_bounded():
    if not has_qkv_rmsnorm_sdpa_metal():
        print("attention q/k RMSNorm SDPA Metal candidate skipped: mx.fast.metal_kernel unavailable")
        return

    cfg = tiny_config()
    mx.random.seed(31)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    x = mx.random.normal((1, 13, cfg.hidden_size))
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    assert block.attn.use_qkv_rmsnorm_sdpa_metal_candidate is False
    block.attn.use_qkv_rmsnorm_sdpa_metal_candidate = False
    baseline_q, baseline_k, baseline_v = block.attn._qkv_sdpa_tensors(h)
    qkv = block.attn._qkv_project(h).reshape(1, 13, cfg.num_attention_heads, 3, cfg.attention_head_dim)
    metal_q, metal_k, metal_v = qkv_rmsnorm_sdpa_metal(
        qkv, block.attn.q_norm.weight, block.attn.k_norm.weight, cfg.qk_norm_eps
    )
    baseline_attn = block.attn(h, rotary)
    baseline_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_qkv_rmsnorm_sdpa_metal_candidate = True
    candidate_q, candidate_k, candidate_v = block.attn._qkv_sdpa_tensors(h)
    candidate_attn = block.attn(h, rotary)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_qkv_rmsnorm_sdpa_metal_candidate = False
    mx.eval(
        baseline_q,
        baseline_k,
        baseline_v,
        metal_q,
        metal_k,
        metal_v,
        candidate_q,
        candidate_k,
        candidate_v,
        baseline_attn,
        candidate_attn,
        baseline_block,
        candidate_block,
    )

    assert metal_q.shape == baseline_q.shape
    assert metal_k.shape == baseline_k.shape
    assert metal_v.shape == baseline_v.shape
    assert candidate_q.shape == baseline_q.shape
    assert candidate_k.shape == baseline_k.shape
    assert candidate_v.shape == baseline_v.shape
    assert metal_q.dtype == baseline_q.dtype
    assert metal_k.dtype == baseline_k.dtype
    assert metal_v.dtype == baseline_v.dtype
    deltas = {
        "direct_q": float(mx.max(mx.abs(baseline_q - metal_q)).item()),
        "direct_k": float(mx.max(mx.abs(baseline_k - metal_k)).item()),
        "direct_v": float(mx.max(mx.abs(baseline_v - metal_v)).item()),
        "candidate_q": float(mx.max(mx.abs(baseline_q - candidate_q)).item()),
        "candidate_k": float(mx.max(mx.abs(baseline_k - candidate_k)).item()),
        "candidate_v": float(mx.max(mx.abs(baseline_v - candidate_v)).item()),
        "attn": float(mx.max(mx.abs(baseline_attn - candidate_attn)).item()),
        "block": float(mx.max(mx.abs(baseline_block - candidate_block)).item()),
    }
    assert deltas["direct_v"] == 0.0 and deltas["candidate_v"] == 0.0, f"Metal SDPA v deltas: {deltas}"
    assert all(delta <= 1e-5 for key, delta in deltas.items() if key not in {"direct_v", "candidate_v"}), (
        f"Metal q/k RMSNorm SDPA deltas: {deltas}"
    )
    print(f"attention q/k RMSNorm SDPA Metal candidate bounded: {deltas}")


def archived_attention_qkv_rmsnorm_rotary_sdpa_metal_candidate_bounded():
    if not has_qkv_rmsnorm_rotary_sdpa_metal():
        print("attention q/k RMSNorm+RoPE SDPA Metal candidate skipped: mx.fast.metal_kernel unavailable")
        return

    cfg = tiny_config()
    mx.random.seed(39)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    x = mx.random.normal((1, 13, cfg.hidden_size))
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    assert block.attn.use_qkv_rmsnorm_rotary_sdpa_metal_candidate is False
    baseline_q, baseline_k, baseline_v = block.attn._qkv_sdpa_tensors(h)
    baseline_q = apply_rotary(baseline_q, *rotary)
    baseline_k = apply_rotary(baseline_k, *rotary)
    qkv = block.attn._qkv_project(h).reshape(1, 13, cfg.num_attention_heads, 3, cfg.attention_head_dim)
    metal_q, metal_k, metal_v = qkv_rmsnorm_rotary_sdpa_metal(
        qkv,
        block.attn.q_norm.weight,
        block.attn.k_norm.weight,
        rotary[0],
        rotary[1],
        cfg.qk_norm_eps,
    )

    block.attn.use_qkv_rmsnorm_rotary_sdpa_metal_candidate = False
    baseline_attn = block.attn(h, rotary)
    baseline_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_qkv_rmsnorm_rotary_sdpa_metal_candidate = True
    candidate_q, candidate_k, candidate_v = block.attn._qkv_rmsnorm_rotary_sdpa_tensors(h, rotary)
    candidate_attn = block.attn(h, rotary)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_qkv_rmsnorm_rotary_sdpa_metal_candidate = False
    mx.eval(
        baseline_q,
        baseline_k,
        baseline_v,
        metal_q,
        metal_k,
        metal_v,
        candidate_q,
        candidate_k,
        candidate_v,
        baseline_attn,
        candidate_attn,
        baseline_block,
        candidate_block,
    )

    assert metal_q.shape == baseline_q.shape
    assert metal_k.shape == baseline_k.shape
    assert metal_v.shape == baseline_v.shape
    assert candidate_q.shape == baseline_q.shape
    assert candidate_k.shape == baseline_k.shape
    assert candidate_v.shape == baseline_v.shape
    assert metal_q.dtype == baseline_q.dtype
    assert metal_k.dtype == baseline_k.dtype
    assert metal_v.dtype == baseline_v.dtype
    deltas = {
        "direct_q": float(mx.max(mx.abs(baseline_q - metal_q)).item()),
        "direct_k": float(mx.max(mx.abs(baseline_k - metal_k)).item()),
        "direct_v": float(mx.max(mx.abs(baseline_v - metal_v)).item()),
        "candidate_q": float(mx.max(mx.abs(baseline_q - candidate_q)).item()),
        "candidate_k": float(mx.max(mx.abs(baseline_k - candidate_k)).item()),
        "candidate_v": float(mx.max(mx.abs(baseline_v - candidate_v)).item()),
        "attn": float(mx.max(mx.abs(baseline_attn - candidate_attn)).item()),
        "block": float(mx.max(mx.abs(baseline_block - candidate_block)).item()),
    }
    assert deltas["direct_v"] == 0.0 and deltas["candidate_v"] == 0.0, f"Metal fused v deltas: {deltas}"
    assert all(delta <= 2e-5 for key, delta in deltas.items() if key not in {"direct_v", "candidate_v"}), (
        f"Metal q/k RMSNorm+RoPE SDPA deltas: {deltas}"
    )

    bf16_qkv = mx.random.normal((1, 7, cfg.num_attention_heads, 3, cfg.attention_head_dim)).astype(mx.bfloat16)
    bf16_q_weight = mx.random.normal((cfg.attention_head_dim,)).astype(mx.bfloat16)
    bf16_k_weight = mx.random.normal((cfg.attention_head_dim,)).astype(mx.bfloat16)
    bf16_position_ids = mx.stack(
        [mx.arange(7) % 3, mx.arange(7) % 5, mx.arange(7) % 7], axis=-1
    ).astype(mx.int32)
    bf16_rotary = dit.rope(bf16_position_ids)
    bf16_q = mx.fast.rms_norm(bf16_qkv[:, :, :, 0], bf16_q_weight, cfg.qk_norm_eps).transpose(0, 2, 1, 3)
    bf16_k = mx.fast.rms_norm(bf16_qkv[:, :, :, 1], bf16_k_weight, cfg.qk_norm_eps).transpose(0, 2, 1, 3)
    bf16_v = bf16_qkv[:, :, :, 2].transpose(0, 2, 1, 3)
    bf16_q = apply_rotary(bf16_q, *bf16_rotary)
    bf16_k = apply_rotary(bf16_k, *bf16_rotary)
    bf16_metal_q, bf16_metal_k, bf16_metal_v = qkv_rmsnorm_rotary_sdpa_metal(
        bf16_qkv, bf16_q_weight, bf16_k_weight, bf16_rotary[0], bf16_rotary[1], cfg.qk_norm_eps
    )
    mx.eval(bf16_q, bf16_k, bf16_v, bf16_metal_q, bf16_metal_k, bf16_metal_v)
    bf16_deltas = {
        "q": float(mx.max(mx.abs(bf16_q.astype(mx.float32) - bf16_metal_q.astype(mx.float32))).item()),
        "k": float(mx.max(mx.abs(bf16_k.astype(mx.float32) - bf16_metal_k.astype(mx.float32))).item()),
        "v": float(mx.max(mx.abs(bf16_v.astype(mx.float32) - bf16_metal_v.astype(mx.float32))).item()),
    }
    assert bf16_deltas == {"q": 0.0, "k": 0.0, "v": 0.0}, f"BF16 RMSNorm+RoPE SDPA deltas: {bf16_deltas}"
    print(f"attention q/k RMSNorm+RoPE SDPA Metal candidate bounded: {deltas}, bf16 {bf16_deltas}")


def archived_attention_sdpa_out_layout_metal_candidate_exact():
    if not has_sdpa_out_layout_metal():
        print("attention SDPA-output layout Metal candidate skipped: mx.fast.metal_kernel unavailable")
        return

    cfg = tiny_config()
    mx.random.seed(37)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]

    sdpa = mx.random.normal((2, cfg.num_attention_heads, 9, cfg.attention_head_dim)).astype(mx.bfloat16)
    baseline_layout = sdpa.transpose(0, 2, 1, 3).reshape(2, 9, cfg.inner_dim)
    metal_layout = sdpa_out_to_bshd_metal(sdpa)

    x = mx.random.normal((1, 13, cfg.hidden_size))
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    assert block.attn.use_sdpa_out_layout_metal_candidate is False
    q, k, v = block.attn._qkv_sdpa_tensors(h)
    q = apply_rotary(q, *rotary)
    k = apply_rotary(k, *rotary)
    sdpa_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=block.attn.scale, mask=None)
    baseline_attn_layout = sdpa_out.transpose(0, 2, 1, 3).reshape(1, 13, cfg.inner_dim)
    metal_attn_layout = sdpa_out_to_bshd_metal(sdpa_out)

    block.attn.use_sdpa_out_layout_metal_candidate = False
    baseline_attn = block.attn(h, rotary)
    baseline_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_sdpa_out_layout_metal_candidate = True
    candidate_attn = block.attn(h, rotary)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_sdpa_out_layout_metal_candidate = False
    mx.eval(
        baseline_layout,
        metal_layout,
        baseline_attn_layout,
        metal_attn_layout,
        baseline_attn,
        candidate_attn,
        baseline_block,
        candidate_block,
    )

    assert metal_layout.shape == baseline_layout.shape
    assert metal_layout.dtype == baseline_layout.dtype
    assert metal_attn_layout.shape == baseline_attn_layout.shape
    assert metal_attn_layout.dtype == baseline_attn_layout.dtype
    assert candidate_attn.shape == baseline_attn.shape
    assert candidate_attn.dtype == baseline_attn.dtype
    deltas = {
        "direct_bf16_layout": float(mx.max(mx.abs(baseline_layout.astype(mx.float32) - metal_layout.astype(mx.float32))).item()),
        "direct_attention_layout": float(
            mx.max(mx.abs(baseline_attn_layout.astype(mx.float32) - metal_attn_layout.astype(mx.float32))).item()
        ),
        "attn": float(mx.max(mx.abs(baseline_attn - candidate_attn)).item()),
        "block": float(mx.max(mx.abs(baseline_block - candidate_block)).item()),
    }
    assert all(delta == 0.0 for delta in deltas.values()), f"SDPA-output layout Metal deltas: {deltas}"
    print(f"attention SDPA-output layout Metal candidate exact: {deltas}")



def archived_attention_rotary_qk_metal_candidate_bounded():
    if not has_rotary_qk_metal():
        print("attention q/k RoPE Metal candidate skipped: mx.fast.metal_kernel unavailable")
        return

    cfg = tiny_config()
    mx.random.seed(29)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    block = dit.blocks[0]
    x = mx.random.normal((1, 13, cfg.hidden_size))
    temb = mx.random.normal((2, cfg.time_embed_dim))
    modulation = block.adaln_proj(temb)
    adaln_indices = mx.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0], dtype=mx.int32)
    position_ids = mx.stack(
        [mx.arange(13) % 3, mx.arange(13) % 5, mx.arange(13) % 7], axis=-1
    ).astype(mx.int32)
    rotary = dit.rope(position_ids)
    h = block.norm1(x) * (1.0 + modulation[1][adaln_indices]) + modulation[0][adaln_indices]

    assert block.attn.use_rotary_qk_metal_candidate is False
    q, k, _v = block.attn._qkv_sdpa_tensors(h)
    baseline_q = apply_rotary(q, *rotary)
    baseline_k = apply_rotary(k, *rotary)
    metal_q, metal_k = apply_rotary_qk_metal(q, k, *rotary)

    block.attn.use_rotary_qk_metal_candidate = False
    baseline_attn = block.attn(h, rotary)
    baseline_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_rotary_qk_metal_candidate = True
    candidate_attn = block.attn(h, rotary)
    candidate_block = block(x, modulation, adaln_indices, rotary)
    block.attn.use_rotary_qk_metal_candidate = False
    mx.eval(baseline_q, baseline_k, metal_q, metal_k, baseline_attn, candidate_attn, baseline_block, candidate_block)

    assert metal_q.shape == baseline_q.shape
    assert metal_k.shape == baseline_k.shape
    assert metal_q.dtype == baseline_q.dtype
    assert metal_k.dtype == baseline_k.dtype
    deltas = {
        "q_rotary": float(mx.max(mx.abs(baseline_q - metal_q)).item()),
        "k_rotary": float(mx.max(mx.abs(baseline_k - metal_k)).item()),
        "attn": float(mx.max(mx.abs(baseline_attn - candidate_attn)).item()),
        "block": float(mx.max(mx.abs(baseline_block - candidate_block)).item()),
    }
    assert deltas["q_rotary"] <= 1e-5 and deltas["k_rotary"] <= 1e-5, f"Metal RoPE q/k deltas: {deltas}"
    assert deltas["attn"] <= 1e-5 and deltas["block"] <= 1e-5, f"Metal RoPE full-path deltas: {deltas}"

    q_bf16 = mx.random.normal((1, cfg.num_attention_heads, 9, cfg.attention_head_dim)).astype(mx.bfloat16)
    k_bf16 = mx.random.normal((1, cfg.num_attention_heads, 9, cfg.attention_head_dim)).astype(mx.bfloat16)
    bf16_pos = mx.stack([mx.arange(9) % 3, mx.arange(9) % 5, mx.arange(9) % 7], axis=-1).astype(mx.int32)
    bf16_rotary = dit.rope(bf16_pos)
    bf16_base_q = apply_rotary(q_bf16, *bf16_rotary)
    bf16_base_k = apply_rotary(k_bf16, *bf16_rotary)
    bf16_metal_q, bf16_metal_k = apply_rotary_qk_metal(q_bf16, k_bf16, *bf16_rotary)
    mx.eval(bf16_base_q, bf16_base_k, bf16_metal_q, bf16_metal_k)
    bf16_delta = max(
        float(mx.max(mx.abs(bf16_base_q.astype(mx.float32) - bf16_metal_q.astype(mx.float32))).item()),
        float(mx.max(mx.abs(bf16_base_k.astype(mx.float32) - bf16_metal_k.astype(mx.float32))).item()),
    )
    assert bf16_delta <= 4e-2, f"bf16 Metal RoPE bounded delta {bf16_delta}"
    print(f"attention q/k RoPE Metal candidate bounded: {deltas}, bf16 max delta {bf16_delta}")


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
    test_archived_hotpath_candidates_are_decollected_and_not_manually_invoked()
    test_schedule_timesteps()
    test_ffn_fc2_tiled_dense_dequant_candidate_exact_for_quantized_tiny_fc2()
    test_refined_text_cache_candidate_exact_and_ignores_text_embeds()
    test_attention_out_dense_dequant_candidate_exact_for_quantized_tiny_out_proj()
    test_attention_out_tiled_dense_dequant_candidate_exact_for_quantized_tiny_out_proj()
    test_attention_qkv_tiled_dense_dequant_candidate_exact_for_quantized_tiny_qkv()
    test_modulation_cache_matches_live_projection()
    test_block_cache_disabled_is_exact_and_enabled_skips_tail()
    print("\nall smoke tests passed")
