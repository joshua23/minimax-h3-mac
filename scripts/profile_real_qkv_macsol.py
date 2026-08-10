#!/usr/bin/env python3
"""Capture real MiniMax-H3 attention Q/K/V and replay MacSol routing statistics.

This is an opt-in profiling harness only. It loads one real quantized DiT block, builds the same
packed-layout geometry used by the MLX generation path, captures that block's post-QK-RMSNorm,
post-RoPE Q/K/V tensors in SDPA layout ``[B,H,S,D]``, and evaluates Sol-Attn/MacSol routing plus
sampled attention-output error against dense attention rows.

The harness deliberately does not wire MacSol into generation, does not materialize a full dense
``S x S`` attention matrix for target-resolution runs, and does not change dense defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from minimax_h3_mlx.config import MODALITY_NUM, TAG_TEXT, DiTConfig  # noqa: E402
from minimax_h3_mlx.dit import MiniMaxH3DiT, RotaryPosEmbed3D, TransformerBlock, apply_rotary  # noqa: E402
from minimax_h3_mlx.macsol_reference import (  # noqa: E402
    H3PackedLengths,
    MacSolReferenceConfig,
    build_macsol_routing,
    h3_macsol_config,
)
from minimax_h3_mlx.packing import (  # noqa: E402
    AUDIO_CHANNELS,
    FPS,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    video_latent_num_frames,
)
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler  # noqa: E402

DEFAULT_OUT = "experiments/macsol_real_qkv/latest.json"


def _tiny_config() -> DiTConfig:
    hidden = 64
    return DiTConfig(
        hidden_size=hidden,
        num_layers=1,
        token_refiner_num_layers=1,
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


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _vm_stat_value(label: str) -> int | None:
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return None
    prefix = f"{label}:"
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            raw = stripped.split(":", 1)[1].strip().rstrip(".").replace(".", "")
            try:
                return int(raw)
            except ValueError:
                return None
    return None


def _swapusage() -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
    except Exception:
        return None


def _current_rss_kib() -> int | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(out.strip())
    except Exception:
        return None


def _mlx_call(name: str) -> int | None:
    func = getattr(mx, name, None)
    if func is None and hasattr(mx, "metal"):
        func = getattr(mx.metal, name, None)
    if func is None:
        return None
    try:
        return int(func())
    except Exception:
        return None


def _reset_mlx_peak() -> None:
    for owner in (mx, getattr(mx, "metal", None)):
        func = getattr(owner, "reset_peak_memory", None) if owner is not None else None
        if func is None:
            continue
        try:
            func()
            return
        except Exception:
            continue


def _metrics() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "wall_time_seconds": time.perf_counter(),
        "current_rss_kib": _current_rss_kib(),
        "ru_maxrss_raw": usage.ru_maxrss,
        "ru_minflt": usage.ru_minflt,
        "ru_majflt": usage.ru_majflt,
        "ru_inblock": usage.ru_inblock,
        "ru_oublock": usage.ru_oublock,
        "ru_nvcsw": usage.ru_nvcsw,
        "ru_nivcsw": usage.ru_nivcsw,
        "vm_pageouts": _vm_stat_value("Pageouts"),
        "vm_swapouts": _vm_stat_value("Swapouts"),
        "vm_swapusage": _swapusage(),
        "mlx_peak_bytes": _mlx_call("get_peak_memory"),
        "mlx_active_bytes": _mlx_call("get_active_memory"),
        "mlx_cache_bytes": _mlx_call("get_cache_memory"),
    }


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(after_value, (int, float)) and isinstance(before_value, (int, float)):
            out[key] = after_value - before_value
    return out


def _process_scan() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    try:
        ps_out = subprocess.check_output(
            ["ps", "-axo", "pid,ppid,%cpu,%mem,rss,comm,args"],
            text=True,
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "rows": []}
    for line in ps_out.splitlines()[1:]:
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, cpu, mem, rss, comm, args = parts
        haystack = f"{comm} {args}".lower()
        if not any(token in haystack for token in ("python", "mlx", "generate", "ffmpeg", "argus")):
            continue
        try:
            rss_kib = int(rss)
        except ValueError:
            rss_kib = 0
        rows.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "cpu_percent": float(cpu),
                "mem_percent": float(mem),
                "rss_kib": rss_kib,
                "comm": comm,
                "args": args[:240],
            }
        )
    high_threshold = 8 * 1024 * 1024
    return {
        "ok": True,
        "filter": "python|mlx|generate|ffmpeg|argus",
        "high_memory_threshold_rss_kib": high_threshold,
        "other_high_memory_processes": [
            row for row in rows if row["pid"] != os.getpid() and row["rss_kib"] >= high_threshold
        ],
        "rows": rows[:20],
    }


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            values.append(float(piece))
    if not values:
        raise ValueError("at least one tau value is required")
    return values


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip().lower() in {"", "auto"}:
        return None
    values: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            values.append(int(piece))
    return values


def _parse_required_int_list(raw: str, *, name: str) -> list[int]:
    values = _parse_int_list(raw)
    if not values:
        raise ValueError(f"at least one {name} value is required")
    return values


def _sweep_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.routing_mode != "threshold":
        raise ValueError(
            "only source-faithful threshold routing is active; budget_topk, "
            "threshold_h3_structure, and spatial-tube variants are rejected provenance-only variants"
        )
    return [
        {
            "routing_mode": "threshold",
            "tau": float(tau),
            "sweep_label": f"tau={float(tau):g}",
        }
        for tau in _parse_float_list(args.taus)
    ]


def _format_sweep_label(item: dict[str, Any]) -> str:
    label = item.get("sweep_label")
    if label:
        return str(label)
    return f"tau={float(item['tau']):g}"


def _load_block(args: argparse.Namespace, events: list[dict[str, Any]]) -> tuple[TransformerBlock, DiTConfig, dict[str, Any]]:
    if args.tiny:
        cfg = _tiny_config()
        mx.random.seed(args.seed)
        model = MiniMaxH3DiT(cfg)
        mx.eval(model.parameters())
        events.append({"time": _now(), "message": "loaded tiny random block", "hidden_size": cfg.hidden_size})
        return model.blocks[0], cfg, {"mode": "tiny_random", "real_4bit": False}

    from mlx.utils import tree_flatten
    from minimax_h3_mlx.streaming import QuantizedBlockProvider

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    started = time.perf_counter()
    provider = QuantizedBlockProvider(model_dir, block_load_mode=args.block_load_mode)
    block = provider.load_block(args.block_index, include_adaln=True)
    flat = dict(tree_flatten(provider.slot.parameters()))
    mx.eval(*flat.values())
    mx.synchronize()
    elapsed = time.perf_counter() - started
    recipe: dict[str, Any] = {}
    quant_path = model_dir / "quant_config.json"
    if quant_path.exists():
        recipe = json.loads(quant_path.read_text())
    events.append(
        {
            "time": _now(),
            "message": "loaded real quantized block",
            "model_dir": str(model_dir),
            "block_index": args.block_index,
            "seconds": elapsed,
            "tensors": len(flat),
        }
    )
    return block, provider.config, {
        "mode": "quantized_provider",
        "real_4bit": True,
        "model_dir": str(model_dir),
        "block_load_mode": args.block_load_mode,
        "block_index": args.block_index,
        "load_materialize_seconds": elapsed,
        "logical_bytes_loaded": provider.logical_bytes_loaded,
        "quant_config": recipe,
    }


def _build_inputs(
    cfg: DiTConfig,
    args: argparse.Namespace,
    block: TransformerBlock,
    events: list[dict[str, Any]],
) -> tuple[
    mx.array,
    tuple[mx.array, ...],
    mx.array,
    tuple[mx.array, mx.array],
    mx.array,
    H3PackedLengths,
    dict[str, Any],
]:
    height = int(args.height)
    width = int(args.width)
    if height % 32 or width % 32:
        raise ValueError(f"height/width must be multiples of 32, got {height}x{width}")
    if height % args.spatial_compression or width % args.spatial_compression:
        raise ValueError(
            f"height/width must be divisible by spatial compression {args.spatial_compression}, got {height}x{width}"
        )
    num_frames = align_num_frames(int(round(args.duration * FPS)))
    num_latent_frames = video_latent_num_frames(num_frames)
    latent_height = height // args.spatial_compression
    latent_width = width // args.spatial_compression
    num_audio_latents = audio_latent_num_frames(num_frames)
    text_tags = np.full((args.text_tokens,), TAG_TEXT, dtype=np.int64)
    layout = build_packed_sequence(
        text_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        cfg.patch_size,
    )

    video_sched = MiniMaxH3Scheduler(shift=args.video_sigma_shift)
    audio_sched = MiniMaxH3Scheduler(shift=args.audio_sigma_shift)
    video_sched.set_timesteps(args.sigma_grid_points)
    audio_sched.set_timesteps(args.sigma_grid_points)
    step_index = min(max(args.step_index, 0), len(video_sched.timesteps.tolist()) - 1)
    distinct, timestep_indices = build_row_timesteps(
        layout,
        float(video_sched.timesteps[step_index].item()),
        float(audio_sched.timesteps[step_index].item()),
        max(float(video_sched.timesteps[step_index].item()), 0.999),
        1.0,
    )
    adaln_indices = timestep_indices * MODALITY_NUM + mx.maximum(layout.token_tags, 0)
    rotary = RotaryPosEmbed3D(cfg)(layout.position_ids)

    mx.random.seed(args.seed + 1)
    temb = mx.random.normal((int(distinct.shape[0]), cfg.time_embed_dim)).astype(mx.float32)
    modulation = block.adaln_proj(temb)
    mx.eval(*modulation, adaln_indices, *rotary)

    dtype = mx.bfloat16 if args.activation_dtype == "bf16" else mx.float32
    mx.random.seed(args.seed)
    x = mx.random.normal((1, layout.sequence_length, cfg.hidden_size)).astype(dtype)
    mx.eval(x)

    _, ph, pw = cfg.patch_size
    rows_per_frame = (latent_height // ph) * (latent_width // pw)
    lengths = H3PackedLengths(
        text_tokens=int(len(layout.text_indices.tolist())),
        conditioning_video_tokens=int(layout.num_condition_video_rows),
        audio_tokens=int(len(layout.audio_indices.tolist())),
        target_video_tokens=int(len(layout.video_indices.tolist()) - layout.num_condition_video_rows),
    )
    if lengths.sequence_length != int(layout.sequence_length):
        raise ValueError(f"H3 length accounting mismatch: {lengths.sequence_length} vs {layout.sequence_length}")

    meta = {
        "height": height,
        "width": width,
        "duration_seconds": args.duration,
        "fps": FPS,
        "aligned_num_frames": num_frames,
        "num_latent_frames": num_latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "rows_per_video_latent_frame": rows_per_frame,
        "num_audio_latents": num_audio_latents,
        "audio_channels": AUDIO_CHANNELS,
        "lengths": lengths.as_dict(),
        "sequence_length": layout.sequence_length,
        "video_rows_total": int(len(layout.video_indices.tolist())),
        "target_video_rows": lengths.target_video_tokens,
        "audio_rows": lengths.audio_tokens,
        "text_rows": lengths.text_tokens,
        "condition_video_rows": lengths.conditioning_video_tokens,
        "packed_order": "[text | conditioning video | audio | target video]",
        "sigma_grid_points": args.sigma_grid_points,
        "profiled_step_index": step_index,
        "video_timestep": float(video_sched.timesteps[step_index].item()),
        "audio_timestep": float(audio_sched.timesteps[step_index].item()),
        "distinct_timestep_count_for_step": int(distinct.shape[0]),
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "heads": cfg.num_attention_heads,
        "head_dim": cfg.attention_head_dim,
        "inner_dim": cfg.inner_dim,
        "ffn_hidden_size": cfg.ffn_hidden_size,
        "rotary_dim": cfg.rotary_dim,
        "patch_size": list(cfg.patch_size),
        "activation_dtype": args.activation_dtype,
        "capture_source": "one real DiT block with representative random packed hidden input and real AdaLN/qkv/qk-norm/rope weights; not a full teacher-forced denoising trajectory",
    }
    events.append(
        {
            "time": _now(),
            "message": "built packed block input",
            "sequence_length": layout.sequence_length,
            "height": height,
            "width": width,
            "duration": args.duration,
        }
    )
    return x, modulation, adaln_indices, rotary, layout.position_ids, lengths, meta


def _capture_attention_qkv(
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
) -> tuple[mx.array, mx.array, mx.array, dict[str, Any]]:
    shift_msa, scale_msa, _gate_msa, _shift_mlp, _scale_mlp, _gate_mlp = modulation
    started = time.perf_counter()
    h = block.norm1(x)
    h = h * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
    mx.eval(h)
    q, k, v = block.attn._qkv_sdpa_tensors(h)  # real qkv_proj + q/k RMSNorm, default layout.
    q = apply_rotary(q, *rotary)
    k = apply_rotary(k, *rotary)
    mx.eval(q, k, v)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    return q, k, v, {
        "seconds": elapsed,
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "q_dtype": str(q.dtype),
        "k_dtype": str(k.dtype),
        "v_dtype": str(v.dtype),
        "q_nbytes": int(q.nbytes),
        "k_nbytes": int(k.nbytes),
        "v_nbytes": int(v.nbytes),
        "sdpa_layout": "[B,H,S,D]",
        "includes_qk_rmsnorm": True,
        "includes_rope_on_qk": True,
        "uses_default_dense_generation_path": True,
    }


def _stack_block_summaries(x: mx.array, ranges: tuple[tuple[int, int], ...], *, reducer: str) -> mx.array:
    parts: list[mx.array] = []
    x32 = x.astype(mx.float32)
    for lo, hi in ranges:
        block = x32[:, :, lo:hi, :]
        if reducer == "mean":
            parts.append(mx.mean(block, axis=2))
        elif reducer == "sum":
            parts.append(mx.sum(block, axis=2))
        else:
            raise ValueError(f"unknown reducer {reducer!r}")
    return mx.stack(parts, axis=2)


def _stable_exact_approx_block(
    q_block: mx.array,
    exact_k: mx.array | None,
    exact_v: mx.array | None,
    approx_k: mx.array | None,
    approx_vsum: mx.array | None,
    approx_lengths: mx.array | None,
    *,
    scale: float,
    head_dim: int,
) -> mx.array:
    pieces: list[mx.array] = []
    exact_scores = None
    approx_scores = None
    if exact_k is not None and exact_k.shape[0] > 0:
        exact_scores = mx.matmul(q_block, exact_k.T) * scale
        pieces.append(mx.max(exact_scores, axis=-1, keepdims=True))
    if approx_k is not None and approx_k.shape[0] > 0:
        approx_scores = mx.matmul(q_block, approx_k.T) * scale
        pieces.append(mx.max(approx_scores, axis=-1, keepdims=True))
    if not pieces:
        raise ValueError("sampled query block has no exact or approximate KV blocks")
    row_max = pieces[0]
    for piece in pieces[1:]:
        row_max = mx.maximum(row_max, piece)

    tokens = int(q_block.shape[0])
    numerator = mx.zeros((tokens, head_dim), dtype=mx.float32)
    denominator = mx.zeros((tokens, 1), dtype=mx.float32)
    if exact_scores is not None and exact_v is not None:
        exact_weights = mx.exp(exact_scores - row_max)
        numerator = numerator + mx.matmul(exact_weights, exact_v)
        denominator = denominator + mx.sum(exact_weights, axis=-1, keepdims=True)
    if approx_scores is not None and approx_vsum is not None and approx_lengths is not None:
        approx_weights = mx.exp(approx_scores - row_max)
        numerator = numerator + mx.matmul(approx_weights, approx_vsum)
        denominator = denominator + mx.sum(approx_weights * approx_lengths[None, :], axis=-1, keepdims=True)
    return numerator / denominator


def _dense_query_block(q_block: mx.array, k_head: mx.array, v_head: mx.array, *, scale: float) -> mx.array:
    scores = mx.matmul(q_block.astype(mx.float32), k_head.astype(mx.float32).T) * scale
    weights = mx.softmax(scores, axis=-1)
    return mx.matmul(weights, v_head.astype(mx.float32))


def _error_metrics(got: mx.array, dense: mx.array) -> dict[str, float]:
    diff = got.astype(mx.float32) - dense.astype(mx.float32)
    mx.eval(diff, dense)
    arr = np.asarray(diff, dtype=np.float32)
    ref = np.asarray(dense.astype(mx.float32), dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    ref_rms = float(np.sqrt(np.mean(np.square(ref)))) if ref.size else 0.0
    return {
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "mean_abs": float(np.mean(np.abs(arr))) if arr.size else 0.0,
        "rms_abs": rms,
        "dense_rms": ref_rms,
        "relative_rms": rms / max(ref_rms, 1e-12),
    }


def _auto_sample_heads(heads: int, requested: list[int] | None, max_count: int) -> list[int]:
    if requested is not None:
        values = sorted({h for h in requested if 0 <= h < heads})
        if not values:
            raise ValueError(f"no requested sample heads fall in [0,{heads})")
        return values
    max_count = max(1, min(int(max_count), heads))
    if max_count == 1:
        return [0]
    values = np.linspace(0, heads - 1, max_count).round().astype(int).tolist()
    return sorted(set(values))


def _auto_sample_q_blocks(
    q_block_ranges: tuple[tuple[int, int], ...],
    prefix_tokens: int,
    requested: list[int] | None,
    max_count: int,
) -> list[int]:
    q_blocks = len(q_block_ranges)
    if requested is not None:
        values = sorted({q for q in requested if 0 <= q < q_blocks})
        if not values:
            raise ValueError(f"no requested sample q-blocks fall in [0,{q_blocks})")
        return values

    target_blocks = [idx for idx, (_lo, hi) in enumerate(q_block_ranges) if hi > prefix_tokens]
    if not target_blocks:
        target_blocks = list(range(q_blocks))
    first = target_blocks[0]
    middle = target_blocks[len(target_blocks) // 2]
    last = target_blocks[-1]
    candidates = [first, middle, last]
    values: list[int] = []
    for item in candidates:
        if item not in values:
            values.append(item)
    return values[: max(1, int(max_count))]


def _effective_exact_block_stats(routing, prefix_tokens: int) -> dict[str, Any]:
    exact = np.array(routing.exact_block_mask, copy=True)
    prefix_q_blocks = [idx for idx, (lo, _hi) in enumerate(routing.q_block_ranges) if lo < prefix_tokens]
    if prefix_q_blocks:
        exact[:, :, prefix_q_blocks, :] = True
    total = int(exact.size)
    count = int(np.count_nonzero(exact))
    return {
        "note": "Conservative block-level compute proxy that treats any q-block overlapping H3 prefix-query dense rows as all-KV exact.",
        "prefix_query_dense_q_block_indices": prefix_q_blocks,
        "effective_exact_block_pairs_with_prefix_q_dense": count,
        "effective_exact_block_ratio_with_prefix_q_dense": count / total if total else 0.0,
    }


def _sample_attention_errors(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    routing,
    *,
    prefix_tokens: int,
    sample_heads: list[int],
    sample_q_blocks: list[int],
    scale: float,
) -> dict[str, Any]:
    k_mean = _stack_block_summaries(k, routing.kv_block_ranges, reducer="mean")
    v_sum = _stack_block_summaries(v, routing.kv_block_ranges, reducer="sum")
    mx.eval(k_mean, v_sum)
    kv_lengths_np = np.array([hi - lo for lo, hi in routing.kv_block_ranges], dtype=np.float32)
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    q32 = q.astype(mx.float32)
    head_dim = int(q.shape[-1])

    samples: list[dict[str, Any]] = []
    all_max_abs: list[float] = []
    all_mean_abs: list[float] = []
    all_rms_abs: list[float] = []
    all_rel_rms: list[float] = []
    started = time.perf_counter()
    for h in sample_heads:
        k_head = k32[0, h]
        v_head = v32[0, h]
        for qi in sample_q_blocks:
            qlo, qhi = routing.q_block_ranges[qi]
            q_block = q32[0, h, qlo:qhi, :]
            dense = _dense_query_block(q_block, k_head, v_head, scale=scale)

            exact_blocks = np.nonzero(routing.exact_block_mask[0, h, qi])[0].astype(int).tolist()
            approx_blocks = [idx for idx in range(len(routing.kv_block_ranges)) if idx not in exact_blocks]
            exact_k_parts = []
            exact_v_parts = []
            for kj in exact_blocks:
                klo, khi = routing.kv_block_ranges[kj]
                exact_k_parts.append(k32[0, h, klo:khi, :])
                exact_v_parts.append(v32[0, h, klo:khi, :])
            exact_k = mx.concatenate(exact_k_parts, axis=0) if exact_k_parts else None
            exact_v = mx.concatenate(exact_v_parts, axis=0) if exact_v_parts else None

            if approx_blocks:
                approx_k = mx.stack([k_mean[0, h, kj, :] for kj in approx_blocks], axis=0)
                approx_vsum = mx.stack([v_sum[0, h, kj, :] for kj in approx_blocks], axis=0)
                approx_lengths = mx.array(kv_lengths_np[approx_blocks], dtype=mx.float32)
            else:
                approx_k = None
                approx_vsum = None
                approx_lengths = None

            approx = _stable_exact_approx_block(
                q_block,
                exact_k,
                exact_v,
                approx_k,
                approx_vsum,
                approx_lengths,
                scale=scale,
                head_dim=head_dim,
            )
            if qlo < prefix_tokens:
                # H3 MacSol keeps prefix query rows dense.  For a q-block crossing the prefix/target
                # boundary, only the prefix rows are replaced by dense rows.
                prefix_count = max(0, min(qhi, prefix_tokens) - qlo)
                if prefix_count >= qhi - qlo:
                    approx = dense
                elif prefix_count > 0:
                    approx = mx.concatenate([dense[:prefix_count], approx[prefix_count:]], axis=0)
            mx.eval(approx, dense)
            errors = _error_metrics(approx, dense)
            all_max_abs.append(errors["max_abs"])
            all_mean_abs.append(errors["mean_abs"])
            all_rms_abs.append(errors["rms_abs"])
            all_rel_rms.append(errors["relative_rms"])
            samples.append(
                {
                    "head": int(h),
                    "q_block": int(qi),
                    "q_token_range": [int(qlo), int(qhi)],
                    "prefix_rows_replaced_by_dense": int(max(0, min(qhi, prefix_tokens) - qlo)),
                    "exact_block_count": int(len(exact_blocks)),
                    "approximate_block_count": int(len(approx_blocks)),
                    "exact_block_indices_first_16": exact_blocks[:16],
                    "errors_vs_dense": errors,
                }
            )
    elapsed = time.perf_counter() - started
    return {
        "sampled_attention_only": True,
        "reason_full_dense_not_materialized": "target-resolution SxS dense attention would exceed local memory; dense comparison is row-block sampled against all K/V tokens",
        "sample_heads": [int(h) for h in sample_heads],
        "sample_q_blocks": [int(qi) for qi in sample_q_blocks],
        "sample_count": len(samples),
        "elapsed_seconds": elapsed,
        "aggregate": {
            "max_sample_max_abs": max(all_max_abs) if all_max_abs else None,
            "median_sample_max_abs": float(statistics.median(all_max_abs)) if all_max_abs else None,
            "mean_of_sample_mean_abs": float(statistics.mean(all_mean_abs)) if all_mean_abs else None,
            "max_sample_rms_abs": max(all_rms_abs) if all_rms_abs else None,
            "median_sample_rms_abs": float(statistics.median(all_rms_abs)) if all_rms_abs else None,
            "max_sample_relative_rms": max(all_rel_rms) if all_rel_rms else None,
            "median_sample_relative_rms": float(statistics.median(all_rel_rms)) if all_rel_rms else None,
        },
        "samples": samples,
    }


def _engine_policy(sequence_length: int, block_index: int, layer_count: int, args: argparse.Namespace) -> dict[str, Any]:
    """Return the default-off MacSol admission policy for this block/sequence."""
    first_guard = max(0, int(args.sensitive_first_layers))
    last_guard = max(0, int(args.sensitive_last_layers))
    block_index = int(block_index)
    layer_count = int(layer_count)
    sequence_length = int(sequence_length)
    dense_fallback_below = int(args.dense_fallback_below)
    if sequence_length < dense_fallback_below:
        mode = "dense_fallback_short_sequence"
        sparse_replay_allowed = False
        reason = f"sequence_length {sequence_length} < dense_fallback_below {dense_fallback_below}"
    elif block_index < first_guard:
        mode = "dense_guard_first_sensitive_layer"
        sparse_replay_allowed = False
        reason = f"block_index {block_index} is inside first {first_guard} sensitive layer guard"
    elif last_guard and block_index >= max(0, layer_count - last_guard):
        mode = "dense_guard_last_sensitive_layer"
        sparse_replay_allowed = False
        reason = f"block_index {block_index} is inside last {last_guard} sensitive layer guard for {layer_count} layers"
    else:
        mode = "macsol_threshold_reference_probe"
        sparse_replay_allowed = True
        reason = "sequence/layer policy admits sparse MacSol threshold reference replay"
    return {
        "respect_engine_guards": bool(args.respect_engine_guards),
        "mode": mode,
        "sparse_replay_allowed": sparse_replay_allowed,
        "effective_replay_mode": mode if args.respect_engine_guards else "forced_sparse_reference_replay",
        "dense_fallback_below_tokens": dense_fallback_below,
        "sequence_length": sequence_length,
        "block_index": block_index,
        "layer_count": layer_count,
        "sensitive_first_layers": first_guard,
        "sensitive_last_layers": last_guard,
        "first_last_sensitive_guards_enabled": bool(first_guard or last_guard),
        "dense_short_sequence_fallback_enabled": True,
        "reason": reason,
    }


def _dense_guard_sweep_results(sweep_items: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    """Return sweep rows for cases where the intended engine must stay dense."""
    return [
        {
            "sweep_label": _format_sweep_label(item),
            "routing_mode": item["routing_mode"],
            "tau": item["tau"],
            "routing_build_seconds": 0.0,
            "routing_stats": {
                "engine_policy_dense_guard": True,
                "routing_mode": item["routing_mode"],
                "exact_block_ratio": 1.0,
                "effective_exact_block_ratio_with_prefix_q_dense": 1.0,
                "approximate_block_ratio": 0.0,
                "reason": policy["reason"],
            },
            "sampled_attention_errors": {
                "sampled_attention_only": True,
                "skipped": True,
                "reason": "engine policy keeps this block/sequence dense; sparse attention error is exactly zero by construction",
                "aggregate": {
                    "max_sample_max_abs": 0.0,
                    "median_sample_max_abs": 0.0,
                    "mean_of_sample_mean_abs": 0.0,
                    "max_sample_rms_abs": 0.0,
                    "median_sample_rms_abs": 0.0,
                    "max_sample_relative_rms": 0.0,
                    "median_sample_relative_rms": 0.0,
                },
                "samples": [],
            },
            "block_output_proxy": {
                "attempted": False,
                "skipped": True,
                "reason": "engine policy keeps this block/sequence dense; output error is exactly zero by construction",
            },
            "metrics_before": {},
            "metrics_after": {},
            "metrics_delta": {},
        }
        for item in sweep_items
    ]


def _sample_block_output_proxy(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    routing,
    *,
    block: TransformerBlock,
    x: mx.array,
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
    prefix_tokens: int,
    sample_heads: list[int],
    sample_q_blocks: list[int],
    scale: float,
) -> dict[str, Any]:
    """Propagate sampled-head attention differences through row-local block outputs.

    This is a compact admission proxy, not full DiT velocity: only the requested heads are populated
    in the SDPA output vector and all unrequested heads are zero in both dense and MacSol paths.
    The subsequent ``out_proj``, gated MSA residual, ``norm2``/AdaLN MLP, and gated MLP residual are
    real block weights and row-local for the sampled query rows.
    """
    k_mean = _stack_block_summaries(k, routing.kv_block_ranges, reducer="mean")
    v_sum = _stack_block_summaries(v, routing.kv_block_ranges, reducer="sum")
    mx.eval(k_mean, v_sum)
    kv_lengths_np = np.array([hi - lo for lo, hi in routing.kv_block_ranges], dtype=np.float32)
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    q32 = q.astype(mx.float32)
    heads_total = int(q.shape[1])
    head_dim = int(q.shape[-1])
    selected_heads = sorted({int(h) for h in sample_heads if 0 <= int(h) < heads_total})
    selected_head_set = set(selected_heads)
    if not selected_heads:
        return {"attempted": False, "skipped": True, "reason": "no valid sampled heads for output proxy"}

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
    samples: list[dict[str, Any]] = []
    attention_rels: list[float] = []
    projected_rels: list[float] = []
    msa_rels: list[float] = []
    block_rels: list[float] = []
    started = time.perf_counter()

    zero_cache: dict[int, mx.array] = {}
    for qi in sample_q_blocks:
        qlo, qhi = routing.q_block_ranges[qi]
        tokens = int(qhi - qlo)
        if tokens not in zero_cache:
            zero_cache[tokens] = mx.zeros((tokens, head_dim), dtype=mx.float32)
        dense_head_outputs: list[mx.array] = []
        approx_head_outputs: list[mx.array] = []
        exact_counts: list[int] = []
        approx_counts: list[int] = []
        for h in range(heads_total):
            if h not in selected_head_set:
                z = zero_cache[tokens]
                dense_head_outputs.append(z)
                approx_head_outputs.append(z)
                continue
            q_block = q32[0, h, qlo:qhi, :]
            dense = _dense_query_block(q_block, k32[0, h], v32[0, h], scale=scale)
            exact_blocks = np.nonzero(routing.exact_block_mask[0, h, qi])[0].astype(int).tolist()
            approx_blocks = [idx for idx in range(len(routing.kv_block_ranges)) if idx not in exact_blocks]
            exact_counts.append(int(len(exact_blocks)))
            approx_counts.append(int(len(approx_blocks)))

            exact_k_parts = []
            exact_v_parts = []
            for kj in exact_blocks:
                klo, khi = routing.kv_block_ranges[kj]
                exact_k_parts.append(k32[0, h, klo:khi, :])
                exact_v_parts.append(v32[0, h, klo:khi, :])
            exact_k = mx.concatenate(exact_k_parts, axis=0) if exact_k_parts else None
            exact_v = mx.concatenate(exact_v_parts, axis=0) if exact_v_parts else None
            if approx_blocks:
                approx_k = mx.stack([k_mean[0, h, kj, :] for kj in approx_blocks], axis=0)
                approx_vsum = mx.stack([v_sum[0, h, kj, :] for kj in approx_blocks], axis=0)
                approx_lengths = mx.array(kv_lengths_np[approx_blocks], dtype=mx.float32)
            else:
                approx_k = None
                approx_vsum = None
                approx_lengths = None
            approx = _stable_exact_approx_block(
                q_block,
                exact_k,
                exact_v,
                approx_k,
                approx_vsum,
                approx_lengths,
                scale=scale,
                head_dim=head_dim,
            )
            if qlo < prefix_tokens:
                prefix_count = max(0, min(qhi, prefix_tokens) - qlo)
                if prefix_count >= tokens:
                    approx = dense
                elif prefix_count > 0:
                    approx = mx.concatenate([dense[:prefix_count], approx[prefix_count:]], axis=0)
            dense_head_outputs.append(dense)
            approx_head_outputs.append(approx)

        dense_sdpa = mx.stack(dense_head_outputs, axis=0)
        approx_sdpa = mx.stack(approx_head_outputs, axis=0)
        dense_inner = dense_sdpa.transpose(1, 0, 2).reshape(1, tokens, heads_total * head_dim)
        approx_inner = approx_sdpa.transpose(1, 0, 2).reshape(1, tokens, heads_total * head_dim)
        dense_projected = block.attn._out_project(dense_inner.astype(x.dtype))
        approx_projected = block.attn._out_project(approx_inner.astype(x.dtype))

        idx = adaln_indices[qlo:qhi]
        x_rows = x[:, qlo:qhi, :]
        gate_msa_rows = gate_msa[idx]
        dense_msa = x_rows + gate_msa_rows * dense_projected
        approx_msa = x_rows + gate_msa_rows * approx_projected

        shift_mlp_rows = shift_mlp[idx]
        scale_mlp_rows = scale_mlp[idx]
        gate_mlp_rows = gate_mlp[idx]
        dense_h = block.norm2(dense_msa)
        dense_h = dense_h * (1.0 + scale_mlp_rows) + shift_mlp_rows
        approx_h = block.norm2(approx_msa)
        approx_h = approx_h * (1.0 + scale_mlp_rows) + shift_mlp_rows
        dense_mlp = block.mlp(dense_h)
        approx_mlp = block.mlp(approx_h)
        dense_block = dense_msa + gate_mlp_rows * dense_mlp
        approx_block = approx_msa + gate_mlp_rows * approx_mlp
        mx.eval(dense_sdpa, approx_sdpa, dense_projected, approx_projected, dense_msa, approx_msa, dense_block, approx_block)

        selected_dense = mx.stack([dense_sdpa[h] for h in selected_heads], axis=0)
        selected_approx = mx.stack([approx_sdpa[h] for h in selected_heads], axis=0)
        attention_error = _error_metrics(selected_approx, selected_dense)
        projected_error = _error_metrics(approx_projected, dense_projected)
        msa_error = _error_metrics(approx_msa, dense_msa)
        block_error = _error_metrics(approx_block, dense_block)
        attention_rels.append(attention_error["relative_rms"])
        projected_rels.append(projected_error["relative_rms"])
        msa_rels.append(msa_error["relative_rms"])
        block_rels.append(block_error["relative_rms"])
        samples.append(
            {
                "q_block": int(qi),
                "q_token_range": [int(qlo), int(qhi)],
                "sample_heads": selected_heads,
                "sampled_heads_only": True,
                "uncomputed_heads_zeroed_in_both_paths": True,
                "prefix_rows_replaced_by_dense": int(max(0, min(qhi, prefix_tokens) - qlo)),
                "mean_exact_blocks_per_sampled_head": float(statistics.mean(exact_counts)) if exact_counts else 0.0,
                "mean_approximate_blocks_per_sampled_head": float(statistics.mean(approx_counts)) if approx_counts else 0.0,
                "attention_error_vs_dense_for_sampled_heads": attention_error,
                "out_proj_error_vs_dense_for_sampled_head_contribution": projected_error,
                "msa_residual_error_vs_dense_for_sampled_head_contribution": msa_error,
                "block_output_error_vs_dense_for_sampled_head_contribution": block_error,
            }
        )

    elapsed = time.perf_counter() - started
    return {
        "attempted": True,
        "sampled_heads_only": True,
        "not_full_velocity": True,
        "reason_not_full_velocity": "compact admission proxy uses one real block and sampled query rows; it does not run all DiT layers or final velocity heads",
        "sample_heads": selected_heads,
        "sample_q_blocks": [int(qi) for qi in sample_q_blocks],
        "sample_count": len(samples),
        "elapsed_seconds": elapsed,
        "aggregate": {
            "max_attention_relative_rms": max(attention_rels) if attention_rels else None,
            "median_attention_relative_rms": float(statistics.median(attention_rels)) if attention_rels else None,
            "max_out_proj_relative_rms": max(projected_rels) if projected_rels else None,
            "median_out_proj_relative_rms": float(statistics.median(projected_rels)) if projected_rels else None,
            "max_msa_residual_relative_rms": max(msa_rels) if msa_rels else None,
            "median_msa_residual_relative_rms": float(statistics.median(msa_rels)) if msa_rels else None,
            "max_block_output_relative_rms": max(block_rels) if block_rels else None,
            "median_block_output_relative_rms": float(statistics.median(block_rels)) if block_rels else None,
        },
        "samples": samples,
    }


def _run_tau_sweep(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    lengths: H3PackedLengths,
    args: argparse.Namespace,
    *,
    block: TransformerBlock | None = None,
    x: mx.array | None = None,
    modulation: tuple[mx.array, ...] | None = None,
    adaln_indices: mx.array | None = None,
    position_ids: mx.array | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sweep_items = _sweep_items(args)
    first_item = sweep_items[0]
    base_cfg = h3_macsol_config(
        lengths,
        tau=float(first_item["tau"]),
        block_size=args.block_size,
        routing_mode=str(first_item["routing_mode"]),
    )
    first_routing = build_macsol_routing(q, k, base_cfg, scale=float(q.shape[-1] ** -0.5))
    sample_heads = _auto_sample_heads(int(q.shape[1]), _parse_int_list(args.sample_heads), args.max_sample_heads)
    sample_q_blocks = _auto_sample_q_blocks(
        first_routing.q_block_ranges,
        lengths.prefix_tokens,
        _parse_int_list(args.sample_q_blocks),
        args.max_sample_q_blocks,
    )
    sample_plan = {
        "sample_heads": sample_heads,
        "sample_q_blocks": sample_q_blocks,
        "sample_q_token_ranges": [list(first_routing.q_block_ranges[i]) for i in sample_q_blocks],
        "prefix_tokens": lengths.prefix_tokens,
        "block_size": args.block_size,
        "routing_mode": args.routing_mode,
        "sweep_labels": [_format_sweep_label(item) for item in sweep_items],
    }

    results: list[dict[str, Any]] = []
    for item in sweep_items:
        cfg = MacSolReferenceConfig(
            block_size=args.block_size,
            tau=float(item["tau"]),
            sink_start=0,
            sink_tokens=lengths.prefix_tokens,
            prefix_query_tokens=lengths.prefix_tokens,
            force_prefix_queries_dense=True,
            include_neighbors=True,
            routing_mode=str(item["routing_mode"]),
        )
        before = _metrics()
        started = time.perf_counter()
        routing = build_macsol_routing(q, k, cfg, scale=float(q.shape[-1] ** -0.5))
        routing_elapsed = time.perf_counter() - started
        errors = _sample_attention_errors(
            q,
            k,
            v,
            routing,
            prefix_tokens=lengths.prefix_tokens,
            sample_heads=sample_heads,
            sample_q_blocks=sample_q_blocks,
            scale=float(q.shape[-1] ** -0.5),
        )
        if args.block_output_proxy:
            if block is None or x is None or modulation is None or adaln_indices is None:
                block_output_proxy = {
                    "attempted": False,
                    "skipped": True,
                    "reason": "block/x/modulation/adaln_indices were not supplied to the routing sweep",
                }
            else:
                block_output_proxy = _sample_block_output_proxy(
                    q,
                    k,
                    v,
                    routing,
                    block=block,
                    x=x,
                    modulation=modulation,
                    adaln_indices=adaln_indices,
                    prefix_tokens=lengths.prefix_tokens,
                    sample_heads=sample_heads,
                    sample_q_blocks=sample_q_blocks,
                    scale=float(q.shape[-1] ** -0.5),
                )
        else:
            block_output_proxy = {
                "attempted": False,
                "skipped": True,
                "reason": "run did not request --block-output-proxy",
            }
        after = _metrics()
        stats = routing.summary()
        stats.update(_effective_exact_block_stats(routing, lengths.prefix_tokens))
        results.append(
            {
                "sweep_label": _format_sweep_label(item),
                "routing_mode": item["routing_mode"],
                "tau": item["tau"],
                "routing_build_seconds": routing_elapsed,
                "routing_stats": stats,
                "sampled_attention_errors": errors,
                "block_output_proxy": block_output_proxy,
                "metrics_before": before,
                "metrics_after": after,
                "metrics_delta": _delta(before, after),
            }
        )
    return results, sample_plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/MiniMax-H3-MLX-4bit")
    parser.add_argument("--block-load-mode", default="mlx", choices=("mlx",))
    parser.add_argument("--block-index", type=int, default=10)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--spatial-compression", type=int, default=16)
    parser.add_argument("--sigma-grid-points", type=int, default=5)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--video-sigma-shift", type=float, default=12.0)
    parser.add_argument("--audio-sigma-shift", type=float, default=3.0)
    parser.add_argument("--activation-dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--routing-mode",
        choices=("threshold",),
        default="threshold",
        help="Only active MacSol routing mode; budget_topk, threshold_h3_structure, and spatial-tube variants are not runnable",
    )
    parser.add_argument("--taus", default="0.0,0.5,1.0,1.5,2.0")
    parser.add_argument("--sample-heads", default="auto", help="comma-separated head indices or auto")
    parser.add_argument("--sample-q-blocks", default="auto", help="comma-separated q-block indices or auto")
    parser.add_argument("--max-sample-heads", type=int, default=2)
    parser.add_argument("--max-sample-q-blocks", type=int, default=3)
    parser.add_argument(
        "--block-output-proxy",
        action="store_true",
        help="Also propagate sampled-head attention error through out_proj, gated MSA residual, and row-local MLP for sampled q-block rows",
    )
    parser.add_argument(
        "--dense-fallback-below",
        type=int,
        default=4096,
        help="MacSol engine policy fallback: sequences shorter than this remain dense when --respect-engine-guards is set",
    )
    parser.add_argument("--sensitive-first-layers", type=int, default=1)
    parser.add_argument("--sensitive-last-layers", type=int, default=1)
    parser.add_argument(
        "--respect-engine-guards",
        action="store_true",
        help="Record dense short-sequence and first/last-layer guard behavior instead of forcing sparse replay",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="Use a tiny random block for harness verification only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    _reset_mlx_peak()
    preflight = {
        "recorded_at": _now(),
        "cwd": str(Path.cwd()),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "metrics": _metrics(),
        "process_scan": _process_scan(),
    }

    ok = False
    failure: dict[str, Any] | None = None
    record: dict[str, Any]
    try:
        block, cfg, block_meta = _load_block(args, events)
        inputs_before = _metrics()
        x, modulation, adaln_indices, rotary, position_ids, lengths, sequence_meta = _build_inputs(cfg, args, block, events)
        inputs_after = _metrics()
        engine_policy = _engine_policy(lengths.sequence_length, args.block_index, cfg.num_layers, args)
        events.append({"time": _now(), "message": "evaluated MacSol engine policy", **engine_policy})

        if args.respect_engine_guards and not engine_policy["sparse_replay_allowed"]:
            capture_before = _metrics()
            capture_after = _metrics()
            capture_meta = {
                "attempted": False,
                "skipped": True,
                "reason": engine_policy["reason"],
                "uses_default_dense_generation_path": True,
            }
            replay_before = _metrics()
            tau_results = _dense_guard_sweep_results(_sweep_items(args), engine_policy)
            sample_plan = {
                "sample_heads": [],
                "sample_q_blocks": [],
                "sample_q_token_ranges": [],
                "prefix_tokens": lengths.prefix_tokens,
                "block_size": args.block_size,
                "skipped": True,
                "reason": engine_policy["reason"],
            }
            replay_after = _metrics()
        else:
            capture_before = _metrics()
            q, k, v, capture_meta = _capture_attention_qkv(block, x, modulation, adaln_indices, rotary)
            capture_after = _metrics()
            events.append({"time": _now(), "message": "captured attention qkv", **capture_meta})

            replay_before = _metrics()
            tau_results, sample_plan = _run_tau_sweep(
                q,
                k,
                v,
                lengths,
                args,
                block=block,
                x=x,
                modulation=modulation,
                adaln_indices=adaln_indices,
                position_ids=position_ids,
            )
            replay_after = _metrics()
        ok = True
        record = {
            "ok": ok,
            "recorded_at": _now(),
            "command": command,
            "preflight": preflight,
            "events": events,
            "engine_policy": engine_policy,
            "parameters": {
                "model_dir": args.model_dir,
                "block_index": args.block_index,
                "height": args.height,
                "width": args.width,
                "duration": args.duration,
                "text_tokens": args.text_tokens,
                "sigma_grid_points": args.sigma_grid_points,
                "step_index": args.step_index,
                "activation_dtype": args.activation_dtype,
                "block_size": args.block_size,
                "routing_mode": args.routing_mode,
                "taus": _parse_float_list(args.taus),
                "sweep_labels": [_format_sweep_label(item) for item in _sweep_items(args)],
                "tiny": bool(args.tiny),
                "block_output_proxy": bool(args.block_output_proxy),
                "respect_engine_guards": bool(args.respect_engine_guards),
                "dense_fallback_below": int(args.dense_fallback_below),
                "sensitive_first_layers": int(args.sensitive_first_layers),
                "sensitive_last_layers": int(args.sensitive_last_layers),
            },
            "model": {
                "block_load": block_meta,
                "config": {
                    "hidden_size": cfg.hidden_size,
                    "num_layers": cfg.num_layers,
                    "num_attention_heads": cfg.num_attention_heads,
                    "attention_head_dim": cfg.attention_head_dim,
                    "inner_dim": cfg.inner_dim,
                    "ffn_hidden_size": cfg.ffn_hidden_size,
                    "patch_size": list(cfg.patch_size),
                    "rotary_dim": cfg.rotary_dim,
                    "qk_norm_eps": cfg.qk_norm_eps,
                },
            },
            "sequence": sequence_meta,
            "lengths": lengths.as_dict(),
            "input_build_metrics": {
                "before": inputs_before,
                "after": inputs_after,
                "delta": _delta(inputs_before, inputs_after),
            },
            "qkv_capture": {
                **capture_meta,
                "metrics_before": capture_before,
                "metrics_after": capture_after,
                "metrics_delta": _delta(capture_before, capture_after),
            },
            "sample_plan": sample_plan,
            "tau_results": tau_results,
            "replay_metrics": {
                "before": replay_before,
                "after": replay_after,
                "delta": _delta(replay_before, replay_after),
            },
            "teacher_forced_velocity_error": {
                "attempted": False,
                "feasible_in_this_harness": False,
                "reason": "This opt-in one-block QKV capture does not run the full streaming DiT denoising trajectory or output heads; velocity error requires a separate teacher-forced end-to-end capture under a larger memory budget.",
            },
            "completion_metrics": _metrics(),
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        record = {
            "ok": False,
            "recorded_at": _now(),
            "command": command,
            "preflight": preflight,
            "events": events,
            "failure": failure,
            "completion_metrics": _metrics(),
            "elapsed_seconds": time.perf_counter() - started,
        }

    out.write_text(json.dumps(record, indent=2 if args.pretty else None, sort_keys=True) + "\n")
    if ok:
        best = []
        for item in record["tau_results"]:
            agg = item["sampled_attention_errors"]["aggregate"]
            best.append(
                f"{item.get('sweep_label', _format_sweep_label(item))}: "
                f"exact={item['routing_stats']['exact_block_ratio']:.4f}, "
                f"eff={item['routing_stats']['effective_exact_block_ratio_with_prefix_q_dense']:.4f}, "
                f"sample_rel_rms={agg['median_sample_relative_rms']:.3e}"
            )
        qkv_shape = record["qkv_capture"].get("q_shape", "skipped")
        print(
            "real-QKV MacSol profile ok "
            f"seq={record['sequence']['sequence_length']} qkv={qkv_shape} "
            f"out={out}\n  " + "\n  ".join(best),
            flush=True,
        )
        return 0
    print(f"real-QKV MacSol profile FAILED {failure} out={out}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
