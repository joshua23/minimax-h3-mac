#!/usr/bin/env python3
"""Verify real MiniMax-H3 block-0 parity between PyTorch/CUDA and MLX."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))

from convert_minimax_h3_to_diffusers import (  # noqa: E402
    convert_transformer_key,
    reorder_interleaved_qkv,
)
from diffusers.models.transformers.transformer_minimax_h3 import (  # noqa: E402
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TransformerBlock,
    _apply_rotary_emb,
)
from minimax_h3_mlx.config import DiTConfig  # noqa: E402
from minimax_h3_mlx.dit import (  # noqa: E402
    RotaryPosEmbed3D,
    TransformerBlock,
    apply_rotary,
)

DEFAULT_TOLERANCES = {
    "adaln": {"relative_l2": 0.012, "cosine": 0.999},
    "qkv": {"relative_l2": 0.012, "cosine": 0.999},
    "rope_qk": {"relative_l2": 0.015, "cosine": 0.999},
    "attention": {"relative_l2": 0.025, "cosine": 0.998},
    "out_proj": {"relative_l2": 0.025, "cosine": 0.998},
    "mlp": {"relative_l2": 0.025, "cosine": 0.998},
    "block": {"relative_l2": 0.035, "cosine": 0.997},
}


def as_diffusers_config(config: DiTConfig) -> dict[str, Any]:
    return {
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "attention_head_dim": config.attention_head_dim,
        "ffn_dim": config.ffn_hidden_size,
        "time_embed_dim": config.time_embed_dim,
    }


def source_shard(transformer: Path, keys: list[str]) -> Path:
    with (transformer / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]
    names = {weight_map[key] for key in keys}
    if len(names) != 1:
        raise RuntimeError(f"block 0 unexpectedly spans multiple shards: {sorted(names)}")
    return transformer / names.pop()


def load_real_blocks(
    transformer: Path,
    config: DiTConfig,
    device: torch.device,
) -> tuple[TransformerBlock, MiniMaxH3TransformerBlock, dict[str, Any]]:
    mlx_block = TransformerBlock(config)
    expected_mlx = {key for key, _ in tree_flatten(mlx_block.parameters())}
    source_keys = [f"blocks.0.{key}" for key in sorted(expected_mlx)]
    shard = source_shard(transformer, source_keys)

    loaded_mlx = mx.load(str(shard))
    mlx_updates = [
        (key.removeprefix("blocks.0."), loaded_mlx[key])
        for key in source_keys
    ]
    missing_mlx = sorted(expected_mlx - {key for key, _ in mlx_updates})
    unexpected_mlx = sorted({key for key, _ in mlx_updates} - expected_mlx)
    if missing_mlx or unexpected_mlx:
        raise RuntimeError(
            f"MLX block mapping mismatch: missing={missing_mlx}, unexpected={unexpected_mlx}"
        )
    mlx_block.update(tree_unflatten(mlx_updates))
    mx.eval(mlx_block.parameters())
    del loaded_mlx

    torch_block = MiniMaxH3TransformerBlock(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        attention_head_dim=config.attention_head_dim,
        ffn_dim=config.ffn_hidden_size,
        time_embed_dim=config.time_embed_dim,
        norm_eps=config.norm_eps,
        qk_norm_eps=config.qk_norm_eps,
    ).to(device=device, dtype=torch.bfloat16)
    converted: dict[str, torch.Tensor] = {}
    dcfg = as_diffusers_config(config)
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for source_key in source_keys:
            tensor = handle.get_tensor(source_key)
            if source_key.endswith(".attn.qkv_proj.weight"):
                tensor = reorder_interleaved_qkv(
                    tensor,
                    config.num_attention_heads,
                    config.attention_head_dim,
                )
            for target_key, target_tensor in convert_transformer_key(
                source_key, tensor, dcfg
            ):
                prefix = "transformer_blocks.0."
                if not target_key.startswith(prefix):
                    raise RuntimeError(f"unexpected converted block key: {target_key}")
                converted[target_key.removeprefix(prefix)] = target_tensor
    incompatible = torch_block.load_state_dict(converted, strict=True)
    missing_torch = list(incompatible.missing_keys)
    unexpected_torch = list(incompatible.unexpected_keys)
    if missing_torch or unexpected_torch:
        raise RuntimeError(
            f"PyTorch block mapping mismatch: missing={missing_torch}, "
            f"unexpected={unexpected_torch}"
        )
    torch_block.eval()
    return mlx_block, torch_block, {
        "source_shard": shard.name,
        "source_key_count": len(source_keys),
        "mlx_missing_keys": missing_mlx,
        "mlx_unexpected_keys": unexpected_mlx,
        "torch_missing_keys": missing_torch,
        "torch_unexpected_keys": unexpected_torch,
    }


def as_numpy(value: mx.array | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def metric(
    reference: torch.Tensor,
    actual: mx.array,
    tolerance: dict[str, float],
) -> dict[str, Any]:
    left = as_numpy(reference).astype(np.float64, copy=False)
    right = as_numpy(actual).astype(np.float64, copy=False)
    if left.shape != right.shape:
        raise ValueError(f"metric shape mismatch: {left.shape} != {right.shape}")
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    delta = right - left
    reference_norm = float(np.linalg.norm(left.reshape(-1)))
    delta_norm = float(np.linalg.norm(delta.reshape(-1)))
    denominator = max(reference_norm, 1e-24)
    cosine_denominator = max(
        float(np.linalg.norm(left.reshape(-1)) * np.linalg.norm(right.reshape(-1))),
        1e-24,
    )
    cosine = float(np.dot(left.reshape(-1), right.reshape(-1)) / cosine_denominator)
    relative_l2 = delta_norm / denominator
    passed = (
        finite
        and relative_l2 <= tolerance["relative_l2"]
        and cosine >= tolerance["cosine"]
    )
    return {
        "shape": list(left.shape),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "finite": finite,
        "tolerance": tolerance,
        "passed": passed,
    }


def run_probe(
    mlx_block: TransformerBlock,
    torch_block: MiniMaxH3TransformerBlock,
    config: DiTConfig,
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    batch, sequence, timestep_count = 1, 13, 3
    hidden_host = rng.standard_normal(
        (batch, sequence, config.hidden_size), dtype=np.float32
    )
    temb_host = rng.standard_normal(
        (timestep_count, config.time_embed_dim), dtype=np.float32
    )
    position_host = np.stack(
        (
            np.arange(sequence) % 3,
            np.arange(sequence) % 5,
            np.arange(sequence) % 7,
        ),
        axis=-1,
    ).astype(np.float32)
    adaln_indices_host = (
        np.arange(sequence, dtype=np.int32) % (timestep_count * 3)
    )

    hidden_t = torch.from_numpy(hidden_host).to(device=device, dtype=torch.bfloat16)
    temb_t = torch.from_numpy(temb_host).to(device=device, dtype=torch.float32)
    position_t = torch.from_numpy(position_host).to(device=device, dtype=torch.float32)
    indices_t = torch.from_numpy(adaln_indices_host.astype(np.int64)).to(device)
    hidden_m = mx.array(hidden_host).astype(mx.bfloat16)
    temb_m = mx.array(temb_host, dtype=mx.float32)
    position_m = mx.array(position_host, dtype=mx.float32)
    indices_m = mx.array(adaln_indices_host, dtype=mx.int32)

    torch_rope = MiniMaxH3RotaryPosEmbed(
        rope_freq_dim=config.rope_inv_freq_len,
        rope_theta=config.rope_theta,
    ).to(device)
    mlx_rope = RotaryPosEmbed3D(config)

    with torch.inference_mode():
        modulation_t = torch_block.adaln_proj(temb_t)
        modulation_m = mlx_block.adaln_proj(temb_m)

        norm1_t = torch_block.norm1(hidden_t)
        norm1_m = mlx_block.norm1(hidden_m)
        h_attn_t = (
            norm1_t * (1.0 + modulation_t[1].index_select(0, indices_t))
            + modulation_t[0].index_select(0, indices_t)
        )
        h_attn_m = norm1_m * (1.0 + modulation_m[1][indices_m]) + modulation_m[0][
            indices_m
        ]

        q_t = torch_block.attn.to_q(h_attn_t).unflatten(
            -1, (config.num_attention_heads, config.attention_head_dim)
        )
        k_t = torch_block.attn.to_k(h_attn_t).unflatten(
            -1, (config.num_attention_heads, config.attention_head_dim)
        )
        v_t = torch_block.attn.to_v(h_attn_t).unflatten(
            -1, (config.num_attention_heads, config.attention_head_dim)
        )
        qkv_m = mlx_block.attn.qkv_proj(h_attn_m).reshape(
            batch,
            sequence,
            config.num_attention_heads,
            3,
            config.attention_head_dim,
        )
        q_m, k_m, v_m = qkv_m[:, :, :, 0], qkv_m[:, :, :, 1], qkv_m[:, :, :, 2]

        qn_t = torch_block.attn.norm_q(q_t)
        kn_t = torch_block.attn.norm_k(k_t)
        qn_m = mlx_block.attn.q_norm(q_m).transpose(0, 2, 1, 3)
        kn_m = mlx_block.attn.k_norm(k_m).transpose(0, 2, 1, 3)
        v_sdpa_m = v_m.transpose(0, 2, 1, 3)

        rotary_t = torch_rope(position_t)
        rotary_m = mlx_rope(position_m)
        qr_t = _apply_rotary_emb(qn_t, *rotary_t).permute(0, 2, 1, 3)
        kr_t = _apply_rotary_emb(kn_t, *rotary_t).permute(0, 2, 1, 3)
        qr_m = apply_rotary(qn_m, *rotary_m)
        kr_m = apply_rotary(kn_m, *rotary_m)

        attention_t = F.scaled_dot_product_attention(
            qr_t,
            kr_t,
            v_t.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
            scale=config.attention_head_dim**-0.5,
        )
        attention_t = attention_t.permute(0, 2, 1, 3).flatten(2, 3)
        attention_m = mx.fast.scaled_dot_product_attention(
            qr_m,
            kr_m,
            v_sdpa_m,
            scale=config.attention_head_dim**-0.5,
        ).transpose(0, 2, 1, 3).reshape(batch, sequence, config.inner_dim)

        out_t = torch_block.attn.to_out[0](attention_t.to(h_attn_t.dtype))
        out_m = mlx_block.attn.out_proj(attention_m.astype(h_attn_m.dtype))
        after_attn_t = hidden_t + modulation_t[2].index_select(0, indices_t) * out_t
        after_attn_m = hidden_m + modulation_m[2][indices_m] * out_m

        norm2_t = torch_block.norm2(after_attn_t)
        norm2_m = mlx_block.norm2(after_attn_m)
        h_mlp_t = (
            norm2_t * (1.0 + modulation_t[4].index_select(0, indices_t))
            + modulation_t[3].index_select(0, indices_t)
        )
        h_mlp_m = norm2_m * (1.0 + modulation_m[4][indices_m]) + modulation_m[3][
            indices_m
        ]
        fused_t = torch_block.ff.net[0].proj(h_mlp_t)
        value_t, gate_t = fused_t.chunk(2, dim=-1)
        mlp_hidden_t = value_t * F.silu(gate_t)
        mlp_t = torch_block.ff.net[2](mlp_hidden_t)
        fused_m = mlx_block.mlp.fc1(h_mlp_m)
        gate_m, value_m = mx.split(fused_m, 2, axis=-1)
        mlp_hidden_m = nn.silu(gate_m) * value_m
        mlp_m = mlx_block.mlp.fc2(mlp_hidden_m)

        block_t = torch_block(
            hidden_t,
            temb_t,
            indices_t,
            rotary_t,
            attention_mask=None,
        )
        block_m = mlx_block(
            hidden_m,
            modulation_m,
            indices_m,
            rotary_m,
            mask=None,
        )

    stages = {
        "adaln": (
            torch.cat(modulation_t, dim=-1),
            mx.concatenate(modulation_m, axis=-1),
        ),
        "qkv": (
            torch.cat((q_t, k_t, v_t), dim=-1),
            mx.concatenate((q_m, k_m, v_m), axis=-1),
        ),
        "rope_qk": (
            torch.cat((qr_t, kr_t), dim=-1),
            mx.concatenate((qr_m, kr_m), axis=-1),
        ),
        "attention": (attention_t, attention_m),
        "out_proj": (out_t, out_m),
        "mlp": (
            torch.cat((mlp_hidden_t, mlp_t), dim=-1),
            mx.concatenate((mlp_hidden_m, mlp_m), axis=-1),
        ),
        "block": (block_t, block_m),
    }
    return {
        name: metric(reference, actual, DEFAULT_TOLERANCES[name])
        for name, (reference, actual) in stages.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="pinned FL2VA directory")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        parser.error("real parity must run with CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    started = time.perf_counter()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    transformer = Path(args.checkpoint) / "transformer"
    config = DiTConfig.from_json(transformer / "config.json")
    mlx_block, torch_block, mapping = load_real_blocks(transformer, config, device)
    metrics = run_probe(mlx_block, torch_block, config, device, args.seed)
    torch.cuda.synchronize()

    report = {
        "schema_version": 1,
        "source": {
            "model": "MiniMaxAI/MiniMax-H3",
            "revision": args.source_revision,
            "checkpoint": str(Path(args.checkpoint).resolve()),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mlx_backend": str(mx.default_device()),
        },
        "mapping": mapping,
        "input_contract": {
            "seed": args.seed,
            "hidden": "host-generated FP32, cast independently to BF16",
            "temb": "host-generated FP32",
            "position_ids": "host-generated FP32 integer coordinates",
            "block": 0,
        },
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "passed": all(record["passed"] for record in metrics.values()),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for name, record in metrics.items():
        print(
            f"{name}: max_abs={record['max_abs']:.6e} "
            f"relative_l2={record['relative_l2']:.6e} "
            f"cosine={record['cosine']:.9f} finite={record['finite']} "
            f"{'PASS' if record['passed'] else 'FAIL'}",
            flush=True,
        )
    if not report["passed"]:
        raise RuntimeError(f"real block-0 parity failed; report: {output}")
    print(
        f"PASS: real block-0 PyTorch/CUDA-to-MLX parity; report: {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
