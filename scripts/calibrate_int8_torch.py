#!/usr/bin/env python3
"""Collect real MiniMax-H3 INT8 calibration activations with a CUDA teacher."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel,
    Qwen3VLTextRotaryEmbedding,
    create_causal_mask,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))

from calibrate_int8 import CASES, validate_cases  # noqa: E402
from convert_minimax_h3_to_diffusers import (  # noqa: E402
    convert_transformer_key,
    reorder_interleaved_qkv,
)
from diffusers.models.transformers.transformer_minimax_h3 import (  # noqa: E402
    MiniMaxH3Transformer3DModel,
)
from eval_quant import build_case, timestep_plan  # noqa: E402
from minimax_h3_mlx.activation_quant import ActivationDataset  # noqa: E402

TEXT_ENCODER_LAYER = 50
TEXT_PARITY_TOLERANCE = {"relative_l2": 0.08, "cosine": 0.997}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def set_parameter(module: torch.nn.Module, path: str, value: torch.Tensor) -> None:
    parent: Any = module
    parts = path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    name = parts[-1]
    if name not in parent._parameters:
        raise KeyError(f"{path!r} is not a parameter")
    requires_grad = parent._parameters[name].requires_grad
    parent._parameters[name] = torch.nn.Parameter(value, requires_grad=requires_grad)


def weight_map(model_dir: Path) -> dict[str, str]:
    with (model_dir / "model.safetensors.index.json").open() as handle:
        return json.load(handle)["weight_map"]


def shard_keys(index: dict[str, str]) -> list[tuple[str, list[str]]]:
    names = sorted(set(index.values()))
    return [
        (name, sorted(key for key, shard in index.items() if shard == name))
        for name in names
    ]


def load_torch_text_encoder(
    model_dir: Path,
    device: torch.device,
) -> tuple[Qwen3VLTextModel, dict[str, Any]]:
    with (model_dir / "config.json").open() as handle:
        raw = json.load(handle)
    text_raw = dict(raw["text_config"])
    full_layers = int(text_raw["num_hidden_layers"])
    if full_layers <= TEXT_ENCODER_LAYER:
        raise ValueError(
            f"text encoder has {full_layers} layers, expected more than {TEXT_ENCODER_LAYER}"
        )
    text_raw["num_hidden_layers"] = TEXT_ENCODER_LAYER
    config = Qwen3VLTextConfig.from_dict(text_raw)
    with torch.device("meta"):
        model = Qwen3VLTextModel(config)
    expected = {name for name, _ in model.named_parameters()}
    assigned: set[str] = set()
    index = weight_map(model_dir)
    prefix = "model.language_model."
    started = time.perf_counter()
    for shard_index, (name, keys) in enumerate(shard_keys(index), start=1):
        selected = [
            key
            for key in keys
            if key.startswith(prefix) and key.removeprefix(prefix) in expected
        ]
        if not selected:
            continue
        with safe_open(model_dir / name, framework="pt", device="cpu") as handle:
            for source_key in selected:
                target_key = source_key.removeprefix(prefix)
                set_parameter(model, target_key, handle.get_tensor(source_key).to(device))
                assigned.add(target_key)
        print(
            f"text shard {shard_index}/{len(set(index.values()))}: {name}, "
            f"{len(selected)} tensors",
            flush=True,
        )
    missing = sorted(expected - assigned)
    unexpected = sorted(assigned - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"text checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    model.rotary_emb = Qwen3VLTextRotaryEmbedding(config, device=device)
    model.eval()
    return model, {
        "full_layers": full_layers,
        "loaded_layers": TEXT_ENCODER_LAYER,
        "loaded_parameter_tensors": len(assigned),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "load_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def encode_text(
    model: Qwen3VLTextModel,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    hidden = model.embed_tokens(input_ids)
    sequence = int(hidden.shape[1])
    position_ids = (
        torch.arange(sequence, device=hidden.device, dtype=torch.long)
        .view(1, 1, sequence)
        .expand(3, int(hidden.shape[0]), sequence)
    )
    attention_mask = create_causal_mask(
        config=model.config,
        inputs_embeds=hidden,
        attention_mask=torch.ones_like(input_ids),
        past_key_values=None,
        position_ids=None,
    )
    position_embeddings = model.rotary_emb(hidden, position_ids)
    for layer in model.layers:
        hidden = layer(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
        )
    return hidden


def collect_text_embeddings(
    source: Path,
    mlx_text_encoder: Path | None,
    device: torch.device,
    mlx_parity_prompts: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, list[int]], dict[str, Any]]:
    source_tokenizer = AutoTokenizer.from_pretrained(str(source / "tokenizer"))
    encoder_tokenizer = AutoTokenizer.from_pretrained(str(source / "text_encoder"))
    token_ids: dict[str, list[int]] = {}
    for case in CASES:
        prompt = str(case["prompt"])
        if prompt in token_ids:
            continue
        left = source_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        right = encoder_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if left != right:
            raise RuntimeError(f"token IDs differ between source tokenizer assets for {prompt!r}")
        if not left:
            raise RuntimeError(f"prompt produced no token IDs: {prompt!r}")
        token_ids[prompt] = [int(value) for value in left]

    torch.cuda.reset_peak_memory_stats()
    model, load_report = load_torch_text_encoder(source / "text_encoder", device)
    embeddings: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for prompt, ids in token_ids.items():
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        hidden = encode_text(model, input_ids)
        embeddings[prompt] = hidden.float().cpu().numpy()
        print(f"CUDA text embedding: {prompt[:48]!r}, {len(ids)} tokens", flush=True)
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    parity: dict[str, Any] = {
        "performed": False,
        "comparison": None,
        "tolerance": TEXT_PARITY_TOLERANCE,
        "prompts": {},
        "passed": None,
    }
    if mlx_text_encoder is not None:
        from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

        parity["performed"] = True
        parity["comparison"] = (
            "pinned BF16 PyTorch source text encoder versus deployed MLX INT8 text encoder"
        )
        mlx_encoder = MiniMaxH3TextEncoder(
            mlx_text_encoder,
            load_vision=False,
            verbose=True,
            tokenizer_dir=source / "tokenizer",
            processor_dir=source / "processor",
        )
        if mlx_parity_prompts <= 0:
            raise ValueError("mlx_parity_prompts must be positive when an MLX encoder is supplied")
        selected_embeddings = list(embeddings.items())[:mlx_parity_prompts]
        passed = True
        for prompt, reference in selected_embeddings:
            mlx_ids, tags, _ = mlx_encoder.build_request(prompt)
            ids_match = np.asarray(mlx_ids).reshape(-1).tolist() == token_ids[prompt]
            if not ids_match:
                raise RuntimeError(f"PyTorch/MLX token IDs differ for {prompt!r}")
            actual_mx, actual_tags = mlx_encoder.encode(prompt)
            actual = np.asarray(actual_mx.astype(mx.float32))
            reference64 = reference.astype(np.float64, copy=False)
            actual64 = actual.astype(np.float64, copy=False)
            delta = actual64 - reference64
            relative_l2 = float(
                np.linalg.norm(delta.reshape(-1))
                / max(np.linalg.norm(reference64.reshape(-1)), 1e-24)
            )
            cosine = float(
                np.dot(actual64.reshape(-1), reference64.reshape(-1))
                / max(
                    np.linalg.norm(actual64.reshape(-1))
                    * np.linalg.norm(reference64.reshape(-1)),
                    1e-24,
                )
            )
            finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
            prompt_passed = (
                ids_match
                and np.array_equal(tags, actual_tags)
                and finite
                and relative_l2 <= TEXT_PARITY_TOLERANCE["relative_l2"]
                and cosine >= TEXT_PARITY_TOLERANCE["cosine"]
            )
            parity["prompts"][prompt] = {
                "token_ids_exact": ids_match,
                "token_tags_exact": bool(np.array_equal(tags, actual_tags)),
                "max_abs": float(np.max(np.abs(delta))),
                "relative_l2": relative_l2,
                "cosine": cosine,
                "finite": finite,
                "passed": prompt_passed,
            }
            passed = passed and prompt_passed
            print(
                f"MLX text parity: {prompt[:32]!r} rel={relative_l2:.6e} "
                f"cos={cosine:.9f} {'PASS' if prompt_passed else 'FAIL'}",
                flush=True,
            )
        parity["passed"] = passed
        del mlx_encoder
        mx.clear_cache()
        if not passed:
            raise RuntimeError("real text embedding parity gate failed")

    return embeddings, token_ids, {
        "token_ids_exact_between_source_assets": True,
        "torch_loader": load_report,
        "encode_seconds": encode_seconds,
        "peak_cuda_memory_bytes": peak_memory,
        "parity": parity,
    }


def diffusers_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "num_refiner_layers": config["token_refiner_num_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "attention_head_dim": config["attention_head_dim"],
        "ffn_dim": config["ffn_hidden_size"],
        "in_channels": config["latents_dim"],
        "audio_in_channels": config["audio_latents_dim"],
        "patch_size": tuple(config["patch_size"]),
        "text_dim": config["text_dim"],
        "freq_dim": config["timestep_input_dim"],
        "time_embed_hidden_dim": config["time_embed_hidden_size"],
        "time_embed_dim": config["time_embed_dim"],
        "rope_freq_dim": config["rope_inv_freq_len"],
        "rope_theta": config.get("rope_theta", 10000.0),
        "norm_eps": config["norm_eps"],
        "qk_norm_eps": config["qk_norm_eps"],
        "final_norm_eps": config["final_norm_eps"],
    }


def load_torch_dit(
    transformer: Path,
    device: torch.device,
) -> tuple[MiniMaxH3Transformer3DModel, dict[str, Any]]:
    with (transformer / "config.json").open() as handle:
        source_config = json.load(handle)
    config = diffusers_config(source_config)
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(**config)
    expected = {name for name, _ in model.named_parameters()}
    assigned: set[str] = set()
    dropped: set[str] = set()
    index = weight_map(transformer)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    groups = shard_keys(index)
    for shard_index, (name, keys) in enumerate(groups, start=1):
        with safe_open(transformer / name, framework="pt", device="cpu") as handle:
            for source_key in keys:
                tensor = handle.get_tensor(source_key)
                if source_key.endswith(".attn.qkv_proj.weight"):
                    tensor = reorder_interleaved_qkv(
                        tensor,
                        config["num_attention_heads"],
                        config["attention_head_dim"],
                    )
                converted = convert_transformer_key(source_key, tensor, config)
                if not converted:
                    dropped.add(source_key)
                    continue
                for target_key, target_tensor in converted:
                    if target_key not in expected:
                        raise KeyError(f"converted unexpected DiT key: {target_key}")
                    set_parameter(
                        model,
                        target_key,
                        target_tensor.to(device=device, dtype=torch.float32),
                    )
                    assigned.add(target_key)
        print(
            f"DiT shard {shard_index}/{len(groups)}: {name}, "
            f"{len(keys)} source tensors",
            flush=True,
        )
        gc.collect()
    missing = sorted(expected - assigned)
    unexpected = sorted(assigned - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"DiT checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    n = int(config["rope_freq_dim"])
    inv_freq = 1.0 / (
        float(config["rope_theta"])
        ** (
            torch.arange(0, 2 * n, 2, dtype=torch.float32, device=device)
            / (2 * n)
        )
    )
    model.rope.inv_freq = inv_freq
    model.eval()
    non_fp32 = [
        f"{name}:{value.dtype}"
        for name, value in model.named_parameters()
        if value.dtype != torch.float32
    ]
    if non_fp32:
        raise RuntimeError(f"DiT teacher retained non-FP32 parameters: {non_fp32[:8]}")
    meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]
    meta_buffers = [name for name, value in model.named_buffers() if value.is_meta]
    if meta_parameters or meta_buffers:
        raise RuntimeError(
            f"streamed DiT retained meta tensors: params={meta_parameters[:4]}, "
            f"buffers={meta_buffers[:4]}"
        )
    return model, {
        "source_tensors": len(index),
        "assigned_parameter_tensors": len(assigned),
        "dropped_source_keys": sorted(dropped),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "compute_dtype": "float32",
        "load_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }


class TorchActivationRecorder:
    def __init__(self, max_rows_per_call: int, max_rows_per_layer: int) -> None:
        self.max_rows_per_call = int(max_rows_per_call)
        self.max_rows_per_layer = int(max_rows_per_layer)
        self.split = "calibration"
        self.active = False
        self.include_adaln = False
        self.rows: dict[str, dict[str, list[np.ndarray]]] = {}

    def set_split(self, split: str) -> None:
        self.split = split

    def record(self, path: str, value: torch.Tensor) -> None:
        if not self.active:
            return
        is_adaln = ".adaln_proj.linear" in path
        if is_adaln != self.include_adaln:
            return
        flat = value.detach().reshape(-1, value.shape[-1])
        count = min(int(flat.shape[0]), self.max_rows_per_call)
        if count == 0:
            return
        if int(flat.shape[0]) == count:
            selected = flat
        else:
            indices = torch.linspace(
                0,
                int(flat.shape[0]) - 1,
                count,
                device=flat.device,
            ).to(torch.long)
            selected = flat.index_select(0, indices)
        host = selected.float().cpu().numpy()
        bucket = self.rows.setdefault(self.split, {}).setdefault(path, [])
        retained = sum(chunk.shape[0] for chunk in bucket)
        remaining = self.max_rows_per_layer - retained
        if remaining > 0:
            bucket.append(np.array(host[:remaining], copy=True))

    def dataset(self, manifest: dict[str, Any]) -> ActivationDataset:
        rows = {
            split: {
                path: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
                for path, chunks in paths.items()
                if chunks
            }
            for split, paths in self.rows.items()
        }
        return ActivationDataset(manifest, rows)


def register_activation_hooks(
    model: MiniMaxH3Transformer3DModel,
    recorder: TorchActivationRecorder,
) -> list[Any]:
    handles = []

    def hook(path: str):
        def capture(_module, inputs):
            recorder.record(path, inputs[0])

        return capture

    for index, block in enumerate(model.transformer_blocks):
        mappings = (
            (block.adaln_proj.linear, f"blocks.{index}.adaln_proj.linear"),
            (block.attn.to_q, f"blocks.{index}.attn.qkv_proj"),
            (block.attn.to_out[0], f"blocks.{index}.attn.out_proj"),
            (block.ff.net[0].proj, f"blocks.{index}.mlp.fc1"),
            (block.ff.net[2], f"blocks.{index}.mlp.fc2"),
        )
        handles.extend(module.register_forward_pre_hook(hook(path)) for module, path in mappings)
    for index, block in enumerate(model.token_refiner.refiner_blocks):
        mappings = (
            (block.attn.to_q, f"token_refiner.blocks.{index}.attn.qkv_proj"),
            (block.attn.to_out[0], f"token_refiner.blocks.{index}.attn.out_proj"),
            (block.ff.net[0].proj, f"token_refiner.blocks.{index}.mlp.fc1"),
            (block.ff.net[2], f"token_refiner.blocks.{index}.mlp.fc2"),
        )
        handles.extend(module.register_forward_pre_hook(hook(path)) for module, path in mappings)
    return handles


def torch_layout(layout, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "position_ids": torch.from_numpy(np.asarray(layout.position_ids)).to(
            device=device, dtype=torch.float32
        ),
        "token_tags": torch.from_numpy(np.asarray(layout.token_tags)).to(
            device=device, dtype=torch.long
        ),
        "video_indices": torch.from_numpy(np.asarray(layout.video_indices)).to(
            device=device, dtype=torch.long
        ),
        "audio_indices": torch.from_numpy(np.asarray(layout.audio_indices)).to(
            device=device, dtype=torch.long
        ),
        "text_indices": torch.from_numpy(np.asarray(layout.text_indices)).to(
            device=device, dtype=torch.long
        ),
    }


def scheduler_step(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: float,
    sigmas: list[float],
    index: int,
) -> torch.Tensor:
    timestep32 = np.float32(timestep)
    sigma_from_timestep = float(np.float32(1.0) - timestep32)
    denoised = sample + sigma_from_timestep * model_output.float()
    sigma = np.float32(sigmas[index])
    sigma_next = np.float32(sigmas[index + 1])
    ratio = sigma_next / sigma
    one_minus_ratio = float(np.float32(1.0) - ratio)
    return float(ratio) * sample.float() + one_minus_ratio * denoised


@torch.inference_mode()
def collect_dit_activations(
    model: MiniMaxH3Transformer3DModel,
    embeddings: dict[str, np.ndarray],
    recorder: TorchActivationRecorder,
    *,
    steps: int,
    duration: float,
    split_timesteps: dict[str, list[float]],
    device: torch.device,
) -> None:
    config = model.config
    for case_index, case in enumerate(CASES):
        prompt = str(case["prompt"])
        embeds = torch.from_numpy(embeddings[prompt]).to(device=device, dtype=torch.float32)
        layout, video_rows_mx, audio_rows_mx, video_sched, audio_sched = build_case(
            len(embeddings[prompt][0]),
            int(case["height"]),
            int(case["width"]),
            duration,
            steps,
            int(config.in_channels),
            int(config.audio_in_channels),
            tuple(config.patch_size),
            int(case["seed"]),
        )
        table_mx, plan_mx = timestep_plan(layout, video_sched, audio_sched)
        layout_t = torch_layout(layout, device)
        table = torch.from_numpy(np.asarray(table_mx)).to(device=device, dtype=torch.float32)
        plans = [
            torch.from_numpy(np.asarray(indices)).to(device=device, dtype=torch.long)
            for indices in plan_mx
        ]
        video_rows = torch.from_numpy(np.asarray(video_rows_mx)).to(
            device=device, dtype=torch.float32
        )
        audio_rows = torch.from_numpy(np.asarray(audio_rows_mx)).to(
            device=device, dtype=torch.float32
        )
        video_timesteps = [float(value) for value in video_sched.timesteps.tolist()]
        audio_timesteps = [float(value) for value in audio_sched.timesteps.tolist()]
        video_sigmas = [float(value) for value in video_sched.sigmas.tolist()]
        audio_sigmas = [float(value) for value in audio_sched.sigmas.tolist()]
        selected_step = int(case["step_index"])
        recorder.set_split(str(case["split"]))
        recorder.include_adaln = False
        for step_index, video_timestep in enumerate(video_timesteps):
            recorder.active = step_index == selected_step
            video_pred, audio_pred = model(
                hidden_states=video_rows[None],
                audio_hidden_states=audio_rows[None],
                encoder_hidden_states=embeds,
                timestep=table,
                timestep_indices=plans[step_index],
                token_tags=layout_t["token_tags"],
                position_ids=layout_t["position_ids"],
                video_indices=layout_t["video_indices"],
                audio_indices=layout_t["audio_indices"],
                text_indices=layout_t["text_indices"],
                return_dict=False,
            )
            recorder.active = False
            if step_index == selected_step:
                break
            video_rows = scheduler_step(
                video_rows,
                video_pred[0],
                video_timestep,
                video_sigmas,
                step_index,
            )
            audio_rows = scheduler_step(
                audio_rows,
                audio_pred[0],
                audio_timesteps[step_index],
                audio_sigmas,
                step_index,
            )
        print(
            f"CUDA case {case_index + 1}/{len(CASES)} {case['split']}: "
            f"seed={case['seed']} shape={case['width']}x{case['height']} "
            f"step={selected_step} rows={layout.sequence_length}",
            flush=True,
        )

    recorder.include_adaln = True
    for split in ("calibration", "holdout"):
        recorder.set_split(split)
        timesteps = torch.tensor(
            split_timesteps[split],
            dtype=torch.float32,
            device=device,
        )
        temb = model.time_embedder(
            model.time_proj(timesteps).to(model.time_embedder.linear_1.weight.dtype)
        )
        recorder.active = True
        for block in model.transformer_blocks:
            block.adaln_proj(temb)
        recorder.active = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="pinned FL2VA directory")
    parser.add_argument("--mlx-text-encoder")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--duration", type=float, default=0.2)
    parser.add_argument("--max-rows-per-call", type=int, default=8)
    parser.add_argument("--max-rows-per-layer", type=int, default=64)
    parser.add_argument(
        "--mlx-parity-prompts",
        type=int,
        default=1,
        help="number of real prompts checked against the slower MLX text encoder",
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        parser.error("CUDA collector must run with CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    split_timesteps = validate_cases(args.steps)

    started = time.perf_counter()
    source = Path(args.checkpoint)
    transformer = source / "transformer"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    embeddings, token_ids, text_report = collect_text_embeddings(
        source,
        Path(args.mlx_text_encoder) if args.mlx_text_encoder else None,
        device,
        mlx_parity_prompts=args.mlx_parity_prompts,
    )
    if torch.cuda.memory_allocated() > 1024**3:
        raise RuntimeError(
            "text encoder was not fully released before DiT load: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB remains"
        )

    model, dit_load_report = load_torch_dit(transformer, device)
    recorder = TorchActivationRecorder(
        args.max_rows_per_call,
        args.max_rows_per_layer,
    )
    handles = register_activation_hooks(model, recorder)
    torch.cuda.reset_peak_memory_stats()
    collection_started = time.perf_counter()
    collect_dit_activations(
        model,
        embeddings,
        recorder,
        steps=args.steps,
        duration=args.duration,
        split_timesteps=split_timesteps,
        device=device,
    )
    torch.cuda.synchronize()
    collection_seconds = time.perf_counter() - collection_started
    collection_peak = torch.cuda.max_memory_allocated()
    for handle in handles:
        handle.remove()

    manifest = {
        "schema_version": 1,
        "algorithm": "diagonal-hessian-activation-aware-affine-int8",
        "bits": 8,
        "group_size": 32,
        "mode": "affine",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "model": "MiniMaxAI/MiniMax-H3",
            "revision": args.source_revision,
            "transformer_config_sha256": sha256(transformer / "config.json"),
        },
        "implementation": {
            "repository": "https://github.com/Argus-AiTeam/minimax-h3-mac",
            **repository_state(),
        },
        "collector": {
            "framework": "PyTorch/CUDA FP32 DiT teacher with one-time BF16 conditioner",
            "script": "scripts/calibrate_int8_torch.py",
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformer_mapping": (
                "official convert_transformer_key + reorder_interleaved_qkv; "
                "PyTorch hooks mapped back to original MLX module paths"
            ),
            "dit_load": dit_load_report,
            "text_encoder": text_report,
        },
        "sampling": {
            "steps": args.steps,
            "duration_seconds": args.duration,
            "max_rows_per_call": args.max_rows_per_call,
            "max_rows_per_layer": args.max_rows_per_layer,
            "selection": "evenly spaced rows from each real layer input",
            "adaln_selection": (
                "real video/audio scheduler timesteps at preregistered case indices, "
                "recorded separately from the full per-forward timestep table"
            ),
            "adaln_split_timesteps": split_timesteps,
        },
        "cases": [dict(case) for case in CASES],
        "token_ids": token_ids,
        "split_contract": (
            "calibration and holdout prompts, seeds, and recorded AdaLN timestep inputs are disjoint"
        ),
    }
    dataset = recorder.dataset(manifest)
    shared = dataset.paths("calibration") & dataset.paths("holdout")
    if len(shared) != 258:
        raise RuntimeError(
            f"captured {len(shared)} shared calibration/holdout layers; expected exactly 258"
        )
    non_finite = [
        f"{split}:{path}"
        for split, paths in dataset.rows.items()
        for path, rows in paths.items()
        if not np.isfinite(rows).all()
    ]
    if non_finite:
        raise RuntimeError(f"non-finite activation rows: {non_finite[:8]}")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output)
    report = {
        "schema_version": 1,
        "dataset": str(output.resolve()),
        "dataset_sha256": sha256(output),
        "calibration_layers": len(dataset.paths("calibration")),
        "holdout_layers": len(dataset.paths("holdout")),
        "shared_layers": len(shared),
        "all_finite": True,
        "text": text_report,
        "dit_load": dit_load_report,
        "collection_seconds": collection_seconds,
        "peak_cuda_memory_bytes": collection_peak,
        "total_seconds": time.perf_counter() - started,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "passed": True,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PASS: wrote {output}, {len(shared)} finite shared layers, "
        f"sha256={report['dataset_sha256']}",
        flush=True,
    )
    print(f"report: {report_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
