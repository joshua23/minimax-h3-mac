#!/usr/bin/env python3
"""Prove Torch/CUDA optimizer parity on tiny data and one real DiT block layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from activation_quant_torch import (  # noqa: E402
    torch_cuda_activation_aware_affine_quantize,
)
from minimax_h3_mlx.activation_quant import (  # noqa: E402
    ActivationDataset,
    activation_aware_affine_quantize,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(
    weight: mx.array,
    rows: np.ndarray,
    *,
    device: str,
) -> dict[str, Any]:
    cpu_started = time.perf_counter()
    cpu = activation_aware_affine_quantize(weight, rows)
    mx.eval(cpu.weight, cpu.scales, cpu.biases)
    cpu_seconds = time.perf_counter() - cpu_started
    cuda_started = time.perf_counter()
    cuda = torch_cuda_activation_aware_affine_quantize(weight, rows, device=device)
    mx.eval(cuda.weight, cuda.scales, cuda.biases)
    cuda_seconds = time.perf_counter() - cuda_started

    packed_equal = bool(np.array_equal(np.asarray(cuda.weight), np.asarray(cpu.weight)))
    scales_equal = bool(
        np.array_equal(
            np.asarray(cuda.scales.astype(mx.float32)),
            np.asarray(cpu.scales.astype(mx.float32)),
        )
    )
    biases_equal = bool(
        np.array_equal(
            np.asarray(cuda.biases.astype(mx.float32)),
            np.asarray(cpu.biases.astype(mx.float32)),
        )
    )
    cpu_dense = mx.dequantize(
        cpu.weight,
        cpu.scales,
        cpu.biases,
        group_size=32,
        bits=8,
        mode="affine",
        dtype=mx.float32,
    )
    cuda_dense = mx.dequantize(
        cuda.weight,
        cuda.scales,
        cuda.biases,
        group_size=32,
        bits=8,
        mode="affine",
        dtype=mx.float32,
    )
    mx.eval(cpu_dense, cuda_dense)
    cpu_host = np.asarray(cpu_dense)
    cuda_host = np.asarray(cuda_dense)
    delta = cuda_host.astype(np.float64) - cpu_host.astype(np.float64)
    dequant_relative_l2 = float(
        np.linalg.norm(delta.reshape(-1))
        / max(np.linalg.norm(cpu_host.astype(np.float64).reshape(-1)), 1e-24)
    )
    baseline_objective_relative_delta = abs(
        cuda.baseline_error - cpu.baseline_error
    ) / max(abs(cpu.baseline_error), 1e-24)
    candidate_objective_relative_delta = abs(
        cuda.candidate_error - cpu.candidate_error
    ) / max(abs(cpu.candidate_error), 1e-24)
    passed = (
        packed_equal
        and scales_equal
        and biases_equal
        and dequant_relative_l2 == 0.0
        and baseline_objective_relative_delta <= 2e-6
        and candidate_objective_relative_delta <= 2e-6
    )
    return {
        "shape": list(weight.shape),
        "weight_dtype": str(weight.dtype),
        "rows": list(rows.shape),
        "packed_uint32_equal": packed_equal,
        "scales_equal": scales_equal,
        "biases_equal": biases_equal,
        "dequantized_relative_l2": dequant_relative_l2,
        "dequantized_max_abs": float(np.max(np.abs(delta))),
        "cpu": {
            "baseline_error": cpu.baseline_error,
            "candidate_error": cpu.candidate_error,
            "improved_group_fraction": cpu.improved_group_fraction,
            "seconds": cpu_seconds,
        },
        "cuda": {
            "baseline_error": cuda.baseline_error,
            "candidate_error": cuda.candidate_error,
            "improved_group_fraction": cuda.improved_group_fraction,
            "seconds": cuda_seconds,
        },
        "baseline_objective_relative_delta": baseline_objective_relative_delta,
        "candidate_objective_relative_delta": candidate_objective_relative_delta,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="source transformer directory")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--path", default="blocks.0.attn.out_proj")
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    transformer = Path(args.checkpoint)
    dataset_path = Path(args.calibration)
    dataset = ActivationDataset.load(dataset_path)
    weight_key = f"{args.path}.weight"
    with (transformer / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]
    if weight_key not in weight_map:
        parser.error(f"source checkpoint has no {weight_key}")
    shard = transformer / weight_map[weight_key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        source = handle.get_tensor(weight_key)
    source_dtype = source.dtype
    dense = source.float().numpy()
    weight = mx.array(dense)
    if source_dtype == torch.bfloat16:
        weight = weight.astype(mx.bfloat16)
    elif source_dtype != torch.float32:
        parser.error(f"unsupported real source dtype: {source_dtype}")
    rows = dataset.get("calibration", args.path)

    rng = np.random.default_rng(29)
    tiny_weight = mx.array(
        rng.normal(0, 0.2, size=(96, 128)).astype(np.float32)
    ).astype(mx.bfloat16)
    tiny_rows = rng.normal(size=(64, 128)).astype(np.float32)
    tiny = compare(tiny_weight, tiny_rows, device=args.cuda_device)
    real = compare(weight, rows, device=args.cuda_device)
    report = {
        "schema_version": 1,
        "optimizer": "Torch/CUDA vectorized diagonal-Hessian affine INT8",
        "deployment_abi": (
            "MLX affine INT8: four consecutive uint8 codes per uint32, "
            "first code in least-significant byte, original auxiliary dtype"
        ),
        "source": {
            "transformer": str(transformer.resolve()),
            "weight_key": weight_key,
            "source_dtype": str(source_dtype),
            "shard": str(shard.resolve()),
            "shard_sha256": sha256(shard),
            "calibration": str(dataset_path.resolve()),
            "calibration_sha256": sha256(dataset_path),
        },
        "environment": {
            "device": args.cuda_device,
            "gpu": torch.cuda.get_device_name(torch.device(args.cuda_device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "tiny": tiny,
        "real_block_layer": real,
        "passed": bool(tiny["passed"] and real["passed"]),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(f"CUDA optimizer parity failed; report: {output.resolve()}")
    print(
        f"PASS: tiny and real {args.path} objective/packing/dequantized parity; "
        f"report: {output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
