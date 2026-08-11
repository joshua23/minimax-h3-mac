"""Single-layer Torch/CUDA optimizer for MLX-native affine INT8 weights."""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from minimax_h3_mlx.activation_quant import ActivationAwareResult


def _torch_round(values: torch.Tensor, dtype: mx.Dtype) -> torch.Tensor:
    if dtype == mx.bfloat16:
        return values.to(torch.bfloat16).float()
    if dtype == mx.float16:
        return values.to(torch.float16).float()
    if dtype == mx.float32:
        return values.float()
    raise TypeError(f"unsupported MLX affine auxiliary dtype: {dtype}")


def _weighted_error(
    weight: torch.Tensor,
    reconstructed: torch.Tensor,
    hessian: torch.Tensor,
) -> torch.Tensor:
    delta = weight.double() - reconstructed.double()
    hessian64 = hessian.double()
    result = torch.zeros(delta.shape[:-1], dtype=torch.float64, device=delta.device)
    for index in range(delta.shape[-1]):
        result = result + delta[..., index].square() * hessian64[None, :, index]
    return result


def _hessian_diagonal(rows: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(rows.shape[1], dtype=torch.float64, device=rows.device)
    for row in rows:
        values = row.double()
        result = result + values * values
    return result / rows.shape[0]


def torch_cuda_activation_aware_affine_quantize(
    weight: mx.array,
    activation_rows: np.ndarray,
    *,
    bits: int = 8,
    group_size: int = 32,
    iterations: int = 3,
    device: str | torch.device = "cuda:0",
) -> ActivationAwareResult:
    """Fit one dense layer on CUDA and return the exact MLX packed-weight ABI."""

    if bits != 8 or group_size != 32:
        raise ValueError("the calibrated MiniMax-H3 delivery requires INT8 group_size=32")
    if weight.ndim != 2 or int(weight.shape[-1]) % group_size:
        raise ValueError(f"invalid weight shape for group {group_size}: {weight.shape}")
    rows = np.asarray(activation_rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != int(weight.shape[-1]) or not rows.shape[0]:
        raise ValueError(
            f"activation rows {rows.shape} do not match weight input width {weight.shape[-1]}"
        )
    if not np.isfinite(rows).all():
        raise ValueError("calibration activations contain non-finite values")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Torch CUDA optimizer requested on unavailable device {target}")

    native_weight, native_scales_array, native_biases_array = mx.quantize(
        weight,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    mx.eval(native_weight, native_scales_array, native_biases_array)
    auxiliary_dtype = native_scales_array.dtype
    dense_host = np.asarray(weight.astype(mx.float32))
    output_rows, input_width = dense_host.shape
    group_count = input_width // group_size
    native_codes_host = (
        np.asarray(native_weight)
        .view(np.uint8)
        .reshape(output_rows, input_width)
        .astype(np.int64)
        .reshape(output_rows, group_count, group_size)
    )
    native_scales_host = np.asarray(native_scales_array.astype(mx.float32))
    native_biases_host = np.asarray(native_biases_array.astype(mx.float32))

    source_dtype = torch.bfloat16 if weight.dtype == mx.bfloat16 else torch.float32
    dense_source = torch.from_numpy(dense_host).to(device=target, dtype=source_dtype)
    grouped = dense_source.float().reshape(output_rows, group_count, group_size)
    rows_device = torch.from_numpy(rows).to(device=target, dtype=torch.float32)
    hessian64 = _hessian_diagonal(rows_device)
    floor = max(float(hessian64.mean().item()) * 1e-6, 1e-12)
    hessian = hessian64.clamp_min(floor).float().reshape(group_count, group_size)

    baseline_codes = torch.from_numpy(native_codes_host).to(device=target)
    baseline_scales = torch.from_numpy(native_scales_host).to(device=target)
    baseline_biases = torch.from_numpy(native_biases_host).to(device=target)
    baseline_reconstructed = (
        baseline_codes.float() * baseline_scales.unsqueeze(-1)
        + baseline_biases.unsqueeze(-1)
    )
    baseline_errors = _weighted_error(grouped, baseline_reconstructed, hessian)

    codes = baseline_codes.clone()
    scales = baseline_scales.clone()
    biases = baseline_biases.clone()
    h = hessian.unsqueeze(0)
    sum_h = torch.zeros(h.shape[:-1], dtype=torch.float64, device=target)
    for index in range(group_size):
        sum_h = sum_h + h[..., index].double()
    for _ in range(iterations):
        q = codes.float()
        sum_hq = torch.zeros(q.shape[:-1], dtype=torch.float64, device=target)
        sum_hqq = torch.zeros(q.shape[:-1], dtype=torch.float64, device=target)
        sum_hw = torch.zeros(q.shape[:-1], dtype=torch.float64, device=target)
        sum_hqw = torch.zeros(q.shape[:-1], dtype=torch.float64, device=target)
        for index in range(group_size):
            hi = h[..., index].double()
            qi = q[..., index].double()
            wi = grouped[..., index].double()
            sum_hq = sum_hq + hi * qi
            sum_hqq = sum_hqq + hi * qi * qi
            sum_hw = sum_hw + hi * wi
            sum_hqw = sum_hqw + hi * qi * wi
        determinant = sum_hqq * sum_h - sum_hq * sum_hq
        valid = determinant > torch.finfo(torch.float32).eps
        fitted_scales = torch.where(
            valid,
            (sum_hqw * sum_h - sum_hq * sum_hw) / determinant,
            scales,
        )
        fitted_biases = torch.where(
            valid,
            (sum_hqq * sum_hw - sum_hq * sum_hqw) / determinant,
            biases,
        )
        fitted_scales = torch.where(
            fitted_scales.abs() > torch.finfo(torch.float32).tiny,
            fitted_scales,
            scales,
        )
        scales = _torch_round(fitted_scales, auxiliary_dtype)
        biases = _torch_round(fitted_biases, auxiliary_dtype)
        codes = torch.round(
            (grouped - biases.unsqueeze(-1)) / scales.unsqueeze(-1)
        ).clamp_(0, (1 << bits) - 1).to(torch.int64)

    candidate_reconstructed = codes.float() * scales.unsqueeze(-1) + biases.unsqueeze(-1)
    candidate_errors = _weighted_error(grouped, candidate_reconstructed, hessian)
    improved = candidate_errors < baseline_errors
    codes = torch.where(improved.unsqueeze(-1), codes, baseline_codes)
    scales = torch.where(improved, scales, baseline_scales)
    biases = torch.where(improved, biases, baseline_biases)
    final_reconstructed = codes.float() * scales.unsqueeze(-1) + biases.unsqueeze(-1)
    final_errors = _weighted_error(grouped, final_reconstructed, hessian)

    per_word = 32 // bits
    shifts = torch.arange(
        0,
        32,
        bits,
        device=target,
        dtype=torch.int64,
    )
    packed = (
        (codes.reshape(output_rows, input_width // per_word, per_word) << shifts)
        .sum(dim=-1)
        .cpu()
        .numpy()
        .astype(np.uint32, copy=False)
    )
    scales_host = scales.cpu().numpy()
    biases_host = biases.cpu().numpy()
    baseline_error = float(baseline_errors.sum().item())
    candidate_error = float(final_errors.sum().item())
    improved_fraction = float(improved.float().mean().item())

    del (
        dense_source,
        grouped,
        rows_device,
        hessian64,
        hessian,
        baseline_codes,
        baseline_scales,
        baseline_biases,
        baseline_reconstructed,
        baseline_errors,
        codes,
        scales,
        biases,
        candidate_reconstructed,
        candidate_errors,
        improved,
        final_reconstructed,
        final_errors,
    )
    torch.cuda.empty_cache()
    return ActivationAwareResult(
        weight=mx.array(packed, dtype=mx.uint32),
        scales=mx.array(scales_host).astype(auxiliary_dtype),
        biases=mx.array(biases_host).astype(auxiliary_dtype),
        baseline_error=baseline_error,
        candidate_error=candidate_error,
        improved_group_fraction=improved_fraction,
    )


class TorchCUDAAffineOptimizer:
    """Callable quantizer that records bounded per-layer CUDA resource evidence."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"Torch CUDA optimizer requested on unavailable device {self.device}")
        torch.cuda.set_device(self.device)
        self.calls = 0
        self.seconds = 0.0
        self.peak_memory_bytes = 0

    def __call__(self, weight: mx.array, rows: np.ndarray, **kwargs: Any) -> ActivationAwareResult:
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        result = torch_cuda_activation_aware_affine_quantize(
            weight,
            rows,
            device=self.device,
            **kwargs,
        )
        torch.cuda.synchronize(self.device)
        self.calls += 1
        self.seconds += time.perf_counter() - started
        self.peak_memory_bytes = max(
            self.peak_memory_bytes,
            int(torch.cuda.max_memory_allocated(self.device)),
        )
        return result

    def report(self) -> dict[str, Any]:
        return {
            "framework": "Torch/CUDA",
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "calls": self.calls,
            "seconds": self.seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "layer_residency": (
                "one dense source layer and its FP32 calibration rows are uploaded at a time; "
                "CUDA tensors are released after MLX uint32/scales/biases are materialized"
            ),
        }


__all__ = [
    "TorchCUDAAffineOptimizer",
    "torch_cuda_activation_aware_affine_quantize",
]
