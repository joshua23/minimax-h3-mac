"""Activation-aware quantization, MLX packing, and mixed BF16 reload contract."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_quant import save_sharded
from calibrate_int8 import capture_disjoint_adaln_inputs, validate_cases
from minimax_h3_mlx.activation_quant import (
    ActivationDataset,
    ActivationRecorder,
    _pack_uint32,
    activation_aware_affine_quantize,
    bind_activation_paths,
    quantize_activation_aware,
    quantized_parameter_paths,
)
from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.quantize import QuantConfig, _class_predicate
def tiny_config() -> DiTConfig:
    return DiTConfig(
        hidden_size=256,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=64,
        ffn_hidden_size=128,
        latents_dim=4,
        audio_latents_dim=8,
        text_dim=128,
        timestep_input_dim=16,
        time_embed_hidden_size=256,
        time_embed_dim=64,
        adaln_out_features=6 * 3 * 256,
        final_adaln_out_features=2 * 256,
        rope_inv_freq_len=4,
    )


def write_config(path: Path, cfg: DiTConfig) -> None:
    path.write_text(
        json.dumps(
            {
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "token_refiner_num_layers": cfg.token_refiner_num_layers,
                "num_attention_heads": cfg.num_attention_heads,
                "attention_head_dim": cfg.attention_head_dim,
                "ffn_hidden_size": cfg.ffn_hidden_size,
                "latents_dim": cfg.latents_dim,
                "audio_latents_dim": cfg.audio_latents_dim,
                "patch_size": list(cfg.patch_size),
                "text_dim": cfg.text_dim,
                "timestep_input_dim": cfg.timestep_input_dim,
                "time_embed_hidden_size": cfg.time_embed_hidden_size,
                "time_embed_dim": cfg.time_embed_dim,
                "adaln_out_features": cfg.adaln_out_features,
                "final_adaln_out_features": cfg.final_adaln_out_features,
                "rope_inv_freq_len": cfg.rope_inv_freq_len,
            }
        )
    )


def test_weighted_affine_fit_and_packing() -> None:
    known_codes = np.arange(32, dtype=np.uint16)[None]
    known_packed = _pack_uint32(known_codes, 8)
    assert int(known_packed[0, 0]) == 0x03020100
    assert int(known_packed[0, 1]) == 0x07060504

    rng = np.random.default_rng(7)
    dense = rng.normal(0, 0.15, size=(4, 32)).astype(np.float32)
    dense[:, 0] = np.array([-8.0, 7.0, -6.0, 9.0], dtype=np.float32)
    rows = rng.normal(size=(64, 32)).astype(np.float32)
    rows[:, 0] *= 1e-4
    result = activation_aware_affine_quantize(mx.array(dense), rows)
    native_rtn = mx.quantize(
        mx.array(dense),
        group_size=32,
        bits=8,
        mode="affine",
    )
    fallback = activation_aware_affine_quantize(mx.array(dense), rows, iterations=0)
    mx.eval(*native_rtn, fallback.weight, fallback.scales, fallback.biases)
    assert bool(mx.array_equal(fallback.weight, native_rtn[0]).item())
    assert bool(mx.array_equal(fallback.scales, native_rtn[1]).item())
    assert bool(mx.array_equal(fallback.biases, native_rtn[2]).item())
    assert result.candidate_error < result.baseline_error
    assert result.improved_group_fraction > 0
    reconstructed = mx.dequantize(
        result.weight,
        result.scales,
        result.biases,
        group_size=32,
        bits=8,
        mode="affine",
        dtype=mx.float32,
    )
    native = mx.quantized_matmul(
        mx.array(rows[:2]),
        result.weight,
        scales=result.scales,
        biases=result.biases,
        transpose=True,
        group_size=32,
        bits=8,
        mode="affine",
    )
    dense_output = mx.array(rows[:2]) @ reconstructed.T
    mx.eval(native, dense_output)
    codes = np.asarray(result.weight).view(np.uint8).reshape(4, 32)
    manual = (
        codes.astype(np.float32) * np.asarray(result.scales.astype(mx.float32))
        + np.asarray(result.biases.astype(mx.float32))
    )
    assert result.weight.dtype == mx.uint32
    assert result.weight.shape == (4, 8)
    assert bool(mx.all(mx.isfinite(native)).item())
    assert np.allclose(np.asarray(reconstructed), manual, rtol=0, atol=0.04)


def test_torch_cuda_optimizer_parity() -> None:
    try:
        import torch
        from activation_quant_torch import torch_cuda_activation_aware_affine_quantize
    except ModuleNotFoundError:
        print("SKIP: Torch CUDA optimizer parity (Torch unavailable)")
        return

    if not torch.cuda.is_available():
        print("SKIP: Torch CUDA optimizer parity (CUDA unavailable)")
        return
    rng = np.random.default_rng(29)
    dense = rng.normal(0, 0.2, size=(96, 128)).astype(np.float32)
    rows = rng.normal(size=(64, 128)).astype(np.float32)
    weight = mx.array(dense).astype(mx.bfloat16)
    cpu = activation_aware_affine_quantize(weight, rows)
    cuda = torch_cuda_activation_aware_affine_quantize(weight, rows)
    mx.eval(
        cpu.weight,
        cpu.scales,
        cpu.biases,
        cuda.weight,
        cuda.scales,
        cuda.biases,
    )
    assert np.array_equal(np.asarray(cuda.weight), np.asarray(cpu.weight))
    assert np.array_equal(
        np.asarray(cuda.scales.astype(mx.float32)),
        np.asarray(cpu.scales.astype(mx.float32)),
    )
    assert np.array_equal(
        np.asarray(cuda.biases.astype(mx.float32)),
        np.asarray(cpu.biases.astype(mx.float32)),
    )
    assert np.isclose(cuda.baseline_error, cpu.baseline_error, rtol=2e-6, atol=1e-6)
    assert np.isclose(cuda.candidate_error, cpu.candidate_error, rtol=2e-6, atol=1e-6)
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
    assert np.array_equal(np.asarray(cuda_dense), np.asarray(cpu_dense))


def test_mixed_bf16_strict_reload() -> None:
    cfg = tiny_config()
    mx.random.seed(0)
    model = MiniMaxH3DiT(cfg)
    mx.eval(model.parameters())
    recipe = QuantConfig(
        bits=8,
        group_size=32,
        mode="affine",
        quantize_adaln=True,
        adaln_bits=8,
    )
    predicate = _class_predicate(recipe)
    selected = quantized_parameter_paths(model, predicate)
    rng = np.random.default_rng(11)
    rows = {
        path: rng.normal(size=(8, int(layer.weight.shape[-1]))).astype(np.float32)
        for path, layer in selected
    }
    dataset = ActivationDataset(
        {"source": {"revision": "test"}},
        {"calibration": rows, "holdout": {path: value.copy() for path, value in rows.items()}},
    )
    bf16_path = selected[0][0]
    metrics = quantize_activation_aware(model, predicate, dataset, {bf16_path})
    assert bf16_path not in metrics

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        write_config(output / "config.json", cfg)
        quant_meta = {
            "bits": 8,
            "group_size": 32,
            "mode": "affine",
            "quantize_adaln": True,
            "adaln_bits": 8,
            "bf16_layers": [bf16_path],
            "quantized_layers": {"8": len(metrics)},
        }
        (output / "quant_config.json").write_text(json.dumps(quant_meta))
        save_sharded(model, output, {"quantization": json.dumps(quant_meta)})
        reloaded = load_dit(output, strict=True)

    value = reloaded
    for part in bf16_path.split("."):
        value = value[int(part)] if isinstance(value, (list, tuple)) else getattr(value, part)
    assert isinstance(value, nn.Linear)
    assert not isinstance(value, nn.QuantizedLinear)
    quantized_path = sorted(metrics)[0]
    value = reloaded
    for part in quantized_path.split("."):
        value = value[int(part)] if isinstance(value, (list, tuple)) else getattr(value, part)
    assert isinstance(value, nn.QuantizedLinear)
    assert value.weight.dtype == mx.uint32


def test_dataset_roundtrip() -> None:
    dataset = ActivationDataset(
        {"split_contract": "disjoint"},
        {
            "calibration": {"blocks.0.mlp.fc1": np.ones((2, 32), dtype=np.float32)},
            "holdout": {"blocks.0.mlp.fc1": np.zeros((3, 32), dtype=np.float32)},
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "activations.npz"
        dataset.save(path)
        loaded = ActivationDataset.load(path)
    assert loaded.manifest == dataset.manifest
    assert loaded.get("calibration", "blocks.0.mlp.fc1").shape == (2, 32)
    assert loaded.get("holdout", "blocks.0.mlp.fc1").shape == (3, 32)


def test_disjoint_adaln_inputs() -> None:
    split_timesteps = validate_cases(6)
    assert not (
        set(split_timesteps["calibration"]) & set(split_timesteps["holdout"])
    )
    try:
        validate_cases(5)
    except ValueError as exc:
        assert "model evaluations" in str(exc)
    else:
        raise AssertionError("five scheduler points unexpectedly covered case index four")

    cfg = tiny_config()
    model = MiniMaxH3DiT(cfg)
    bind_activation_paths(model)
    recorder = ActivationRecorder(max_rows_per_call=8, max_rows_per_layer=8)
    capture_disjoint_adaln_inputs(model, recorder, split_timesteps)
    dataset = recorder.dataset({"split_contract": "disjoint AdaLN"})
    path = "blocks.0.adaln_proj.linear"
    calibration = dataset.get("calibration", path)
    holdout = dataset.get("holdout", path)
    assert calibration.shape[0] == len(split_timesteps["calibration"])
    assert holdout.shape[0] == len(split_timesteps["holdout"])
    assert not np.array_equal(calibration[: holdout.shape[0]], holdout)


def main() -> int:
    test_weighted_affine_fit_and_packing()
    test_torch_cuda_optimizer_parity()
    test_mixed_bf16_strict_reload()
    test_dataset_roundtrip()
    test_disjoint_adaln_inputs()
    print("activation-aware INT8 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
