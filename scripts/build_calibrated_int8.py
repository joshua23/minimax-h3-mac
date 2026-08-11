#!/usr/bin/env python3
"""Build the calibration-aware MLX-native MiniMax-H3 INT8 DiT."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_quant import save_sharded  # noqa: E402
from minimax_h3_mlx.activation_quant import (  # noqa: E402
    ActivationDataset,
    diagonal_hessian_relative_error,
    quantize_activation_aware,
    quantized_parameter_paths,
)
from minimax_h3_mlx.load import load_dit  # noqa: E402
from minimax_h3_mlx.quantize import (  # noqa: E402
    QuantConfig,
    _class_predicate,
    resident_footprint,
)


IMPLEMENTATION_PATHS = (
    Path("scripts/calibrate_int8.py"),
    Path("scripts/calibrate_int8_torch.py"),
    Path("scripts/probe_real_cuda_parity.py"),
    Path("scripts/activation_quant_torch.py"),
    Path("scripts/build_calibrated_int8.py"),
    Path("scripts/eval_calibrated_int8.py"),
    Path("scripts/build_quant.py"),
    Path("minimax_h3_mlx/activation_quant.py"),
    Path("minimax_h3_mlx/quantize.py"),
    Path("minimax_h3_mlx/load.py"),
    Path("minimax_h3_mlx/streaming.py"),
    Path("minimax_h3_mlx/dit.py"),
)


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
    return {"repository_commit": commit, "repository_dirty": dirty}


def module_at_path(module: Any, path: str) -> Any:
    value = module
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, (list, tuple)) else getattr(value, part)
    return value


def sensitivity_scan(
    model,
    predicate,
    calibration: ActivationDataset,
    quantizer,
) -> list[dict[str, Any]]:
    records = []
    selected = quantized_parameter_paths(model, predicate)
    for index, (path, layer) in enumerate(selected, start=1):
        score = diagonal_hessian_relative_error(
            layer.weight,
            calibration.get("calibration", path),
            bits=8,
            group_size=32,
            quantizer=quantizer,
        )
        records.append(
            {
                "path": path,
                "parameters": int(layer.weight.size),
                "rtn_diagonal_hessian_relative_error": score,
            }
        )
        print(
            f"sensitivity {index}/{len(selected)}: {path} "
            f"rel={score:.8f} params={int(layer.weight.size):,}",
            flush=True,
        )
    return records


def select_bf16_layers(
    records: list[dict[str, Any]],
    *,
    total_parameters: int,
    max_layers: int,
    max_parameter_percent: float,
) -> list[dict[str, Any]]:
    budget = int(total_parameters * max_parameter_percent / 100.0)
    selected = []
    used = 0
    for record in sorted(
        records,
        key=lambda item: (-float(item["rtn_diagonal_hessian_relative_error"]), str(item["path"])),
    ):
        parameters = int(record["parameters"])
        if len(selected) >= max_layers:
            break
        if used + parameters > budget:
            continue
        selected.append(
            {
                **record,
                "full_dit_parameter_fraction": parameters / total_parameters,
                "selection_reason": (
                    "highest calibration diagonal-Hessian RTN sensitivity within the "
                    "preregistered BF16 parameter budget"
                ),
            }
        )
        used += parameters
    if max_layers > 0 and not selected:
        raise RuntimeError(
            "the BF16 sensitivity budget selected no layer; increase --max-bf16-parameter-percent"
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="pinned FL2VA source directory")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--optimizer",
        choices=("cuda", "cpu"),
        default="cuda",
        help="CUDA is the delivery path; CPU is retained only as a parity reference",
    )
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--max-bf16-layers", type=int, default=4)
    parser.add_argument("--max-bf16-parameter-percent", type=float, default=1.5)
    args = parser.parse_args()
    if args.max_bf16_layers < 0:
        parser.error("--max-bf16-layers must be non-negative")
    if not 0 <= args.max_bf16_parameter_percent <= 5:
        parser.error("--max-bf16-parameter-percent must be between 0 and 5")

    source = Path(args.checkpoint)
    transformer = source / "transformer"
    output = Path(args.out)
    report_path = Path(args.report)
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite non-empty output: {output}")
    calibration_path = Path(args.calibration)
    calibration = ActivationDataset.load(calibration_path)
    recorded_source = calibration.manifest.get("source", {})
    if recorded_source.get("revision") != args.source_revision:
        parser.error(
            "calibration source revision mismatch: "
            f"{recorded_source.get('revision')!r} != {args.source_revision!r}"
        )
    config_hash = sha256(transformer / "config.json")
    if recorded_source.get("transformer_config_sha256") != config_hash:
        parser.error("calibration transformer config hash does not match the source checkpoint")

    print(f"loading strict bfloat16 source: {transformer}", flush=True)
    started = time.perf_counter()
    model = load_dit(transformer, strict=True, verbose=True)
    print(f"loaded source in {time.perf_counter() - started:.2f}s", flush=True)
    total_parameters = sum(int(value.size) for _, value in tree_flatten(model.parameters()))
    config = QuantConfig(
        bits=8,
        group_size=32,
        mode="affine",
        quantize_adaln=True,
        adaln_bits=8,
    )
    predicate = _class_predicate(config)
    cuda_optimizer = None
    if args.optimizer == "cuda":
        from activation_quant_torch import TorchCUDAAffineOptimizer

        cuda_optimizer = TorchCUDAAffineOptimizer(args.cuda_device)
        quantizer = cuda_optimizer
    else:
        from minimax_h3_mlx.activation_quant import activation_aware_affine_quantize

        quantizer = activation_aware_affine_quantize
    expected_paths = {path for path, _ in quantized_parameter_paths(model, predicate)}
    missing_calibration = sorted(expected_paths - calibration.paths("calibration"))
    missing_holdout = sorted(expected_paths - calibration.paths("holdout"))
    if missing_calibration or missing_holdout:
        parser.error(
            f"activation dataset is incomplete: {len(missing_calibration)} calibration and "
            f"{len(missing_holdout)} holdout layers missing"
        )

    scan = sensitivity_scan(model, predicate, calibration, quantizer)
    bf16_records = select_bf16_layers(
        scan,
        total_parameters=total_parameters,
        max_layers=args.max_bf16_layers,
        max_parameter_percent=args.max_bf16_parameter_percent,
    )
    bf16_layers = {str(record["path"]) for record in bf16_records}
    print(f"BF16 sensitivity exceptions ({len(bf16_layers)}):", flush=True)
    for record in bf16_records:
        print(
            f"  {record['path']}: rel={record['rtn_diagonal_hessian_relative_error']:.8f}, "
            f"params={record['parameters']:,}",
            flush=True,
        )

    started = time.perf_counter()
    layer_metrics = quantize_activation_aware(
        model,
        predicate,
        calibration,
        bf16_layers,
        quantizer=quantizer,
    )
    build_seconds = time.perf_counter() - started
    if set(layer_metrics) | bf16_layers != expected_paths:
        raise RuntimeError("calibrated quantization did not account for every selected linear")
    baseline_error = sum(
        float(record["calibration_rtn_diagonal_hessian_sse"])
        for record in layer_metrics.values()
    )
    candidate_error = sum(
        float(record["calibration_candidate_diagonal_hessian_sse"])
        for record in layer_metrics.values()
    )
    if not candidate_error < baseline_error:
        raise RuntimeError(
            f"activation-aware objective did not improve RTN: {candidate_error} >= {baseline_error}"
        )

    footprint = resident_footprint(model)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy(transformer / "config.json", output / "config.json")
    state = repository_state()
    implementation_hashes = {
        str(path): sha256(ROOT / path)
        for path in IMPLEMENTATION_PATHS
        if (ROOT / path).is_file()
    }
    quant_meta = {
        "schema_version": 3,
        "profile": "m4-pro-calibrated-int8",
        "algorithm": {
            "family": "activation-aware diagonal-Hessian affine weight-only INT8",
            "objective": "minimize the diagonal-Hessian approximation to ||WX-Q(W)X||_F^2",
            "optimizer": (
                "three alternating weighted least-squares affine refits on Torch/CUDA"
                if cuda_optimizer is not None
                else "three alternating weighted least-squares affine refits on NumPy/MLX CPU"
            ),
            "fallback": "retain the plain MLX RTN group whenever a fitted group is not better",
            "packing": "four consecutive uint8 codes per uint32, first code in least-significant bits",
        },
        "bits": 8,
        "group_size": 32,
        "mode": "affine",
        "quantize_adaln": True,
        "adaln_bits": 8,
        "quantized_layers": {"8": len(layer_metrics)},
        "bf16_layers": sorted(bf16_layers),
        "bf16_exceptions": bf16_records,
        "bf16_exception_parameters": sum(int(record["parameters"]) for record in bf16_records),
        "bf16_exception_full_dit_parameter_fraction": (
            sum(int(record["parameters"]) for record in bf16_records) / total_parameters
        ),
        "protected_source_precision_modules": [
            "video_patch_proj",
            "audio_patch_proj",
            "condition_proj",
            "time_embedder",
            "final_layer",
            "norms",
            "linear_biases",
        ],
        "calibration": {
            "path": str(calibration_path.resolve()),
            "sha256": sha256(calibration_path),
            "manifest": calibration.manifest,
        },
        "calibration_objective": {
            "plain_rtn_diagonal_hessian_sse": baseline_error,
            "candidate_diagonal_hessian_sse": candidate_error,
            "relative_improvement": (baseline_error - candidate_error) / baseline_error,
        },
        "optimizer": (
            cuda_optimizer.report()
            if cuda_optimizer is not None
            else {
                "framework": "NumPy/MLX CPU",
                "purpose": "reference implementation only",
            }
        ),
        "source": {
            "model": "MiniMaxAI/MiniMax-H3",
            "revision": args.source_revision,
            "subfolder": f"{source.name}/transformer",
            "config_sha256": config_hash,
        },
        "builder": {
            "repository": "https://github.com/Argus-AiTeam/minimax-h3-mac",
            **state,
            "mlx_version": importlib.metadata.version("mlx-cpu")
            if platform.system() == "Linux"
            else importlib.metadata.version("mlx"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation_sha256": implementation_hashes,
        },
        "gb_on_disk": round(footprint["total_gb"], 3),
        "gb_resident_after_adaln_drop": round(footprint["resident_gb"], 3),
    }
    with (output / "quant_config.json").open("w") as handle:
        json.dump(quant_meta, handle, indent=2)
    names = save_sharded(model, output, {"quantization": json.dumps(quant_meta)})
    del model
    mx.clear_cache()

    print("strictly reloading final artifact", flush=True)
    reloaded = load_dit(output, strict=True, verbose=True)
    quantized_path = sorted(layer_metrics)[0]
    quantized_layer = module_at_path(reloaded, quantized_path)
    if not isinstance(quantized_layer, nn.QuantizedLinear):
        raise TypeError(f"{quantized_path} reloaded as {type(quantized_layer).__name__}")
    if quantized_layer.weight.dtype != mx.uint32:
        raise TypeError(f"{quantized_path} packed weight dtype is {quantized_layer.weight.dtype}")
    projection = quantized_layer(
        mx.array(calibration.get("holdout", quantized_path)[:1]).astype(mx.bfloat16)
    )
    mx.eval(projection)
    finite = bool(mx.all(mx.isfinite(projection)).item())
    if not finite:
        raise RuntimeError("strictly reloaded real quantized projection produced non-finite output")
    for path in bf16_layers:
        if isinstance(module_at_path(reloaded, path), nn.QuantizedLinear):
            raise TypeError(f"BF16 exception {path} reloaded as QuantizedLinear")

    report = {
        "artifact": str(output.resolve()),
        "shards": names,
        "quantized_layer_count": len(layer_metrics),
        "bf16_exceptions": bf16_records,
        "total_parameters": total_parameters,
        "build_seconds": build_seconds,
        "optimizer": quant_meta["optimizer"],
        "calibration_objective": quant_meta["calibration_objective"],
        "strict_load": {
            "passed": True,
            "strict": True,
            "quantized_layer": quantized_path,
            "quantized_layer_type": type(quantized_layer).__name__,
            "packed_weight_dtype": str(quantized_layer.weight.dtype),
            "real_projection_finite": finite,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PASS: {len(layer_metrics)} calibrated INT8 layers, {len(bf16_layers)} BF16 exceptions, "
        f"{len(names)} shards, strict MLX reload and real projection finite"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
