#!/usr/bin/env python3
"""Evaluate calibrated INT8 against plain group-32 and community group-64."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.activation_quant import ActivationDataset  # noqa: E402
from minimax_h3_mlx.load import load_dit  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_at_path(module: Any, path: str) -> Any:
    value = module
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, (list, tuple)) else getattr(value, part)
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def output(layer: nn.Module, rows: np.ndarray) -> np.ndarray:
    scales = getattr(layer, "scales", None)
    dtype = scales.dtype if scales is not None else layer.weight.dtype
    value = layer(mx.array(rows).astype(dtype))
    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def paired_bootstrap(
    candidate: list[float],
    reference: list[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    delta = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(iterations, len(delta)))
    draws = delta[indices].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True, help="plain affine INT8 group-32")
    parser.add_argument("--community", required=True, help="community affine INT8 group-64")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="evaluate only this exact module path (repeatable); default evaluates all 258",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")

    source_path = Path(args.source)
    candidate_path = Path(args.candidate)
    baseline_path = Path(args.baseline)
    community_path = Path(args.community)
    calibration_path = Path(args.calibration)
    dataset = ActivationDataset.load(calibration_path)
    if dataset.manifest.get("source", {}).get("revision") != args.source_revision:
        parser.error("activation source revision does not match --source-revision")

    config_hashes = {
        name: sha256(path / "config.json")
        for name, path in (
            ("source", source_path),
            ("candidate", candidate_path),
            ("baseline", baseline_path),
            ("community", community_path),
        )
    }
    if len(set(config_hashes.values())) != 1:
        parser.error(f"transformer config mismatch: {config_hashes}")
    candidate_recipe = read_json(candidate_path / "quant_config.json")
    baseline_recipe = read_json(baseline_path / "quant_config.json")
    community_recipe = read_json(community_path / "quant_config.json")
    if (
        candidate_recipe.get("bits"),
        candidate_recipe.get("group_size"),
        candidate_recipe.get("mode"),
    ) != (8, 32, "affine"):
        parser.error("candidate is not MLX affine INT8 group-32")
    if (baseline_recipe.get("bits"), baseline_recipe.get("group_size")) != (8, 32):
        parser.error("baseline is not plain INT8 group-32")
    if (community_recipe.get("bits"), community_recipe.get("group_size")) != (8, 64):
        parser.error("community artifact is not INT8 group-64")
    if "activation-aware" not in str(candidate_recipe.get("algorithm", {})):
        parser.error("candidate does not record the activation-aware algorithm")

    models = {}
    load_seconds = {}
    for name, path in (
        ("source", source_path),
        ("candidate", candidate_path),
        ("baseline", baseline_path),
        ("community", community_path),
    ):
        started = time.perf_counter()
        models[name] = load_dit(path, strict=True, verbose=True)
        load_seconds[name] = time.perf_counter() - started
        print(f"strict load {name}: {load_seconds[name]:.2f}s", flush=True)

    all_paths = sorted(dataset.paths("holdout"))
    paths = list(dict.fromkeys(args.path)) if args.path else all_paths
    unknown_paths = sorted(set(paths) - set(all_paths))
    if unknown_paths:
        parser.error(f"requested holdout paths are absent: {unknown_paths}")
    bf16_layers = set(candidate_recipe.get("bf16_layers", []))
    observations = []
    totals = {
        name: {"error_sq": 0.0, "reference_sq": 0.0, "quantized_error_sq": 0.0, "quantized_reference_sq": 0.0}
        for name in ("candidate", "baseline", "community")
    }
    normalized = {name: [] for name in ("candidate", "baseline", "community")}
    quantized_types = 0
    for index, path in enumerate(paths, start=1):
        rows = dataset.get("holdout", path)
        source_layer = module_at_path(models["source"], path)
        reference = output(source_layer, rows)
        reference_sq = float(np.sum(np.square(reference, dtype=np.float64)))
        record = {
            "path": path,
            "rows": int(rows.shape[0]),
            "output_elements": int(reference.size),
            "bf16_exception": path in bf16_layers,
            "variants": {},
        }
        for name in ("candidate", "baseline", "community"):
            layer = module_at_path(models[name], path)
            actual = output(layer, rows)
            error_sq = float(np.sum(np.square(actual - reference, dtype=np.float64)))
            finite = bool(np.isfinite(actual).all())
            relative_l2 = float(np.sqrt(error_sq / max(reference_sq, 1e-24)))
            if not finite:
                raise RuntimeError(f"{name} produced non-finite output at {path}")
            if name == "candidate" and isinstance(layer, nn.QuantizedLinear):
                quantized_types += 1
                if layer.weight.dtype != mx.uint32:
                    raise TypeError(f"candidate {path} packed dtype is {layer.weight.dtype}")
            totals[name]["error_sq"] += error_sq
            totals[name]["reference_sq"] += reference_sq
            if path not in bf16_layers:
                totals[name]["quantized_error_sq"] += error_sq
                totals[name]["quantized_reference_sq"] += reference_sq
            normalized[name].append(relative_l2)
            record["variants"][name] = {
                "relative_l2": relative_l2,
                "finite": finite,
                "layer_type": type(layer).__name__,
            }
        observations.append(record)
        print(
            f"holdout {index}/{len(paths)} {path}: "
            f"candidate={record['variants']['candidate']['relative_l2']:.8f} "
            f"baseline={record['variants']['baseline']['relative_l2']:.8f} "
            f"community={record['variants']['community']['relative_l2']:.8f}",
            flush=True,
        )

    aggregate = {}
    for name, values in totals.items():
        aggregate[name] = {
            "all_layers_relative_l2": float(
                np.sqrt(values["error_sq"] / max(values["reference_sq"], 1e-24))
            ),
            "quantized_only_relative_l2": float(
                np.sqrt(
                    values["quantized_error_sq"]
                    / max(values["quantized_reference_sq"], 1e-24)
                )
            ),
            "mean_layer_relative_l2": float(np.mean(normalized[name])),
        }
    gates = {
        "finite": all(
            variant["finite"]
            for record in observations
            for variant in record["variants"].values()
        ),
        "candidate_better_than_plain_baseline_primary": (
            aggregate["candidate"]["all_layers_relative_l2"]
            < aggregate["baseline"]["all_layers_relative_l2"]
        ),
        "candidate_better_than_community_primary": (
            aggregate["candidate"]["all_layers_relative_l2"]
            < aggregate["community"]["all_layers_relative_l2"]
        ),
        "candidate_quantized_layers_better_than_plain_baseline": (
            aggregate["candidate"]["quantized_only_relative_l2"]
            < aggregate["baseline"]["quantized_only_relative_l2"]
        ),
        "candidate_quantized_layer_types_match_recipe": (
            quantized_types
            == (
                int(candidate_recipe["quantized_layers"]["8"])
                if not args.path
                else sum(path not in bf16_layers for path in paths)
            )
        ),
    }
    report = {
        "schema_version": 1,
        "primary_metric": (
            "aggregate relative-L2 over all held-out real layer outputs; "
            "sum squared errors before taking one square root"
        ),
        "source_revision": args.source_revision,
        "config_sha256": config_hashes["source"],
        "evaluation_scope": {
            "mode": "selected_paths" if args.path else "full",
            "evaluated_layer_count": len(paths),
            "available_layer_count": len(all_paths),
            "paths": paths,
        },
        "calibration": {
            "path": str(calibration_path.resolve()),
            "sha256": sha256(calibration_path),
            "holdout_contract": dataset.manifest.get("split_contract"),
            "holdout_cases": [
                case
                for case in dataset.manifest.get("cases", [])
                if case.get("split") == "holdout"
            ],
        },
        "artifacts": {
            "candidate": str(candidate_path.resolve()),
            "plain_group32_baseline": str(baseline_path.resolve()),
            "community_group64": str(community_path.resolve()),
        },
        "strict_load_seconds": load_seconds,
        "candidate_bf16_exceptions": candidate_recipe.get("bf16_exceptions", []),
        "aggregate": aggregate,
        "paired_layer_bootstrap": {
            "candidate_minus_plain_baseline": paired_bootstrap(
                normalized["candidate"],
                normalized["baseline"],
                iterations=args.bootstrap,
                seed=20260811,
            ),
            "candidate_minus_community": paired_bootstrap(
                normalized["candidate"],
                normalized["community"],
                iterations=args.bootstrap,
                seed=20260812,
            ),
        },
        "gates": gates,
        "observations": observations,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        print(json.dumps(aggregate, indent=2))
        raise RuntimeError(f"calibrated INT8 holdout gates failed: {failed}")
    print(
        "PASS: calibrated candidate is finite and strictly better than plain group-32 "
        "and community group-64 on the held-out primary metric"
    )
    print(f"report: {output_path.resolve()} sha256={sha256(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
