#!/usr/bin/env python3
"""Compare INT8 DiT projection fidelity against the pinned bfloat16 weights.

This uses the real checkpoint tensors and MLX QuantizedLinear kernels, but keeps
the evaluation block-local so it can run without loading the complete 33B DiT.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_quant import audit  # noqa: E402
from minimax_h3_mlx.config import DiTConfig  # noqa: E402
from minimax_h3_mlx.dit import RotaryPosEmbed3D, TransformerBlock  # noqa: E402
from minimax_h3_mlx.selective_loading import (  # noqa: E402
    load_selected_mlx_tensors,
    load_weight_map,
)
from minimax_h3_mlx.streaming import QuantizedBlockProvider  # noqa: E402

LINEARS = (
    "attn.qkv_proj",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
    "adaln_proj.linear",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-12))


def cosine(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(np.linalg.norm(actual) * np.linalg.norm(reference), 1e-12)
    return float(actual.ravel() @ reference.ravel() / denominator)


def paired_delta_ci(
    candidate: list[float],
    community: list[float],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(community, dtype=np.float64)
    if left.shape != right.shape or not left.size:
        raise ValueError(f"paired observations differ or are empty: {left.shape} vs {right.shape}")
    delta = left - right
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(delta), size=(iterations, len(delta)))
    draws = delta[selected].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def module_at_path(module: Any, path: str) -> Any:
    value = module
    for part in path.split("."):
        value = getattr(value, part)
    return value


def input_array(rows: int, width: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((rows, width), dtype=np.float32)
    return mx.array(values).astype(mx.bfloat16)


def block_inputs(config: DiTConfig, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    hidden = mx.array(
        rng.standard_normal((1, rows, config.hidden_size), dtype=np.float32)
    ).astype(mx.bfloat16)
    temb = mx.array(
        rng.standard_normal((2, config.time_embed_dim), dtype=np.float32)
    )
    adaln_indices = mx.array(np.arange(rows, dtype=np.int32) % 6)
    positions = np.stack(
        [
            np.arange(rows, dtype=np.int32) % 3,
            np.arange(rows, dtype=np.int32) % 5,
            np.arange(rows, dtype=np.int32) % 7,
        ],
        axis=-1,
    )
    rotary = RotaryPosEmbed3D(config)(mx.array(positions))
    return hidden, temb, adaln_indices, rotary


def source_references(
    source: Path,
    config: DiTConfig,
    block_index: int,
    rows: int,
    cases: int,
    base_seed: int,
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, mx.array]]:
    weight_map = load_weight_map(source)
    block_prefix = f"blocks.{block_index}."
    keys = [key for key in weight_map if key.startswith(block_prefix)]
    tensors = load_selected_mlx_tensors(source, keys)
    block = TransformerBlock(config)
    block.update(
        tree_unflatten(
            [(key[len(block_prefix) :], tensor) for key, tensor in tensors.items()]
        )
    )
    mx.eval(block.parameters())
    references: dict[tuple[int, str], np.ndarray] = {}
    source_weights: dict[str, mx.array] = {}
    for layer_index, layer in enumerate(LINEARS):
        module = module_at_path(block, layer)
        source_weights[layer] = module.weight
        for case_index in range(cases):
            seed = base_seed + block_index * 1000 + layer_index * 100 + case_index
            inputs = input_array(rows, int(module.weight.shape[-1]), seed)
            output = module(inputs)
            mx.eval(output)
            references[(case_index, layer)] = np.asarray(output.astype(mx.float32))
    for case_index in range(cases):
        seed = base_seed + block_index * 1000 + 900 + case_index
        hidden, temb, adaln_indices, rotary = block_inputs(config, rows, seed)
        output = block(
            hidden,
            block.adaln_proj(temb),
            adaln_indices,
            rotary,
        )
        mx.eval(output)
        references[(case_index, "block.forward")] = np.asarray(output.astype(mx.float32))
    del block, tensors
    gc.collect()
    mx.clear_cache()
    return references, source_weights


def mlx_relative_l2(actual: mx.array, reference: mx.array) -> float:
    actual_f32 = actual.astype(mx.float32)
    reference_f32 = reference.astype(mx.float32)
    numerator = mx.sum(mx.square(actual_f32 - reference_f32))
    denominator = mx.maximum(mx.sum(mx.square(reference_f32)), mx.array(1e-24))
    value = mx.sqrt(numerator / denominator)
    mx.eval(value)
    return float(value.item())


def evaluate_artifact(
    artifact: Path,
    block_index: int,
    references: dict[tuple[int, str], np.ndarray],
    source_weights: dict[str, mx.array],
    rows: int,
    cases: int,
    base_seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], float]:
    started = time.perf_counter()
    provider = QuantizedBlockProvider(artifact)
    block = provider.load_block(block_index)
    weight_observations = []
    output_observations = []
    block_observations = []
    for layer_index, layer_name in enumerate(LINEARS):
        layer = module_at_path(block, layer_name)
        dequantized = mx.dequantize(
            layer.weight,
            layer.scales,
            layer.biases,
            group_size=provider.quantization.group_size,
            bits=provider.quantization.bits,
            mode=provider.quantization.mode,
        )
        weight_observations.append(
            {
                "block": block_index,
                "layer": layer_name,
                "relative_l2": mlx_relative_l2(dequantized, source_weights[layer_name]),
                "finite": bool(mx.all(mx.isfinite(dequantized)).item()),
            }
        )
        for case_index in range(cases):
            seed = base_seed + block_index * 1000 + layer_index * 100 + case_index
            inputs = input_array(rows, int(layer.scales.shape[-1]) * provider.quantization.group_size, seed)
            call_started = time.perf_counter()
            native_output = layer(inputs)
            mx.eval(native_output)
            elapsed = time.perf_counter() - call_started
            dense_output = mx.matmul(inputs, dequantized.T)
            if "bias" in layer:
                dense_output = dense_output + layer.bias
            mx.eval(dense_output)
            native_actual = np.asarray(native_output.astype(mx.float32))
            dense_actual = np.asarray(dense_output.astype(mx.float32))
            reference = references[(case_index, layer_name)]
            output_observations.append(
                {
                    "block": block_index,
                    "case": case_index,
                    "layer": layer_name,
                    "dequantized_relative_l2": relative_l2(dense_actual, reference),
                    "dequantized_cosine": cosine(dense_actual, reference),
                    "native_relative_l2": relative_l2(native_actual, reference),
                    "native_cosine": cosine(native_actual, reference),
                    "finite": bool(
                        np.isfinite(native_actual).all() and np.isfinite(dense_actual).all()
                    ),
                    "kernel_seconds": elapsed,
                }
            )
        del dequantized
    for case_index in range(cases):
        seed = base_seed + block_index * 1000 + 900 + case_index
        hidden, temb, adaln_indices, rotary = block_inputs(provider.config, rows, seed)
        call_started = time.perf_counter()
        output = block(
            hidden,
            block.adaln_proj(temb),
            adaln_indices,
            rotary,
        )
        mx.eval(output)
        elapsed = time.perf_counter() - call_started
        actual = np.asarray(output.astype(mx.float32))
        reference = references[(case_index, "block.forward")]
        block_observations.append(
            {
                "block": block_index,
                "case": case_index,
                "layer": "block.forward",
                "relative_l2": relative_l2(actual, reference),
                "cosine": cosine(actual, reference),
                "finite": bool(np.isfinite(actual).all()),
                "kernel_seconds": elapsed,
            }
        )
    elapsed = time.perf_counter() - started
    del block, provider
    gc.collect()
    mx.clear_cache()
    return {
        "weight_observations": weight_observations,
        "output_observations": output_observations,
        "block_observations": block_observations,
    }, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="pinned bfloat16 transformer directory")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--community", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 24, 49])
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.rows <= 0 or args.cases <= 0 or args.bootstrap <= 0:
        parser.error("--rows, --cases, and --bootstrap must be positive")

    source = Path(args.source)
    candidate = Path(args.candidate)
    community = Path(args.community)
    source_config = sha256(source / "config.json")
    candidate_config = sha256(candidate / "config.json")
    community_config = sha256(community / "config.json")
    if len({source_config, candidate_config, community_config}) != 1:
        parser.error(
            "transformer config mismatch: "
            f"source={source_config}, candidate={candidate_config}, community={community_config}"
        )
    config = DiTConfig.from_json(source / "config.json")
    invalid_blocks = [block for block in args.blocks if not 0 <= block < config.num_layers]
    if invalid_blocks:
        parser.error(f"blocks outside 0..{config.num_layers - 1}: {invalid_blocks}")

    candidate_audit = audit(candidate)
    community_audit = audit(community)
    results = {
        "candidate": {
            "weight_observations": [],
            "output_observations": [],
            "block_observations": [],
            "load_and_kernel_seconds": 0.0,
        },
        "community": {
            "weight_observations": [],
            "output_observations": [],
            "block_observations": [],
            "load_and_kernel_seconds": 0.0,
        },
    }
    for block_index in args.blocks:
        print(f"block {block_index}: loading bfloat16 reference projections", flush=True)
        references, source_weights = source_references(
            source,
            config,
            block_index,
            args.rows,
            args.cases,
            args.seed,
        )
        for name, artifact in (("candidate", candidate), ("community", community)):
            print(f"block {block_index}: evaluating {name}", flush=True)
            observations, elapsed = evaluate_artifact(
                artifact,
                block_index,
                references,
                source_weights,
                args.rows,
                args.cases,
                args.seed,
            )
            for key in ("weight_observations", "output_observations", "block_observations"):
                results[name][key].extend(observations[key])
            results[name]["load_and_kernel_seconds"] += elapsed
        del references, source_weights
        gc.collect()
        mx.clear_cache()

    paired_metrics = {
        "weight_relative_l2": (
            "weight_observations",
            "relative_l2",
        ),
        "dequantized_output_relative_l2": (
            "output_observations",
            "dequantized_relative_l2",
        ),
        "native_output_relative_l2": (
            "output_observations",
            "native_relative_l2",
        ),
        "native_block_relative_l2": (
            "block_observations",
            "relative_l2",
        ),
    }
    deltas = {}
    for metric, (collection, key) in paired_metrics.items():
        candidate_values = [row[key] for row in results["candidate"][collection]]
        community_values = [row[key] for row in results["community"][collection]]
        deltas[metric] = paired_delta_ci(
            candidate_values,
            community_values,
            args.bootstrap,
            args.seed,
        )
    for name in ("candidate", "community"):
        results[name]["mean_weight_relative_l2"] = float(
            np.mean([row["relative_l2"] for row in results[name]["weight_observations"]])
        )
        results[name]["mean_dequantized_output_relative_l2"] = float(
            np.mean(
                [
                    row["dequantized_relative_l2"]
                    for row in results[name]["output_observations"]
                ]
            )
        )
        results[name]["mean_native_output_relative_l2"] = float(
            np.mean([row["native_relative_l2"] for row in results[name]["output_observations"]])
        )
        results[name]["mean_native_block_relative_l2"] = float(
            np.mean([row["relative_l2"] for row in results[name]["block_observations"]])
        )
        results[name]["finite"] = all(
            row["finite"]
            for collection in (
                "weight_observations",
                "output_observations",
                "block_observations",
            )
            for row in results[name][collection]
        )

    decision = {
        "finite_outputs": results["candidate"]["finite"] and results["community"]["finite"],
        "candidate_strictly_lower_weight_relative_l2": (
            deltas["weight_relative_l2"]["mean"] < 0.0
        ),
        "weight_relative_l2_ci95_excludes_zero": (
            deltas["weight_relative_l2"]["ci95"][1] < 0.0
        ),
        "candidate_strictly_lower_dequantized_output_relative_l2": (
            deltas["dequantized_output_relative_l2"]["mean"] < 0.0
        ),
    }
    report = {
        "schema_version": 1,
        "metric": {
            "primary": "dequantized real-weight relative-L2 against pinned bfloat16 weights",
            "secondary": "isolated dense projection-output relative-L2 on identical BF16 inputs",
            "runtime_gate": "finite native QuantizedLinear and full TransformerBlock outputs",
        },
        "scope": {
            "blocks": args.blocks,
            "layers": [*LINEARS, "block.forward"],
            "rows_per_case": args.rows,
            "cases_per_layer": args.cases,
            "synthetic_activation_distribution": "deterministic standard normal cast to bfloat16",
            "claim_boundary": "weight/projection fidelity; not an end-to-end media-quality score",
        },
        "source": {
            "path": str(source.resolve()),
            "model": "MiniMaxAI/MiniMax-H3",
            "revision": args.source_revision,
            "config_sha256": source_config,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "metal_available": bool(getattr(mx, "metal", None) and mx.metal.is_available()),
        },
        "artifacts": {
            "candidate": {
                "path": str(candidate.resolve()),
                "recipe": candidate_audit["recipe"],
                "format": candidate_audit["format"],
            },
            "community": {
                "path": str(community.resolve()),
                "recipe": community_audit["recipe"],
                "format": community_audit["format"],
            },
        },
        "results": results,
        "paired_delta_candidate_minus_community": deltas,
        "decision": decision,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(
        "candidate-community weight relative-L2 delta "
        f"{deltas['weight_relative_l2']['mean']:+.8f} "
        f"[{deltas['weight_relative_l2']['ci95'][0]:+.8f}, "
        f"{deltas['weight_relative_l2']['ci95'][1]:+.8f}]"
    )
    print(f"wrote {Path(args.out).resolve()}")
    passed = all(decision.values())
    return 0 if passed or args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
