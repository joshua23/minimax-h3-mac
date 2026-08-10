"""Synthetic tests for the quality-suite matrix runner.

These tests exercise manifest expansion and dry-run summary writing only. They do
not run MiniMax-H3 generation, open model weights, or require ffmpeg/ffprobe.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_runner():
    path = ROOT / "scripts" / "run_quality_suite_matrix.py"
    spec = importlib.util.spec_from_file_location("run_quality_suite_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_case(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} failed{(': ' + detail) if detail else ''}")
    print(f"ok  {name}{(' — ' + detail) if detail else ''}")


def test_matrix_expands_six_cases_three_conditions_first_seed_only() -> None:
    runner = load_runner()
    manifest = runner.load_manifest(ROOT / "experiments" / "quality_suite_manifest.json")
    matrix = runner.build_matrix(manifest)
    assert_case("matrix has 18 generation items", len(matrix) == 18)
    assert_case("matrix uses 320x192 small gate", {(item.width, item.height) for item in matrix} == {(320, 192)})
    assert_case("matrix uses first seed only", {item.seed for item in matrix} == {101, 202, 303, 404, 505, 606})
    by_condition = {condition["condition_id"] for condition in manifest["planned_initial_conditions"]}
    assert_case("matrix includes all three conditions", {item.condition_id for item in matrix} == by_condition)
    turbo = [item for item in matrix if item.condition_id == "4bit_turbo_5sigma"]
    non_turbo = [item for item in matrix if item.condition_id != "4bit_turbo_5sigma"]
    assert_case("only turbo condition requires lora", all(item.turbo_lora_required for item in turbo) and not any(item.turbo_lora_required for item in non_turbo))
    assert_case("5 sigma maps to 4 NFE", all(item.denoiser_evaluations == 4 for item in matrix if item.steps_sigma_points == 5))
    assert_case("9 sigma maps to 8 NFE", all(item.denoiser_evaluations == 8 for item in matrix if item.steps_sigma_points == 9))


def test_generation_command_turbo_lora_only_for_turbo() -> None:
    runner = load_runner()
    manifest = runner.load_manifest(ROOT / "experiments" / "quality_suite_manifest.json")
    matrix = runner.build_matrix(manifest)
    turbo = next(item for item in matrix if item.condition_id == "4bit_turbo_5sigma")
    non_turbo = next(item for item in matrix if item.condition_id == "4bit_non_turbo_9sigma")
    turbo_cmd = runner.generation_command(turbo, manifest)
    non_turbo_cmd = runner.generation_command(non_turbo, manifest)
    assert_case("turbo command includes lora flag", "--turbo-lora" in turbo_cmd)
    assert_case("non-turbo command omits lora flag", "--turbo-lora" not in non_turbo_cmd)
    assert_case("commands require muxed MP4", "--require-muxed-mp4" in turbo_cmd and "--require-muxed-mp4" in non_turbo_cmd)
    assert_case("commands use repo-local ffmpeg", ".venv/bin/static_ffmpeg" in turbo_cmd)


def test_dry_run_writes_machine_readable_plan_and_summary() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as raw:
        code = runner.main(["--dry-run", "--run-id", "unit-dry-run", "--suite-root", raw])
        summary_path = Path(raw) / "h3_delivery_quality_v1" / "runs" / "unit-dry-run" / "summary.json"
        plan_path = summary_path.with_name("plan.json")
        assert_case("dry-run exits zero", code == 0)
        assert_case("dry-run writes plan", plan_path.exists())
        assert_case("dry-run writes summary", summary_path.exists())
        summary = json.loads(summary_path.read_text())
        assert_case("dry-run status is explicit", summary["status"] == "dry_run_plan_only")
        assert_case("dry-run plans 18 generations", summary["matrix_count"] == 18)
        assert_case("dry-run plans 12 metric pairs", summary["expected_pair_metrics"] == 12)
        assert_case("dry-run records no completed generations", summary["completed_generations"] == 0)


def main() -> int:
    test_matrix_expands_six_cases_three_conditions_first_seed_only()
    test_generation_command_turbo_lora_only_for_turbo()
    test_dry_run_writes_machine_readable_plan_and_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
