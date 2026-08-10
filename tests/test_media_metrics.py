"""Synthetic tests for the paired quality-suite media metrics.

These tests use tiny NumPy arrays and the fixed manifest only.  They do not run
MiniMax-H3 generation, open model weights, or require ffmpeg/ffprobe.

Run with:
    ./.venv/bin/python tests/test_media_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.media_metrics import audio_pair_metrics, video_pair_metrics, write_metrics_json


REQUIRED_AXES = {
    "person_motion",
    "animal",
    "detail_texture",
    "text_geometry",
    "camera_motion",
    "audio_sync",
}
ALLOWED_TASK_ACTIVITIES = {"research", "execute", "verify"}


def assert_case(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} failed{(': ' + detail) if detail else ''}")
    print(f"ok  {name}{(' — ' + detail) if detail else ''}")


def test_video_metrics_identity_and_degradation() -> None:
    reference = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    reference[0, :, :, 0] = 20
    reference[1, :, :, 1] = np.arange(8, dtype=np.uint8)[None, :] * 20
    reference[2, :, :, 2] = np.arange(8, dtype=np.uint8)[:, None] * 20

    identical = video_pair_metrics(reference, reference.copy(), fps=24.0)
    assert_case("identical video has zero RGB MSE", identical["pixel"]["rgb_mse"] == 0.0)
    assert_case("identical video has infinite PSNR flag", identical["pixel"]["rgb_psnr_is_infinite"] is True)
    assert_case("identical video has SSIM 1", abs(identical["ssim"]["luma_global_mean"] - 1.0) < 1e-12)
    assert_case("identical video has zero temporal delta error", identical["temporal"]["temporal_delta_luma_mse"] == 0.0)

    candidate = reference.copy()
    candidate[:, 2:6, 2:6, :] = np.clip(candidate[:, 2:6, 2:6, :] + 35, 0, 255)
    degraded = video_pair_metrics(reference, candidate, fps=24.0)
    assert_case("degraded video has positive RGB MSE", degraded["pixel"]["rgb_mse"] > 0.0)
    assert_case("degraded video has finite PSNR", degraded["pixel"]["rgb_psnr_is_infinite"] is False)
    assert_case("degraded video lowers SSIM", degraded["ssim"]["luma_global_mean"] < 1.0)
    assert_case("degraded video records color delta", degraded["color"]["rgb_mean_abs_delta"] > 0.0)
    assert_case("degraded video records edge proxy", degraded["perceptual_proxy"]["edge_luma_mse"] >= 0.0)


def test_audio_metrics_activity_and_sync_lag() -> None:
    sample_rate = 1000
    reference = np.zeros(sample_rate * 2, dtype=np.float64)
    reference[250:350] = 0.8
    reference[900:1000] = -0.6
    lag_samples = 100
    candidate = np.zeros_like(reference)
    candidate[lag_samples:] = reference[:-lag_samples]

    metrics = audio_pair_metrics(reference, candidate, sample_rate_hz=sample_rate, max_lag_seconds=0.25)
    assert_case("audio candidate has activity", metrics["activity"]["candidate"]["activity_ok"] is True)
    assert_case("audio waveform MSE detects shift", metrics["waveform"]["mse"] > 0.0)
    assert_case(
        "audio sync lag sign is candidate-positive",
        abs(metrics["sync"]["candidate_lag_seconds"] - 0.1) <= 0.02,
        str(metrics["sync"]),
    )
    assert_case("audio sync correlation is strong", metrics["sync"]["max_envelope_correlation"] > 0.8)


def test_quality_suite_manifest_contract() -> None:
    manifest_path = ROOT / "experiments" / "quality_suite_manifest.json"
    payload = json.loads(manifest_path.read_text())
    assert_case("manifest declares schema version", payload["schema_version"] == 1)
    assert_case("manifest uses only allowed task activities", set(payload["task_activities"]) <= ALLOWED_TASK_ACTIVITIES)
    assert_case("manifest fixes execute/verify activities", payload["task_activities"] == ["execute", "verify"])
    assert_case("manifest records no-generation policy", payload["generation_policy"]["this_increment_runs_model_generation"] is False)
    assert_case("manifest records no-download policy", payload["generation_policy"]["this_increment_downloads_weights"] is False)

    cases = payload["cases"]
    axes = {case["coverage_axis"] for case in cases}
    assert_case("manifest covers required prompt axes", REQUIRED_AXES <= axes, str(sorted(axes)))
    assert_case("manifest has one case per required axis", len(cases) >= len(REQUIRED_AXES))
    seen_seeds: set[int] = set()
    for case in cases:
        assert_case(f"{case['case_id']} prompt is fixed text", isinstance(case["prompt"], str) and len(case["prompt"]) > 40)
        assert_case(f"{case['case_id']} has exactly two fixed seeds", len(case["seeds"]) == 2)
        assert_case(f"{case['case_id']} has expected observables", len(case["expected_observables"]) >= 3)
        for seed in case["seeds"]:
            assert_case(f"seed {seed} is an int", isinstance(seed, int))
            assert_case(f"seed {seed} is unique", seed not in seen_seeds)
            seen_seeds.add(seed)

    layout = payload["output_layout"]
    for key in ("media_template", "metadata_template", "pair_metrics_template", "blind_review_table_template"):
        assert_case(f"output layout includes {key}", key in layout and "out/quality_suite/" in layout[key])
    metric_families = set(payload["metrics_contract"]["metric_families"])
    for required in ("rgb_psnr", "luma_ssim", "temporal_luma_delta_psnr", "audio_activity_sync_lag"):
        assert_case(f"metrics contract includes {required}", required in metric_families)


def test_metrics_json_is_strict(tmp_dir: Path | None = None) -> None:
    import tempfile

    reference = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    payload = {"video": video_pair_metrics(reference, reference.copy())}
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "metrics.json"
        write_metrics_json(path, payload)
        loaded = json.loads(path.read_text())
    assert_case("strict JSON converts infinite PSNR to null", loaded["video"]["pixel"]["rgb_psnr_db"] is None)
    assert_case("strict JSON preserves infinite PSNR flag", loaded["video"]["pixel"]["rgb_psnr_is_infinite"] is True)


def main() -> int:
    test_video_metrics_identity_and_degradation()
    test_audio_metrics_activity_and_sync_lag()
    test_quality_suite_manifest_contract()
    test_metrics_json_is_strict()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
