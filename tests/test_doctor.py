"""Synthetic tests for the lightweight MiniMax-H3 release doctor.

These tests monkeypatch host probes and never generate media, download models, or
materialize safetensors tensors.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.doctor import CONTRACT_FAILURE, OK, DoctorConfig, ProbeSet, run_doctor

GIB = 1024 ** 3


def assert_case(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} failed{(': ' + detail) if detail else ''}")
    print(f"ok  {name}{(' — ' + detail) if detail else ''}")


def issue_codes(result: dict) -> set[str]:
    return {issue["code"] for issue in result["issues"]}


def write_source_lock(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname='synthetic'\nversion='0'\n")
    (root / "uv.lock").write_text("version = 1\n")


def healthy_probes(clt_path: Path, *, missing_tools: set[str] | None = None) -> ProbeSet:
    missing_tools = missing_tools or set()
    tool_map = {
        "uv": "/usr/local/bin/uv",
        "ffmpeg": "/usr/local/bin/ffmpeg",
        "ffprobe": "/usr/local/bin/ffprobe",
        "xcode-select": "/usr/bin/xcode-select",
    }

    def which(name: str) -> str | None:
        if name in missing_tools:
            return None
        return tool_map.get(name)

    def run(cmd, **kwargs):
        if cmd == ["/usr/bin/xcode-select", "-p"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{clt_path}\n", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected command")

    return ProbeSet(
        system=lambda: "Darwin",
        machine=lambda: "arm64",
        mac_ver=lambda: ("15.2", ("", "", ""), ""),
        python_version=lambda: "3.12.13",
        python_executable=lambda: "/tmp/project/.venv/bin/python",
        python_prefix=lambda: "/tmp/project/.venv",
        python_base_prefix=lambda: "/Library/Frameworks/Python.framework/Versions/3.12",
        which=which,
        run=run,
        memory_bytes=lambda: 24 * GIB,
        disk_usage=lambda path: shutil._ntuple_diskusage(400 * GIB, 100 * GIB, 300 * GIB),
        geteuid=lambda: 501,
    )


def test_happy_path_all_contract_prereqs_available_but_no_assets_or_models_loaded() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_source_lock(root)
        (root / "CommandLineTools").mkdir()
        (root / "cache").mkdir()
        (root / "out").mkdir()
        result = run_doctor(
            DoctorConfig(
                cwd=root,
                env={"VIRTUAL_ENV": str(root / ".venv")},
                cache_dir="cache",
                output_dir="out",
            ),
            healthy_probes(root / "CommandLineTools"),
        )
        assert_case("doctor succeeds when fatal contract checks pass", result["exit_code"] == OK and result["ok"])
        assert_case("missing model paths are warnings, not implicit defaults", "model_path_not_configured" in issue_codes(result))
        assert_case("asset preflight is not run without explicit roots", "asset_preflight_not_run" in issue_codes(result))
        assert_case("doctor states it does not generate or download", result["contract"]["runs_generation"] is False and result["contract"]["downloads_models"] is False)


def test_missing_tools_are_structured_fatal_contract_failures() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_source_lock(root)
        (root / "CommandLineTools").mkdir()
        result = run_doctor(
            DoctorConfig(cwd=root, env={"VIRTUAL_ENV": str(root / ".venv")}, cache_dir=root, output_dir=root),
            healthy_probes(root / "CommandLineTools", missing_tools={"uv", "ffmpeg", "ffprobe"}),
        )
        codes = issue_codes(result)
        assert_case("doctor fails when required executables are absent", result["exit_code"] == CONTRACT_FAILURE and not result["ok"])
        assert_case("missing tool issue is machine-readable", "missing_required_tool" in codes)
        missing = {issue["tool"] for issue in result["issues"] if issue["code"] == "missing_required_tool"}
        assert_case("uv ffmpeg and ffprobe are all named", {"uv", "ffmpeg", "ffprobe"} <= missing)


def test_forbidden_argus_paths_and_asset_failures_are_reported_without_loading_weights() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_source_lock(root)
        (root / "CommandLineTools").mkdir()
        result = run_doctor(
            DoctorConfig(
                cwd=root,
                env={"VIRTUAL_ENV": str(root / ".venv")},
                checkpoint="/tmp/.argus-skill/models/MiniMax-H3/FL2VA",
                transformer="missing-transformer",
                text_encoder="missing-text-encoder",
                cache_dir=root,
                output_dir=root,
                asset_roots=("missing-assets",),
                require_model_paths=True,
            ),
            healthy_probes(root / "CommandLineTools"),
        )
        codes = issue_codes(result)
        assert_case("Argus paths are fatal for release configuration", "forbidden_argus_path" in codes)
        assert_case("missing configured model roots are fatal when required", "configured_model_path_missing" in codes)
        assert_case("header-only asset failures are summarized", "asset_preflight_failed" in codes)
        preflight = next(check for check in result["checks"] if check["id"] == "assets")
        assert_case("asset preflight records missing root code", preflight["observed"]["preflight_issues"][0]["code"] == "missing_root")


def test_cli_emits_json_and_exit_code_matches_report() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "doctor.py"), "--json", "--min-free-gib", "999999"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert_case("doctor CLI emits JSON", payload["tool"] == "minimax-h3-doctor")
    assert_case("doctor CLI return code matches report", proc.returncode == payload["exit_code"] == CONTRACT_FAILURE)
    assert_case("forced disk failure is explicit", "insufficient_free_disk" in issue_codes(payload))


def main() -> int:
    test_happy_path_all_contract_prereqs_available_but_no_assets_or_models_loaded()
    test_missing_tools_are_structured_fatal_contract_failures()
    test_forbidden_argus_paths_and_asset_failures_are_reported_without_loading_weights()
    test_cli_emits_json_and_exit_code_matches_report()
    print("doctor focused tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
