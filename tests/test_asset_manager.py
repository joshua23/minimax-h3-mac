"""Synthetic tests for the release asset manifest and fetch manager.

These tests use tiny local files only. They never download model weights, require
credentials, or touch the real model cache.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.asset_manager import ASSET_FAILURE, LICENSE_FAILURE, OK, load_manifest, validate_manifest
from minimax_h3_mlx.doctor import DoctorConfig, ProbeSet, run_doctor

GIB = 1024 ** 3


def assert_case(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} failed{(': ' + detail) if detail else ''}")
    print(f"ok  {name}{(' — ' + detail) if detail else ''}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(path: Path, *, source: Path, digest: str | None = None, size: int | None = None) -> Path:
    payload = source.read_bytes()
    manifest = {
        "schema_version": 1,
        "manifest_id": "synthetic-assets",
        "manifest_version": "test-v1",
        "contract": {"license_acceptance_env_var": "MINIMAX_H3_ACCEPT_LICENSE"},
        "assets": [
            {
                "id": "fake-asset",
                "aliases": ["fake"],
                "kind": "synthetic",
                "precision": "test",
                "repo_id": "local/fake",
                "repo_type": "model",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "local_subdir": "fake-asset",
                "fetch_enabled": True,
                "pinning_status": "synthetic_pinned",
                "license": {
                    "name": "synthetic-license",
                    "url": "file://synthetic",
                    "requires_user_acceptance": True,
                    "acceptance_env_var": "MINIMAX_H3_ACCEPT_LICENSE",
                },
                "files": [
                    {
                        "path": "payload.bin",
                        "size_bytes": len(payload) if size is None else size,
                        "sha256": sha256(payload) if digest is None else digest,
                        "source_path": str(source),
                        "role": "test_payload",
                        "required": True,
                    }
                ],
                "streaming_24gb_notes": {"risk_level": "synthetic"},
                "quality_claim_status": "synthetic_no_claim",
            }
        ],
    }
    path.write_text(json.dumps(manifest))
    return path


def cli(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "assets.py"), *args],
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )


def healthy_probes(clt_path: Path) -> ProbeSet:
    def which(name: str) -> str | None:
        return {
            "uv": "/usr/local/bin/uv",
            "ffmpeg": "/usr/local/bin/ffmpeg",
            "ffprobe": "/usr/local/bin/ffprobe",
            "xcode-select": "/usr/bin/xcode-select",
        }.get(name)

    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{clt_path}\n", stderr="")

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


def test_manifest_validation_and_package_resource() -> None:
    package_manifest = load_manifest()
    validation = validate_manifest(package_manifest)
    assert_case("package release asset manifest validates", validation["ok"], str(validation["issues"][:1]))

    invalid = {"schema_version": 1, "manifest_version": "bad", "assets": [{"id": "x", "repo_id": "r", "fetch_enabled": True, "license": {"name": "l"}, "files": []}]}
    validation = validate_manifest(invalid)
    assert_case("synthetic manifest validation catches incomplete pins", not validation["ok"])
    codes = {issue["code"] for issue in validation["issues"]}
    assert_case("missing revision and file pins are explicit", {"missing_revision", "missing_files"} <= codes)


def test_license_gate_offline_reuse_bad_hash_atomic_cleanup_and_resume() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.bin"
        source.write_bytes(b"0123456789abcdef" * 4)
        manifest = write_manifest(root / "manifest.json", source=source)
        cache = root / "cache"

        proc = cli("--manifest", str(manifest), "--cache-dir", str(cache), "fetch", "--asset", "fake")
        payload = json.loads(proc.stdout)
        assert_case("license refusal exits with license failure", proc.returncode == payload["exit_code"] == LICENSE_FAILURE)
        assert_case("license refusal does not create target", not (cache / "fake-asset" / "payload.bin").exists())

        target = cache / "fake-asset" / "payload.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        proc = cli("--manifest", str(manifest), "--cache-dir", str(cache), "fetch", "--asset", "fake", "--offline")
        payload = json.loads(proc.stdout)
        assert_case("verified fake asset is reused offline", proc.returncode == payload["exit_code"] == OK and payload["assets"][0]["status"] == "reused")

        target.write_bytes(b"x" * len(source.read_bytes()))
        proc = cli("--manifest", str(manifest), "--cache-dir", str(cache), "verify", "--asset", "fake")
        payload = json.loads(proc.stdout)
        codes = {issue["code"] for issue in payload["issues"]}
        assert_case("bad hash is a nonzero verification failure", proc.returncode == payload["exit_code"] == ASSET_FAILURE and "asset_hash_mismatch" in codes)
        target.unlink()

        bad_manifest = write_manifest(root / "bad-manifest.json", source=source, digest="0" * 64)
        proc = cli("--manifest", str(bad_manifest), "--cache-dir", str(cache), "fetch", "--asset", "fake", "--accept-license")
        payload = json.loads(proc.stdout)
        assert_case("bad downloaded hash fails", proc.returncode == payload["exit_code"] != OK)
        assert_case("hash failure leaves no final or part file", not target.exists() and not target.with_name("payload.bin.part").exists())

        part = target.with_name("payload.bin.part")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(source.read_bytes()[:7])
        proc = cli("--manifest", str(manifest), "--cache-dir", str(cache), "fetch", "--asset", "fake", "--accept-license")
        payload = json.loads(proc.stdout)
        events = payload["assets"][0]["files"][0]["fetch_events"]
        assert_case("local fetch resumes from existing part", proc.returncode == OK and events[0]["resumed_from_bytes"] == 7)
        assert_case("resumed fetch lands atomically with verified content", target.read_bytes() == source.read_bytes() and not part.exists())


def test_doctor_manifest_integration() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "pyproject.toml").write_text("[project]\nname='synthetic'\nversion='0'\n")
        (root / "uv.lock").write_text("version = 1\n")
        (root / "CommandLineTools").mkdir()
        (root / "source.bin").write_bytes(b"doctor")
        manifest = write_manifest(root / "manifest.json", source=root / "source.bin")
        result = run_doctor(
            DoctorConfig(
                cwd=root,
                env={"VIRTUAL_ENV": str(root / ".venv")},
                cache_dir=str(root / "cache"),
                output_dir=str(root),
                asset_manifest=str(manifest),
                asset_cache_dir=str(root / "cache"),
            ),
            healthy_probes(root / "CommandLineTools"),
        )
        check = next(item for item in result["checks"] if item["id"] == "asset_manifest")
        assert_case("doctor includes manifest asset status", check["observed"]["manifest_version"] == "test-v1")
        assert_case("doctor manifest status is non-generative", check["observed"]["downloads_models"] is False and check["observed"]["runs_generation"] is False)
        assert_case("doctor does not make missing manifest assets fatal by default", result["exit_code"] == OK)


def main() -> int:
    test_manifest_validation_and_package_resource()
    test_license_gate_offline_reuse_bad_hash_atomic_cleanup_and_resume()
    test_doctor_manifest_integration()
    print("asset manager focused tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
