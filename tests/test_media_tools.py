"""Synthetic tests for release-safe media-tool resolution.

These tests create tiny executable placeholders only; they never invoke real
ffmpeg/ffprobe or require system media tools.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.media_tools import FFMPEG_ENV_VAR, media_tool_error, resolve_media_tool


def assert_case(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} failed{(': ' + detail) if detail else ''}")
    print(f"ok  {name}{(' — ' + detail) if detail else ''}")


def write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_explicit_env_local_then_path_priority() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        explicit = write_executable(root / "explicit" / "ffmpeg")
        env_tool = write_executable(root / "env" / "ffmpeg")
        local_static = write_executable(root / "venv" / "static_ffmpeg")
        path_tool = str(root / "path" / "ffmpeg")

        def which(name: str) -> str | None:
            return path_tool if name == "ffmpeg" else None

        env = {FFMPEG_ENV_VAR: str(env_tool), "PATH": "<synthetic-path>"}
        explicit_result = resolve_media_tool(
            "ffmpeg",
            str(explicit),
            env=env,
            cwd=root,
            local_bin_dirs=[root / "venv"],
            which=which,
        )
        assert_case("explicit path wins", explicit_result.path == str(explicit))
        assert_case("explicit path route is recorded", explicit_result.route == "explicit_path")

        env_result = resolve_media_tool(
            "ffmpeg",
            None,
            env=env,
            cwd=root,
            local_bin_dirs=[root / "venv"],
            which=which,
        )
        assert_case("env path wins over local candidates", env_result.path == str(env_tool))
        assert_case("env source is recorded", env_result.source == f"env:{FFMPEG_ENV_VAR}")

        local_result = resolve_media_tool(
            "ffmpeg",
            None,
            env={"PATH": "<synthetic-path>"},
            cwd=root,
            local_bin_dirs=[root / "venv"],
            which=which,
        )
        assert_case("local static_ffmpeg alias wins before PATH", local_result.path == str(local_static))
        assert_case("local route is recorded", local_result.route == "local_candidate")

        path_result = resolve_media_tool(
            "ffmpeg",
            None,
            env={"PATH": "<synthetic-path>"},
            cwd=root,
            local_bin_dirs=[root / "empty"],
            which=which,
        )
        assert_case("PATH fallback is last", path_result.path == path_tool)
        assert_case("PATH fallback route is recorded", path_result.route == "path_fallback")


def test_sanitized_path_failure_message_is_actionable() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        result = resolve_media_tool(
            "ffmpeg",
            None,
            env={"PATH": ""},
            cwd=root,
            local_bin_dirs=[root / "empty"],
            allow_path=False,
            which=lambda name: None,
        )
        message = media_tool_error(result)
        assert_case("unresolved result is not ok", not result.ok)
        assert_case("message names explicit flag", "--ffmpeg" in message)
        assert_case("message names env override", FFMPEG_ENV_VAR in message)
        assert_case("message names sanitized PATH state", "system PATH lookup was disabled" in message and "PATH=''" in message)


def main() -> int:
    test_explicit_env_local_then_path_priority()
    test_sanitized_path_failure_message_is_actionable()
    print("media tool resolver tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
