"""Installed-package media smoke for clean-room release checks.

This CLI intentionally uses only the standard library plus the package media-tool
resolver.  It creates a tiny synthetic RGB+stereo WAV clip, muxes it with the
selected ffmpeg-compatible executable, validates the result with ffprobe, and
writes a machine-readable report.  It does not import MLX, load model weights, or
use the development checkout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

from minimax_h3_mlx.media_tools import media_tool_error, resolve_media_tool

OK = 0
VALIDATION_FAILURE = 3
USAGE_ERROR = 64
TOOL_FAILURE = 66


def _utc_seconds() -> float:
    return time.time()


def _run(
    cmd: list[str],
    *,
    input_bytes: bytes | None = None,
    env: Mapping[str, str] | None = None,
    capture_limit: int | None = 2000,
) -> dict[str, Any]:
    started = _utc_seconds()
    try:
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            check=False,
            text=False,
            capture_output=True,
            env=None if env is None else dict(env),
            timeout=60,
        )
        stdout = proc.stdout.decode(errors="replace")
        stderr = proc.stderr.decode(errors="replace")
        if capture_limit is not None:
            stdout = stdout[:capture_limit]
            stderr = stderr[:capture_limit]
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "wall_seconds": _utc_seconds() - started,
            "stdout": stdout,
            "stderr": stderr,
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "returncode": 124,
            "wall_seconds": _utc_seconds() - started,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _version_line(command: str) -> tuple[str | None, dict[str, Any]]:
    record = _run([command, "-version"])
    line = (record.get("stdout") or record.get("stderr") or "").splitlines()
    return (line[0][:500] if line else None), record


def _write_stereo_wav(path: Path, *, duration_seconds: float, sample_rate: int) -> dict[str, Any]:
    samples = max(1, int(duration_seconds * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(samples):
            t = index / sample_rate
            left = int(0.22 * 32767.0 * math.sin(2.0 * math.pi * 440.0 * t))
            right = int(0.18 * 32767.0 * math.sin(2.0 * math.pi * 660.0 * t))
            frames.extend(struct.pack("<hh", left, right))
        handle.writeframes(bytes(frames))
    return {"path": str(path), "samples": samples, "sample_rate_hz": sample_rate, "channels": 2}


def _rgb_frames(width: int, height: int, frames: int) -> bytes:
    payload = bytearray(width * height * 3 * frames)
    offset = 0
    for frame in range(frames):
        for y in range(height):
            for x in range(width):
                payload[offset] = (x * 3 + frame * 17) % 256
                payload[offset + 1] = (y * 5 + frame * 29) % 256
                payload[offset + 2] = ((x + y) * 2 + frame * 11) % 256
                offset += 3
    return bytes(payload)


def _stream_validation(ffprobe_payload: dict[str, Any], *, expected_sample_rate: int) -> dict[str, Any]:
    streams = ffprobe_payload.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    audio_rate = None
    audio_channels = None
    if audio_streams:
        try:
            audio_rate = int(audio_streams[0].get("sample_rate"))
        except Exception:
            audio_rate = None
        audio_channels = audio_streams[0].get("channels")
    ok = bool(video_streams and audio_streams and audio_rate == expected_sample_rate and audio_channels == 2)
    return {
        "ok": ok,
        "video_streams": len(video_streams),
        "audio_streams": len(audio_streams),
        "audio_sample_rate_hz": audio_rate,
        "audio_channels": audio_channels,
        "expected_audio_sample_rate_hz": expected_sample_rate,
        "expected_audio_channels": 2,
        "video_codec": video_streams[0].get("codec_name") if video_streams else None,
        "audio_codec": audio_streams[0].get("codec_name") if audio_streams else None,
    }


def run_media_smoke(args: argparse.Namespace, *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env_map = os.environ if env is None else env
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).expanduser() if args.work_dir else output.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / "synthetic_stereo_32k.wav"

    ffmpeg = resolve_media_tool("ffmpeg", args.ffmpeg, env=env_map, cwd=Path.cwd(), allow_path=not args.no_system_path)
    ffprobe = resolve_media_tool("ffprobe", args.ffprobe, env=env_map, cwd=Path.cwd(), allow_path=not args.no_system_path)
    commands: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "minimax-h3-media-smoke",
        "ok": False,
        "exit_code": OK,
        "created_at": _utc_seconds(),
        "cwd": str(Path.cwd()),
        "python": {"executable": sys.executable, "version": sys.version.split()[0], "prefix": sys.prefix},
        "environment": {
            "PATH": env_map.get("PATH", ""),
            "PYTHONPATH": env_map.get("PYTHONPATH", ""),
            "HOME": env_map.get("HOME", ""),
            "XDG_CACHE_HOME": env_map.get("XDG_CACHE_HOME", ""),
            "system_path_lookup_allowed": not args.no_system_path,
        },
        "media_tools": {"ffmpeg": ffmpeg.to_dict(), "ffprobe": ffprobe.to_dict()},
        "commands": commands,
        "output": {"path": str(output), "exists": False, "size_bytes": 0},
        "validation": {"ok": False},
    }

    missing = [resolution for resolution in (ffmpeg, ffprobe) if not resolution.ok]
    if missing:
        report["exit_code"] = TOOL_FAILURE
        report["issues"] = [media_tool_error(item) for item in missing]
        return report

    ffmpeg_version, ffmpeg_version_record = _version_line(str(ffmpeg.command))
    ffprobe_version, ffprobe_version_record = _version_line(str(ffprobe.command))
    commands.extend([ffmpeg_version_record, ffprobe_version_record])
    report["media_tools"]["ffmpeg"]["version_line"] = ffmpeg_version
    report["media_tools"]["ffprobe"]["version_line"] = ffprobe_version
    if ffmpeg_version_record["returncode"] != 0 or ffprobe_version_record["returncode"] != 0:
        report["exit_code"] = TOOL_FAILURE
        report["issues"] = ["media tool -version command failed"]
        return report

    audio_info = _write_stereo_wav(wav_path, duration_seconds=args.duration, sample_rate=args.sample_rate)
    video_bytes = _rgb_frames(args.width, args.height, args.frames)
    fps = args.frames / args.duration
    mux_cmd = [
        str(ffmpeg.command),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-i",
        str(wav_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        str(args.sample_rate),
        "-ac",
        "2",
        "-shortest",
        str(output),
    ]
    mux_record = _run(mux_cmd, input_bytes=video_bytes)
    commands.append(mux_record)
    report["synthetic_input"] = {
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": fps,
        "duration_seconds": args.duration,
        "audio": audio_info,
        "raw_video_bytes": len(video_bytes),
    }
    if mux_record["returncode"] != 0:
        report["exit_code"] = TOOL_FAILURE
        report["issues"] = ["ffmpeg mux command failed"]
        return report

    probe_cmd = [
        str(ffprobe.command),
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(output),
    ]
    probe_record = _run(probe_cmd, capture_limit=None)
    commands.append(probe_record)
    size = output.stat().st_size if output.exists() else 0
    report["output"] = {"path": str(output), "exists": output.exists(), "size_bytes": size}
    if probe_record["returncode"] != 0:
        report["exit_code"] = VALIDATION_FAILURE
        report["issues"] = ["ffprobe validation command failed"]
        return report

    try:
        payload = json.loads(probe_record["stdout"])
    except json.JSONDecodeError as exc:
        report["exit_code"] = VALIDATION_FAILURE
        report["issues"] = [f"ffprobe JSON parse failed: {exc}"]
        return report
    validation = _stream_validation(payload, expected_sample_rate=args.sample_rate)
    validation["ffprobe_format"] = payload.get("format", {})
    report["validation"] = validation
    report["ok"] = bool(validation["ok"] and size > 0)
    report["exit_code"] = OK if report["ok"] else VALIDATION_FAILURE
    if not report["ok"]:
        report["issues"] = ["MP4 stream validation did not find the expected video+32kHz stereo audio streams"]
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and validate a tiny synthetic audio+video MP4 using release media tools.")
    parser.add_argument("--output", required=True, help="MP4 output path")
    parser.add_argument("--report", default=None, help="write JSON report to this path")
    parser.add_argument("--work-dir", default=None, help="scratch directory for the temporary WAV")
    parser.add_argument("--ffmpeg", default=None, help="explicit ffmpeg-compatible executable")
    parser.add_argument("--ffprobe", default=None, help="explicit ffprobe-compatible executable")
    parser.add_argument("--no-system-path", action="store_true", help="disable PATH fallback in the media resolver")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=32000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.frames <= 0 or args.duration <= 0 or args.sample_rate <= 0:
        parser.error("width, height, frames, duration, and sample-rate must be positive")
    report = run_media_smoke(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n")
    print(text)
    return int(report["exit_code"])


if __name__ == "__main__":  # pragma: no cover - exercised through the console entry point.
    raise SystemExit(main())
