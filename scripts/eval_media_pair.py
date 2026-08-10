#!/usr/bin/env python3
"""Compare two existing MiniMax-H3 media artifacts with paired final-media metrics.

This is a decoder/evaluator only: it does not run generation and does not fetch
or mutate model weights.  The two clips must come from the same prompt, seed,
resolution, duration, and sigma schedule for the pixel/waveform metrics to be
interpretable.

Example:
    ./.venv/bin/python scripts/eval_media_pair.py \
      --reference out/quality_suite/ref.mp4 \
      --candidate out/quality_suite/candidate.mp4 \
      --ffmpeg .venv/bin/static_ffmpeg \
      --ffprobe .venv/bin/static_ffprobe \
      --out out/quality_suite/pair_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.media_metrics import paired_media_file_metrics, write_metrics_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="reference/teacher media path")
    parser.add_argument("--candidate", required=True, help="candidate media path")
    parser.add_argument("--out", default=None, help="write metrics JSON here; stdout when omitted")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg executable, preferably repo-local .venv/bin/static_ffmpeg")
    parser.add_argument("--ffprobe", default=None, help="ffprobe executable, preferably repo-local .venv/bin/static_ffprobe")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="audio decode/evaluation sample rate")
    parser.add_argument("--max-audio-lag-seconds", type=float, default=0.5, help="sync search window")
    parser.add_argument("--max-video-frames", type=int, default=None, help="optional prefix-frame cap for quick checks")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    args = parser.parse_args()

    payload = paired_media_file_metrics(
        args.reference,
        args.candidate,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        sample_rate_hz=args.sample_rate,
        max_audio_lag_seconds=args.max_audio_lag_seconds,
        max_video_frames=args.max_video_frames,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
    )
    if args.out:
        out_path = write_metrics_json(args.out, payload)
        print(f"wrote {out_path}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
