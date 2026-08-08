"""Does quantizing `adaln_proj` actually hurt?

`adaln_proj` is 13B of H3's 33B and therefore dominates the download of every quantized build — a
2-bit core is 5 GB resident but ships inside a 31 GB repository because the AdaLN projections sit
next to it in bfloat16. Whether that is necessary is worth measuring rather than arguing.

The measurement is direct and cheap. `adaln_proj` exists only to produce the modulation table, so
the question is not what it does to a generated clip but simply: **how much does the table change?**
Build it from the bfloat16 projections, build it again from quantized ones, and compare — no forward
passes, no sampling, no trajectory.

The comparison reports both the relative error of the table and the resulting shift in the modulation
a block actually applies. `scale` and `gate` enter multiplicatively and `shift` additively, so a
constant relative error in the table is not equally harmful across the six tensors; they are broken
out separately.

    ./.venv/bin/python scripts/eval_adaln_quant.py --bits 8 6 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.adaln import ModulationCache
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.quantize import QuantConfig, _class_predicate
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler

NAMES = ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]


def schedule_table(steps: int) -> mx.array:
    video = MiniMaxH3Scheduler(shift=12.0)
    audio = MiniMaxH3Scheduler(shift=3.0)
    video.set_timesteps(steps)
    audio.set_timesteps(steps)
    values = sorted({round(float(v), 9) for v in video.timesteps.tolist() + audio.timesteps.tolist()})
    return mx.array(np.array(values, dtype=np.float32))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/Volumes/models/MiniMax-H3/FL2VA/transformer")
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()

    table = schedule_table(args.steps)
    print(f"schedule: {args.steps} sigma points -> {table.shape[0]} distinct timesteps")

    print("loading bfloat16 reference")
    started = time.perf_counter()
    model = load_dit(Path(args.checkpoint))
    print(f"  {time.perf_counter() - started:.1f}s")

    reference = ModulationCache.build(model, table, dtype=mx.float32)
    ref_tables = [[np.array(t) for t in reference.get(i)] for i in range(len(model.blocks))]
    print(f"  reference cache: {reference.nbytes() / 1e6:.0f} MB (float32)")

    import mlx.nn as nn

    print(f"\n{'bits':>5}  {'table rel-L2':>13}  {'worst tensor':>26}  {'scale drift':>12}  {'gate drift':>11}")
    for bits in args.bits:
        # Quantize `adaln_proj` alone, leaving the rest of the model untouched.
        config = QuantConfig(bits=bits, group_size=args.group_size, quantize_adaln=True, adaln_bits=bits)

        def predicate(path: str, module: nn.Module):
            if ".adaln_proj." not in path or not path.startswith("blocks."):
                return False
            return _class_predicate(config)(path, module)

        variant = load_dit(Path(args.checkpoint))
        nn.quantize(variant, group_size=args.group_size, bits=bits, class_predicate=predicate)
        mx.eval(variant.parameters())

        got = ModulationCache.build(variant, table, dtype=mx.float32)

        per_tensor = np.zeros(6)
        total_num = total_den = 0.0
        for i in range(len(variant.blocks)):
            for j, arr in enumerate(got.get(i)):
                a, b = np.array(arr), ref_tables[i][j]
                per_tensor[j] += float(np.linalg.norm(a - b) ** 2)
                total_num += float(np.linalg.norm(a - b) ** 2)
                total_den += float(np.linalg.norm(b) ** 2)
        rel = (total_num / total_den) ** 0.5

        # Per-tensor relative error, using each tensor's own norm.
        denom = np.zeros(6)
        for i in range(len(variant.blocks)):
            for j in range(6):
                denom[j] += float(np.linalg.norm(ref_tables[i][j]) ** 2)
        per_rel = np.sqrt(per_tensor / np.maximum(denom, 1e-30))
        worst = int(np.argmax(per_rel))

        # `scale` and `gate` act multiplicatively: report the drift in (1 + scale) and in gate.
        scale_drift = max(per_rel[1], per_rel[4])
        gate_drift = max(per_rel[2], per_rel[5])
        print(f"{bits:>5}  {rel:>13.6f}  {NAMES[worst] + ' ' + format(per_rel[worst], '.6f'):>26}  "
              f"{scale_drift:>12.6f}  {gate_drift:>11.6f}")
        del variant, got

    print("\nper-tensor relative error, all widths, for reference:")
    print("  shift/scale/gate enter the block as  x * (1 + scale) + shift  and  x + gate * f(x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
