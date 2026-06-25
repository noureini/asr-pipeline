"""
Side-by-side compare of two eval_omni_inproc.py result JSONs.

Usage:
    uv run python scripts/compare_omni_inproc.py \
        results/omni_inproc_300m.json results/omni_inproc_1b.json

Prints a per-clip M1/M2 table for both models with deltas, plus the
mean row and a one-line verdict (which model wins, by how much).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: str) -> dict:
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    d["_path"] = p
    return d


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: compare_omni_inproc.py <A.json> <B.json>")
        sys.exit(1)
    a, b = load(sys.argv[1]), load(sys.argv[2])

    la = a.get("model_card", a["_path"])
    lb = b.get("model_card", b["_path"])
    ra = {r["file"]: r for r in a["rows"]}
    rb = {r["file"]: r for r in b["rows"]}
    files = sorted(set(ra) | set(rb))

    print(f"\nA = {la}   ({a['_path']})")
    print(f"B = {lb}   ({b['_path']})")
    print(f"A: load {a.get('load_s',0):.1f}s infer {a.get('infer_total_s',0):.1f}s "
          f"device {a.get('device','?')} GPU {a.get('gpu_mem_mb',0):.0f}MiB")
    print(f"B: load {b.get('load_s',0):.1f}s infer {b.get('infer_total_s',0):.1f}s "
          f"device {b.get('device','?')} GPU {b.get('gpu_mem_mb',0):.0f}MiB")
    print("=" * 74)
    print(f"{'file':>10} | {'A M1':>6} {'A M2':>6} | {'B M1':>6} {'B M2':>6} "
          f"| {'ΔM1':>6} {'ΔM2':>6}")
    print("-" * 74)
    for f in files:
        am1 = ra.get(f, {}).get("cer_m1", float("nan"))
        am2 = ra.get(f, {}).get("cer_m2", float("nan"))
        bm1 = rb.get(f, {}).get("cer_m1", float("nan"))
        bm2 = rb.get(f, {}).get("cer_m2", float("nan"))
        # delta B-A: negative = B better (lower CER)
        d1, d2 = bm1 - am1, bm2 - am2
        print(f"{f:>10} | {am1:6.1f} {am2:6.1f} | {bm1:6.1f} {bm2:6.1f} "
              f"| {d1:+6.1f} {d2:+6.1f}")
    print("-" * 74)
    print(f"{'MEAN':>10} | {a['mean_cer_m1']:6.1f} {a['mean_cer_m2']:6.1f} "
          f"| {b['mean_cer_m1']:6.1f} {b['mean_cer_m2']:6.1f} "
          f"| {b['mean_cer_m1']-a['mean_cer_m1']:+6.1f} "
          f"{b['mean_cer_m2']-a['mean_cer_m2']:+6.1f}")

    dm2 = b["mean_cer_m2"] - a["mean_cer_m2"]
    if abs(dm2) < 0.1:
        verdict = f"≈ tie on M2 (Δ {dm2:+.2f} pts)"
    elif dm2 < 0:
        verdict = (f"B ({lb}) wins: {abs(dm2):.2f} pts lower M2 CER "
                   f"({a['mean_cer_m2']:.1f}% → {b['mean_cer_m2']:.1f}%)")
    else:
        verdict = (f"A ({la}) wins: {dm2:.2f} pts lower M2 CER "
                   f"({b['mean_cer_m2']:.1f}% vs {a['mean_cer_m2']:.1f}%)")
    print(f"\nVERDICT (M2): {verdict}")


if __name__ == "__main__":
    main()
