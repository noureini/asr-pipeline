"""Calibrate the GER snap threshold from a real run's distance distribution.

The snap fires when the fused [bn] is farther than `ger_snap_tau` from its
NEAREST witness. Picking that tau by intuition is wrong — it must sit where
GOOD reconciliations and BAD (hallucinated) ones actually separate on YOUR
audio. This reads the `min_dist` values the guard wrote into a run's
`.reasoning.txt` (observe-mode, ger_snap=false) and prints the distribution
so you can choose the threshold, plus the segments at the high end to eyeball.

Workflow:
  1. run with ger_guard: true, ger_snap: false  (observe-mode)
  2. uv run --no-sync python scripts/ger_calibrate.py outputs/<run>/<stem>.reasoning.txt
  3. eyeball the high-min_dist segments: are they hallucinations (snap helps)
     or good reconciliations (snap would HURT)? Set ger_snap_tau just ABOVE
     the worst good segment, then set ger_snap: true and A/B.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: ger_calibrate.py <run>.reasoning.txt", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")
    # Each segment block has a header "[HH:MM:SS] ROLE", a "bn   : ..." line,
    # and a "ger  : min_dist=.. agree=.. [reasons]" line (observe-mode).
    blocks = re.split(r"\n(?=\[\d\d:\d\d:\d\d\])", text)
    rows = []
    for b in blocks:
        m = re.search(r"min_dist=([0-9.]+)(?:\s+agree=([0-9.]+))?", b)
        if not m:
            continue
        head = b.splitlines()[0].strip()
        bn = ""
        mb = re.search(r"^\s*bn\s*:\s*(.*)$", b, re.M)
        if mb:
            bn = mb.group(1).strip()
        reasons = ""
        mr = re.search(r"min_dist=[^\[]*\[([^\]]*)\]", b)
        if mr:
            reasons = mr.group(1)
        rows.append({
            "head": head,
            "min_dist": float(m.group(1)),
            "agree": float(m.group(2)) if m.group(2) else None,
            "bn": bn[:80],
            "reasons": reasons,
        })

    if not rows:
        print("no `ger  : min_dist=...` lines found. Run with ger_guard: true "
              "(observe-mode writes them into .reasoning.txt).", file=sys.stderr)
        return 1

    dists = sorted(r["min_dist"] for r in rows)
    n = len(dists)

    def pct(q: float) -> float:
        return dists[min(n - 1, int(q * n))]

    print(f"GER min_dist distribution over {n} segments  ({path.name})")
    print(f"  min   {dists[0]:.3f}")
    print(f"  p25   {pct(0.25):.3f}")
    print(f"  p50   {pct(0.50):.3f}")
    print(f"  p75   {pct(0.75):.3f}")
    print(f"  p90   {pct(0.90):.3f}")
    print(f"  p95   {pct(0.95):.3f}")
    print(f"  max   {dists[-1]:.3f}")

    # coarse histogram
    print("\n  histogram (bin width 0.1):")
    for lo in [x / 10 for x in range(0, 10)]:
        c = sum(1 for d in dists if lo <= d < lo + 0.1)
        if c:
            print(f"    {lo:.1f}–{lo + 0.1:.1f}  {'█' * c} {c}")

    print("\n  highest-distance segments — EYEBALL these (hallucination → snap "
          "helps; good reconciliation → snap would HURT):")
    for r in sorted(rows, key=lambda x: -x["min_dist"])[:12]:
        rs = f"  [{r['reasons']}]" if r["reasons"] else ""
        print(f"    {r['min_dist']:.3f}  {r['head']}{rs}")
        print(f"           bn: {r['bn']}")

    print("\n  → set ger_snap_tau just ABOVE the worst GOOD segment, then "
          "ger_snap: true and A/B (watch for snap-induced regressions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
