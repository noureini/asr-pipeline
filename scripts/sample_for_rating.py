"""Draw a reproducible random sample of speech segments for a manual
coherence rating (fixed seed = auditable). Prints index / time / role /
bn / en so each can be rated on a 3-level rubric.

Usage:
    uv run python scripts/sample_for_rating.py <transcript.txt> --n 50
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

HEAD = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s+([A-Z_]+):")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    segs, cur = [], None
    for ln in args.transcript.read_text(encoding="utf-8").splitlines():
        m = HEAD.match(ln)
        if m:
            cur = {"t": m.group(1), "role": m.group(2), "bn": "", "en": ""}
            segs.append(cur)
            continue
        if cur is None:
            continue
        if ln.startswith("[bn]"):
            cur["bn"] = ln[4:].strip()
        elif ln.startswith("[en]"):
            cur["en"] = ln[4:].strip()

    # only segments with an actual English line (skip empties)
    segs = [s for s in segs if s["en"]]
    rng = random.Random(args.seed)
    sample = rng.sample(segs, min(args.n, len(segs)))
    sample.sort(key=lambda s: s["t"])

    print(f"# population: {len(segs)} speech segments | "
          f"sample: {len(sample)} | seed: {args.seed}")
    for i, s in enumerate(sample, 1):
        print(f"\n#{i:02d} [{s['t']}] {s['role']}")
        print(f"   bn: {s['bn'][:130]}")
        print(f"   en: {s['en'][:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
