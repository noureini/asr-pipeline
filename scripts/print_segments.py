"""Print N speaker segments from a finished transcript for inspection.

Parses the [HH:MM:SS] ROLE: / [bn-raw] / [bn] / [en] blocks (skipping
SILENCE/MUSIC/NOISE markers) so you can eyeball roles + translation.

Usage:
    uv run python scripts/print_segments.py <transcript.txt>
    uv run python scripts/print_segments.py <transcript.txt> --start 10 --count 8
    uv run python scripts/print_segments.py <transcript.txt> --role ENUMERATOR
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEAD = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s+([A-Z_0-9]+):\s*$")


def parse(path: Path) -> list[dict]:
    segs: list[dict] = []
    cur: dict | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEAD.match(line)
        if m:
            cur = {"time": m.group(1), "role": m.group(2),
                   "bn_raw": "", "bn": "", "en": ""}
            segs.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("[bn-raw]"):
            cur["bn_raw"] = line[len("[bn-raw]"):].strip()
        elif line.startswith("[bn]"):
            cur["bn"] = line[len("[bn]"):].strip()
        elif line.startswith("[en]"):
            cur["en"] = line[len("[en]"):].strip()
    return segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--start", type=int, default=0,
                    help="first segment index (0-based)")
    ap.add_argument("--count", type=int, default=10,
                    help="how many segments to print")
    ap.add_argument("--role", default=None,
                    help="only show this role (e.g. ENUMERATOR)")
    args = ap.parse_args()

    if not args.transcript.exists():
        print(f"not found: {args.transcript}", file=sys.stderr)
        return 1

    segs = parse(args.transcript)
    if args.role:
        segs = [s for s in segs if s["role"] == args.role]
    sel = segs[args.start:args.start + args.count]

    print(f"=== {len(segs)} segments total; showing "
          f"{args.start}..{args.start + len(sel) - 1} ===\n")
    for i, s in enumerate(sel, args.start):
        print(f"[#{i}] [{s['time']}] {s['role']}")
        if s["bn_raw"]:
            print(f"  bn-raw: {s['bn_raw']}")
        print(f"  bn    : {s['bn']}")
        print(f"  en    : {s['en']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
