"""Print a clean, readable transcript from a pipeline .txt output.

Shows just [time] ROLE / bn / en per segment — drops the header, the
[bn-raw] witness line, and the [SILENCE]/[MUSIC] markers.

Usage:
    uv run --no-sync python scripts/print_transcript.py \
        outputs/v3_roles/15_part1_mid5min.txt
    # English only / Bengali only:
    python scripts/print_transcript.py <file> --en
    python scripts/print_transcript.py <file> --bn
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEAD = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s+([A-Z_]+):")


def parse(path: Path) -> list[dict]:
    segs, cur = [], None
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = HEAD.match(ln)
        if m:
            cur = {"t": m.group(1), "role": m.group(2), "bn": "", "en": ""}
            segs.append(cur)
            continue
        if cur is None:
            continue
        if ln.startswith("[bn]"):
            cur["bn"] = ln[len("[bn]"):].strip()
        elif ln.startswith("[en]"):
            cur["en"] = ln[len("[en]"):].strip()
    return segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--bn", action="store_true", help="Bengali only")
    ap.add_argument("--en", action="store_true", help="English only")
    args = ap.parse_args()

    if not args.transcript.exists():
        print(f"not found: {args.transcript}", file=sys.stderr)
        return 1

    show_bn = args.bn or not args.en
    show_en = args.en or not args.bn

    segs = parse(args.transcript)
    for s in segs:
        print(f"\n[{s['t']}] {s['role']}")
        if show_bn and s["bn"]:
            print(f"  {s['bn']}")
        if show_en and s["en"]:
            print(f"  {s['en']}")
    print(f"\n— {len(segs)} segments —")
    return 0


if __name__ == "__main__":
    sys.exit(main())
