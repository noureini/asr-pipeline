"""Compare two ASR raw transcripts segment-by-segment.

Parses two .raw.txt files (the formatter's [HH:MM:SS] SPEAKER / [bn]
format) and prints the first N segments side by side, aligned by index.
Use it to eyeball titu vs IndicConformer on the same audio.

Usage:
    uv run python scripts/compare_asr.py \
        --a outputs/v1_deliverable/15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635/15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.raw.txt \
        --a-label titu \
        --b outputs/asr_ab/15_indic_conformer/15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.raw.txt \
        --b-label IndicConformer \
        --n 15
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+(\S+):\s*\n\[bn\]\s+(.+?)"
        r"(?=\n\s*\n|\n\[|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    segs = []
    for m in pattern.finditer(text):
        hh, mm, ss, spk, body = m.groups()
        segs.append({
            "ts": f"{hh}:{mm}:{ss}",
            "spk": spk,
            "text": " ".join(body.split()),
        })
    return segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b", type=Path)
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--c", type=Path)
    ap.add_argument("--c-label", default="C")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()

    a = parse(args.a)
    b = parse(args.b) if args.b else None
    c = parse(args.c) if args.c else None

    n = min(args.n, len(a))
    parts = [f"{args.a_label}: {len(a)} segs"]
    if b is not None:
        parts.append(f"{args.b_label}: {len(b)} segs")
    if c is not None:
        parts.append(f"{args.c_label}: {len(c)} segs")
    print(" | ".join(parts) + "\n")
    print("=" * 78)

    for i in range(n):
        sa = a[i]
        print(f"\n# Segment {i+1}")
        print(f"[{sa['ts']}] {sa['spk']}")
        print(f"  [{args.a_label}] {sa['text']}")
        if b and i < len(b):
            print(f"  [{args.b_label}] {b[i]['text']}")
        if c and i < len(c):
            print(f"  [{args.c_label}] {c[i]['text']}")
        print("-" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
