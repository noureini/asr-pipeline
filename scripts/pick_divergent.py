"""Pick the N most-divergent segments from a multi-model A/B report.

Parses a side-by-side markdown report with ANY number of models and
ranks segments by pairwise divergence in their corrected Bengali [bn]
AND English [en] outputs. Picks the top-N segments where models disagree
most — those are where the quality differences actually show up.

Works on both quant_ab_test.md (2 models) and multi_model_ab_test.md
(N models) report formats.

Usage:
    uv run python scripts/pick_divergent.py \
        --input outputs/multi_model_ab_test.md \
        --n 4
"""

from __future__ import annotations

import argparse
import re
import sys
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path


# Match per-model blocks: "### LABEL\n**[bn]** ...\n**[en]** ..."
_MODEL_BLOCK = re.compile(
    r"###\s+(\S+)\s*\n+"
    r"\*\*\[bn\]\*\*\s+(.+?)\n+"
    r"\*\*\[en\]\*\*\s+(.+?)(?=\n+###|\n+---|\n+##|\Z)",
    re.DOTALL,
)

# Match a Segment block header up to the next "---" / next "##"
_SEG_BLOCK = re.compile(
    r"##\s+Segment\s+(\d+)[^\n]*\n+"
    r"\*\*\[bn-raw\]\*\*\s+(.+?)\n+"
    r"(?P<body>.+?)(?=\n+---|\n+## Segment|\Z)",
    re.DOTALL,
)


def parse_report(text: str) -> list[dict]:
    """Extract segments with all per-model outputs.

    Returns list of dicts:
        {idx, raw, models: {label: {"bn": ..., "en": ...}}}
    """
    segs = []
    for m in _SEG_BLOCK.finditer(text):
        idx, raw, body = m.group(1), m.group(2), m.group("body")
        models: dict[str, dict] = {}
        for mb in _MODEL_BLOCK.finditer(body):
            label, bn, en = mb.groups()
            models[label] = {"bn": bn.strip(), "en": en.strip()}
        segs.append({
            "idx": int(idx),
            "raw": raw.strip(),
            "models": models,
        })
    return segs


def divergence(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def seg_score(seg: dict) -> tuple[float, float, float]:
    """Average pairwise divergence across all model pairs.
    Returns (combined, bn_avg, en_avg)."""
    labels = list(seg["models"].keys())
    if len(labels) < 2:
        return 0.0, 0.0, 0.0
    bn_divs, en_divs = [], []
    for a, b in combinations(labels, 2):
        bn_divs.append(divergence(seg["models"][a]["bn"],
                                  seg["models"][b]["bn"]))
        en_divs.append(divergence(seg["models"][a]["en"],
                                  seg["models"][b]["en"]))
    bn = sum(bn_divs) / len(bn_divs)
    en = sum(en_divs) / len(en_divs)
    # Bengali correction is where the substantive quality difference lives.
    return 0.7 * bn + 0.3 * en, bn, en


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--n", type=int, default=4,
                    help="Number of top-divergent segments to print")
    args = ap.parse_args()

    text = args.input.read_text(encoding="utf-8")
    segs = parse_report(text)
    if not segs:
        print(f"No segments parsed from {args.input}.", file=sys.stderr)
        print("First 500 chars:\n", text[:500], file=sys.stderr)
        return 1

    n_models = len(segs[0]["models"]) if segs else 0
    print(f"Parsed {len(segs)} segments × {n_models} models.\n")

    scored = sorted(segs, key=lambda s: seg_score(s)[0], reverse=True)
    top = scored[: args.n]

    print(f"Top {args.n} most-divergent segments "
          f"(pairwise across {n_models} models):")
    print("=" * 72)

    for rank, seg in enumerate(top, 1):
        combined, bn, en = seg_score(seg)
        print(f"\n### Rank {rank} — Segment {seg['idx']} "
              f"(combined={combined:.2f}, bn_avg={bn:.2f}, en_avg={en:.2f})")
        print(f"\n**[bn-raw]** {seg['raw']}")
        for label, out in seg["models"].items():
            print(f"\n**{label} [bn]** {out['bn']}")
            print(f"**{label} [en]** {out['en']}")
        print("\n" + "-" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
