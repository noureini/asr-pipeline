"""Build a human-QA worklist from an ALREADY-FINISHED transcript.

For runs made before in-pipeline review-flagging existed. Flags segments
from signals present in the .txt: a number to verify, an unclear marker,
heavy repetition, or a near-empty translation. (Witness-disagreement is
not available retroactively — it isn't stored in the transcript.)

Usage:
    uv run python scripts/make_review.py <transcript.txt>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

H = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s+([A-Z_]+):")


def repscore(s: str) -> int:
    w = s.split()
    best = run = 1
    for i in range(1, len(w)):
        run = run + 1 if w[i] == w[i - 1] else 1
        best = max(best, run)
    return best


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: make_review.py <transcript.txt>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    cur = None
    segs = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = H.match(ln)
        if m:
            cur = {"t": m.group(1), "role": m.group(2),
                   "raw": "", "bn": "", "en": ""}
            segs.append(cur)
            continue
        if cur is None:
            continue
        if ln.startswith("[bn-raw]"):
            cur["raw"] = ln[8:].strip()
        elif ln.startswith("[bn]"):
            cur["bn"] = ln[4:].strip()
        elif ln.startswith("[en]"):
            cur["en"] = ln[4:].strip()

    flagged = []
    for s in segs:
        reasons = []
        if re.search(r"[0-9০-৯]", s["bn"] + " " + s["en"]):
            reasons.append("number")
        if "[অস্পষ্ট]" in s["bn"] or "[unclear]" in s["en"].lower():
            reasons.append("unclear")
        if repscore(s["bn"]) >= 3 or repscore(s["en"]) >= 3:
            reasons.append("repetition")
        if s["bn"] and len(s["en"].strip()) < 3:
            reasons.append("empty-output")
        if reasons:
            flagged.append((s, reasons))

    out = path.with_suffix(".review.txt")
    L = [
        "HUMAN-QA WORKLIST (retroactive — no witness-disagreement signal)",
        f"{len(flagged)} of {len(segs)} segments flagged "
        f"({100 * len(flagged) // max(1, len(segs))}%)",
        "=" * 70, "",
    ]
    for s, r in flagged:
        L.append(f"[{s['t']}] {s['role']}  ⚠ {', '.join(r)}")
        if s["raw"]:
            L.append(f"  raw: {s['raw']}")
        L.append(f"  bn : {s['bn']}")
        L.append(f"  en : {s['en']}")
        L.append("")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"{len(flagged)}/{len(segs)} flagged → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
