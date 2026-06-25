"""Snap audit — the faithfulness metric for the survey-vocabulary anchor.

A "snap" is when the fused output contains a vocabulary / proper-noun term
that appears in NEITHER raw witness (titu LITERAL, Whisper FLUENT) — i.e.
the model inserted it from the injected anchor, not from the audio. On
sensitive survey data that's a data-integrity defect, so this count (not an
eyeball) is what gates whether the proper-noun list ships.

Reads a `.reasoning.txt` produced by a vocab-ON run (it carries LITERAL /
FLUENT / bn / en per segment) and the vocabulary files it was given.

Usage:
    uv run --no-sync python scripts/snap_audit.py \
        outputs/v3_test/15_part1_mid5min.reasoning.txt \
        --wiki data/survey_wiki
    # add --proper-nouns to also audit the proper-noun list
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BENGALI = re.compile(r"[ঀ-৿]{2,}(?:\s+[ঀ-৿]{2,})?")
HEAD = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s+([A-Z_]+)")


def parse_reasoning(path: Path) -> list[dict]:
    segs, cur = [], None
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = HEAD.match(ln)
        if m:
            cur = {"t": m.group(1), "role": m.group(2),
                   "literal": "", "fluent": "", "bn": "", "en": ""}
            segs.append(cur)
            continue
        if cur is None:
            continue
        for key, pre in (("literal", "LITERAL"), ("fluent", "FLUENT"),
                         ("bn", "bn"), ("en", "en")):
            if ln.strip().startswith(pre):
                cur[key] = ln.split(":", 1)[1].strip() if ":" in ln else ""
    return segs


def load_terms(wiki: Path, proper_nouns: bool) -> set[str]:
    terms: set[str] = set()
    files = ["domain_vocabulary.md"] + (["proper_nouns.txt"]
                                        if proper_nouns else [])
    for fn in files:
        p = wiki / fn
        if not p.exists():
            continue
        for tok in BENGALI.findall(p.read_text(encoding="utf-8")):
            tok = tok.strip()
            if len(tok) >= 3:                # skip 1-2 char particles
                terms.add(tok)
    return terms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reasoning", type=Path)
    ap.add_argument("--wiki", type=Path, default=Path("data/survey_wiki"))
    ap.add_argument("--proper-nouns", action="store_true")
    args = ap.parse_args()

    segs = parse_reasoning(args.reasoning)
    terms = load_terms(args.wiki, args.proper_nouns)
    if not segs or not terms:
        print("nothing to audit (no segments or no terms)", file=sys.stderr)
        return 1

    snaps, checked = [], 0
    for s in segs:
        out = f"{s['bn']} {s['en']}"
        witness = f"{s['literal']} {s['fluent']}"
        for term in terms:
            if term in out:
                checked += 1
                if term not in witness:       # in output, in no witness
                    snaps.append((s["t"], s["role"], term, s))

    n = len(segs)
    print(f"# snap audit: {n} segments | vocab terms: {len(terms)} | "
          f"proper_nouns: {args.proper_nouns}")
    print(f"# term-in-output occurrences checked: {checked}")
    print(f"# SNAPS (term in output but in NEITHER witness): {len(snaps)} "
          f"({100*len(snaps)/max(1,checked):.1f}% of occurrences)\n")
    for t, role, term, s in snaps:
        print(f"[{t}] {role}  ⚠ snap term: {term!r}")
        print(f"   bn      : {s['bn'][:160]}")
        print(f"   LITERAL : {s['literal'][:160]}")
        print(f"   FLUENT  : {s['fluent'][:160]}\n")
    print("VERDICT:", "CLEAN — anchor stayed faithful" if not snaps
          else f"{len(snaps)} snaps — inspect before trusting this arm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
