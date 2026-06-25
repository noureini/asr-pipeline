"""Score a filled gold template — the integrity metrics that replace the whole
thread's eyeballing with ground truth.

Workflow:
  1. a --titu run writes <audio>.goldtemplate.tsv
  2. open it in a spreadsheet; for each segment listen to [start,end] and fill:
       gold_numbers  — the TRUE numbers/ages/counts spoken (space-separated,
                       in the form you hear, e.g.  সাঁইত্রিশ  ছয়  বা  ৩৭ ৬)
       gold_names    — the TRUE person/place names spoken
       gold_text     — (optional) the full true transcript, for prose CER
  3. uv run --no-sync python scripts/score_goldset.py <audio>.goldtemplate.tsv

Reports, per system (whisper / titu_*):
  • RECALL on the survey-critical tokens (numbers, names) — the metric that
    actually decides the witness and large-vs-fast (not prose cleanliness)
  • prose CER vs gold_text
Plus the gate analysis: for each gold high-value token — present in BOTH
systems (safe / agree), ONE only (witness adds value), or NEITHER (acoustic
floor); and the base-miss rate = numbers the prose base (whisper) dropped,
which is exactly what the flag-gate protects against.
"""
from __future__ import annotations

import csv
import re
import sys

sys.path.insert(0, "src")

from asr_pipeline.ger_guard import ndist  # noqa: E402
from asr_pipeline.ger_guard import numbers as _digits  # noqa: E402

_PUNCT = re.compile(r"[\s।,.;:!?\-—–…\"'`()\[\]{}/]+")


def _norm(s: str) -> str:
    return _PUNCT.sub("", s or "")


def _has(token: str, text: str) -> bool:
    """Is gold `token` captured in `text`? Normalised-substring match, plus a
    digit-VALUE match so a digit gold (৩৭) matches a digit system token."""
    t, x = _norm(token), _norm(text)
    if t and t in x:
        return True
    dt, dx = _digits(token), _digits(text)
    return bool(dt and (dt & dx))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: score_goldset.py <file>.goldtemplate.tsv", file=sys.stderr)
        return 2
    with open(sys.argv[1], encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        print("empty template", file=sys.stderr)
        return 1
    sys_cols = (["whisper"]
                + [c for c in rows[0] if c.startswith("titu_")])

    recall = {c: {"num": 0, "name": 0} for c in sys_cols}
    cer = {c: [] for c in sys_cols}
    g_num = g_name = labeled = 0
    both = one = neither = base_miss = witness_saves = 0

    for r in rows:
        gnums = (r.get("gold_numbers") or "").split()
        gnames = (r.get("gold_names") or "").split()
        gtext = (r.get("gold_text") or "").strip()
        if not (gnums or gnames or gtext):
            continue
        labeled += 1
        if gtext:
            for c in sys_cols:
                cer[c].append(ndist(r.get(c, ""), gtext))
        for tok in gnums:
            g_num += 1
            present = {c: _has(tok, r.get(c, "")) for c in sys_cols}
            for c in sys_cols:
                recall[c]["num"] += int(present[c])
            w = present.get("whisper", False)
            t = any(present[c] for c in sys_cols if c != "whisper")
            both += int(w and t)
            one += int(w ^ t)
            neither += int(not (w or t))
            if not w:
                base_miss += 1            # whisper dropped a real number
                if t:
                    witness_saves += 1    # ...and titu caught it
        for tok in gnames:
            g_name += 1
            for c in sys_cols:
                recall[c]["name"] += int(_has(tok, r.get(c, "")))

    print(f"GOLD SET: {labeled}/{len(rows)} segments labelled | "
          f"{g_num} gold numbers | {g_name} gold names\n")
    if g_num == 0 and g_name == 0:
        print("No gold_numbers/gold_names filled yet — fill them and re-run.")
        return 0

    print("per-system recall on survey-critical tokens (THE metric):")
    print(f"  {'system':<22} {'numbers':>12} {'names':>12} {'prose CER':>11}")
    for c in sys_cols:
        nr = recall[c]["num"] / g_num if g_num else 0.0
        ar = recall[c]["name"] / g_name if g_name else 0.0
        cc = (sum(cer[c]) / len(cer[c])) if cer[c] else float("nan")
        print(f"  {c:<22} {recall[c]['num']:>3}/{g_num} ({nr:4.0%}) "
              f"{recall[c]['name']:>3}/{g_name} ({ar:4.0%}) {cc:>11.3f}")

    print("\ngate analysis on gold NUMBERS:")
    print(f"  both systems have it (safe/agree) : {both}/{g_num}")
    print(f"  one system only (witness value)   : {one}/{g_num}")
    print(f"  neither (acoustic floor)          : {neither}/{g_num}")
    print(f"  base(whisper) MISSED a number     : {base_miss}/{g_num}  "
          f"← what the flag-gate protects")
    print(f"  ...of which titu (witness) caught : {witness_saves}/{base_miss or 1}"
          f"  ← concrete witness value")
    return 0


if __name__ == "__main__":
    sys.exit(main())
