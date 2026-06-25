"""Unit tests for the bin resolver's TOKEN-TYPE GATE (combine.resolve_bin).

The one test that matters: a NUMBER disagreement must be FLAGGED, never
confidence-voted — even when the WRONG candidate is the more confident one
(ASR confidence is weakly correlated with correctness; that's the whole risk).

    uv run --no-sync python scripts/test_combine.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from asr_pipeline.combine import (  # noqa: E402
    is_numberish, resolve_bin, token_type,
)


def main() -> int:
    fails = 0

    def check(cond, ok, bad):
        nonlocal fails
        if cond:
            print(f"✓ {ok}")
        else:
            print(f"✗ {bad}")
            fails += 1

    # ── number detection: high recall, incl. irregular compounds ──
    check(is_numberish("সাঁইত্রিশ"), "numberish: compound সাঁইত্রিশ (37) via ত্রিশ",
          "missed সাঁইত্রিশ")
    check(is_numberish("সাড়ে") and is_numberish("আড়াইশো") and is_numberish("১৩"),
          "numberish: fractions/scales/digits caught", "missed a number form")
    check(not is_numberish("পরিবার") and not is_numberish("আমার"),
          "numberish: ordinary words not caught", "false-fired on a normal word")
    check(token_type(["আমার", "পরিবার"]) == "prose",
          "token_type: ordinary words → prose", "misclassified prose")

    # ── THE CRITICAL GATE: number disagreement is FLAGGED, not voted, even
    #    when the WRONG candidate is far more confident ──
    per_system = [
        ("whisper", [("সাঁইত্রিশ", 0.40)]),   # correct (37) but LOW confidence
        ("titu",    [("সাড়ে", 0.95)]),        # WRONG but HIGH confidence
    ]
    d = resolve_bin(per_system, base_system="whisper")
    check(d.needs_review and d.token_type == "number",
          "GATE: number disagreement → FLAGGED for review",
          f"number disagreement NOT flagged: {d}")
    check(d.token == "সাঁইত্রিশ",
          "GATE: flagged bin keeps the BASE (whisper) token, not the "
          "confident-wrong one", f"flagged bin took {d.token!r} (wrong-confident leak!)")
    check("সাড়ে" in d.candidates and "সাঁইত্রিশ" in d.candidates,
          "GATE: all candidates logged for the worklist", "candidates not logged")

    # ── number AGREEMENT → accepted, no review ──
    d2 = resolve_bin([("whisper", [("সাঁইত্রিশ", 0.5)]),
                      ("titu", [("সাঁইত্রিশ", 0.6)])], base_system="whisper")
    check(not d2.needs_review and d2.reason == "number-agree",
          "number agreement → accepted, no flag", f"agree mishandled: {d2}")

    # ── PROSE bin → confidence-weighted vote IS allowed (standard CNC) ──
    d3 = resolve_bin([("whisper", [("করে", 0.6)]), ("titu", [("করি", 0.9)])],
                     base_system="whisper", weights={"whisper": 1.0, "titu": 1.0})
    check(not d3.needs_review and d3.token_type == "prose"
          and d3.token == "করি",
          "prose disagreement → confidence-voted (higher conf wins)",
          f"prose vote mishandled: {d3}")

    # ── one-sided number (titu has it, whisper deleted) → still flagged ──
    d4 = resolve_bin([("whisper", []), ("titu", [("তেরো", 0.8)])],
                     base_system="whisper")
    check(d4.needs_review and d4.token_type == "number",
          "one-sided number → flagged (don't silently insert a survey number)",
          f"one-sided number mishandled: {d4}")

    print("\n" + ("ALL PASS ✓" if fails == 0 else f"{fails} FAILURE(S) ✗"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
