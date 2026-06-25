"""Unit tests for the GER guard (constrained closest-to-N-best + agreement).

Pure-python — no model, no GPU. Validates the grapheme distance, the number
extraction, the closest-to-N-best snap, and the agreement / number flags.

    uv run --no-sync python scripts/test_ger_guard.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from asr_pipeline.ger_guard import (  # noqa: E402
    assess, grapheme_clusters, ndist, numbers,
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

    # ── grapheme clustering: combining vowel signs attach to their base ──
    # জ + া (vowel sign) = one cluster; "জা" → 1 cluster, not 2.
    check(len(grapheme_clusters("জা")) == 1,
          "grapheme: জ+া is one cluster", "grapheme clustering wrong")
    check(len(grapheme_clusters("জায়গা")) < len("জায়গা"),
          "grapheme: combining marks reduce cluster count", "no clustering")

    # ── ndist: identical content = 0 (punctuation/space ignored) ──
    check(ndist("এটা এক নম্বর ওয়ার্ড", "এটা এক নম্বর ওয়ার্ড।") == 0.0,
          "ndist: punctuation/danda ignored → 0", "punct inflates distance")
    check(ndist("পানি আসে", "পানি আসে") == 0.0,
          "ndist: identical → 0", "identical not 0")
    check(ndist("পানি", "গরু") > 0.5,
          "ndist: unrelated words → large", "unrelated not large")

    # ── numbers: Bengali + ASCII digits, value-normalised ──
    check(numbers("আমার ১৩টা ছাগল") == {"13"},
          "numbers: Bengali ১৩ → '13'", f"got {numbers('আমার ১৩টা ছাগল')}")
    check(numbers("ward 1 আর ২০২২") == {"1", "2022"},
          "numbers: mixed scripts", f"got {numbers('ward 1 আর ২০২২')}")

    # ── snap target = witness NEAREST [bn] (NOT medoid). titu is uniquely
    # close → must snap to titu, proving the correlated-Whisper bias is gone ──
    titu = "বাইশ বন্যায় তেরোটা ছাগল ছিল"
    whyps = ["বাইশ বন্যায় তেরো ছাগল",       # 3 correlated Whisper beams,
             "বাইশ বন্যায় তেরোটা ছাগল",      # all slightly different from titu
             "বাইশ বন্যায় তেরো ছাগল ছিল"]
    # a [bn] far from everything but closest to titu
    halluc_near_titu = "বাইশ বন্যায় তেরোটা ছাগল ছিল আমার অনেক"
    v = assess(halluc_near_titu, "en", titu, whyps, snap=True, snap_tau=0.10)
    check(v.snapped and v.bn == titu,
          "snap: target is the NEAREST witness (titu), not a Whisper medoid",
          f"snapped to {v.bn!r} not titu (min_dist={v.min_dist:.2f})")
    check(v.en == "" and any("snapped" in r for r in v.reasons),
          "snap: stale [en] blanked + flagged", f"en={v.en!r} reasons={v.reasons}")

    # ── snap OFF by default → observe-mode 'would-snap', text untouched ──
    vo = assess(halluc_near_titu, "en", titu, whyps, snap_tau=0.10)  # snap=False
    check(not vo.snapped and vo.bn == halluc_near_titu
          and any(r.startswith("would-snap") for r in vo.reasons),
          "observe: snap off → would-snap hint, [bn] untouched",
          f"observe-mode wrong: snapped={vo.snapped} reasons={vo.reasons}")

    # ── no-snap: a legit reconciliation (close to witnesses) is kept ──
    recon = "বাইশ বন্যায় তেরোটা ছাগল ছিল।"
    v2 = assess(recon, "en", titu, whyps, snap=True)
    check(not v2.snapped and v2.bn == recon,
          "no-snap: reconciliation close to witnesses is kept",
          f"good reconciliation wrongly snapped (min_dist={v2.min_dist:.2f})")

    # ── fabricated number ABSTAIN: witness ১৩, [bn] ১৫ → ১৫ → [অস্পষ্ট] ──
    v3 = assess("আমার ১৫টা ছাগল ছিল", "I had 15 goats",
                "আমার ১৩টা ছাগল ছিল", ["আমার ১৩ টা ছাগল"])
    check("[অস্পষ্ট]" in v3.bn and "১৫" not in v3.bn,
          "number-abstain: fabricated ১৫ blanked to [অস্পষ্ট] in bn",
          f"not abstained: {v3.bn}")
    check("[unclear]" in v3.en and "15" not in v3.en,
          "number-abstain: fabricated 15 blanked to [unclear] in en",
          f"en not abstained: {v3.en}")

    # word→digit is NOT a fabrication: witness has no digits → kept
    v4 = assess("আমার বয়স ৩৫", "age 35", "আমার বয়স পঁয়ত্রিশ",
                ["বয়স পঁয়ত্রিশ"])
    check("৩৫" in v4.bn and not any(r.startswith("number-abstain")
                                     for r in v4.reasons),
          "number: word→digit (no witness digit) kept, not abstained",
          f"word→digit wrongly abstained: {v4.bn} {v4.reasons}")

    # ── agreement: titu vs whisper-top far apart → witness-disagree flag ──
    v5 = assess("পানি আসে নদী থেকে", "en", "পানি আসে নদী থেকে",
                ["গরু মাঠে ঘাস খায় বিকেলে"])
    check(any(r.startswith("witness-disagree") for r in v5.reasons),
          "agreement: divergent witnesses flagged",
          f"disagreement not flagged: {v5.reasons}")

    # ── empty / single-witness edge cases don't crash ──
    check(assess("", "", titu, whyps).bn == ""
          and assess("ক", "", "", []).bn == "ক",
          "edge: empty [bn] / no witnesses handled", "edge case crashed")

    print("\n" + ("ALL PASS ✓" if fails == 0 else f"{fails} FAILURE(S) ✗"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
