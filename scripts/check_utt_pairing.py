"""Diagnose wav<->gt_text pairing in the longform eval jsonls.

The qualitative ASR run surfaced that titu (a known-good ASR) transcribes
entirely different fluent Bengali than the GT claims for 07_Shatkhira
utterances. Two possible breakages:

  (A) wav extracted at WRONG offsets — actual wav duration won't match
      (end - start) from the jsonl.
  (B) gt_text assigned to the WRONG wav — durations match, but content
      doesn't (the text belongs to a different segment).

This checks (A) directly (cheap, no model) and sets up (B) by printing
jsonl duration vs actual wav duration per utterance. If durations match
but content was wrong in the qualitative run, it's (B).

Usage:
    uv run python scripts/check_utt_pairing.py --n 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path("data/w2v_train_longform")
JSONLS = [
    ("07_Shatkhira",
     DATA_DIR / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl"),
    ("15_Sunamganj",
     DATA_DIR / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.longform.jsonl"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    import torchaudio

    for group, path in JSONLS:
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f]
        n = min(args.n, len(recs))
        step = max(1, len(recs) // n)
        sample = recs[::step][:n]
        print(f"\n=== {group} ({len(recs)} utts) ===")
        print(f"{'idx':>5} {'jsonl_dur':>9} {'wav_dur':>8} {'match':>6}  "
              f"wav_exists  gt_text[:40]")
        mism = 0
        for r in sample:
            jdur = r.get("end", 0) - r.get("start", 0)
            wp = Path(r["wav"])
            exists = wp.exists()
            wdur = float("nan")
            if exists:
                try:
                    info = torchaudio.info(str(wp))
                    wdur = info.num_frames / info.sample_rate
                except Exception:
                    pass
            ok = (exists and abs(wdur - jdur) < 0.3)
            if not ok:
                mism += 1
            gt = (r.get("gt_text", "") or "")[:40]
            print(f"{r.get('idx','?'):>5} {jdur:>9.2f} {wdur:>8.2f} "
                  f"{'OK' if ok else 'BAD':>6}  {str(exists):>10}  {gt}")
        print(f"  duration-mismatch (or missing) rows: {mism}/{n}")
        # File-level sanity: do total wav durations roughly sum to the
        # interview length implied by the max end-time?
        max_end = max((r.get("end", 0) for r in recs), default=0)
        print(f"  max end-time in jsonl: {max_end/60:.1f} min")
    print("\nReading: if wav_dur matches jsonl_dur (mostly OK) but the "
          "qualitative run showed wrong content, the breakage is (B) "
          "text<->wav mispairing, not offset extraction. If durations "
          "are BAD, the wavs were cut at wrong offsets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
