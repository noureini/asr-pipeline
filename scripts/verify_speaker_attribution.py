"""Verify the ENUMERATOR / RESPONDENT speaker attribution produced
by assemble_transcript.py against the consent-script keywords from
the survey wiki.

The instrument's consent intro is read VERBATIM by the interviewer.
Even when translated on-the-fly to Bengali, certain English terms
(organization names, formal nouns, the word 'interview' itself,
numbers like '1 hour') survive as code-switch tokens in titu's
Bengali output. The speaker who hits the most of these keywords in
the first ~10 min should be the ENUMERATOR.

If the count disagrees with what assemble_transcript.py labelled
(via its anchor-window heuristic), we print a WARNING so the user
can spot-check. We do NOT auto-relabel — the heuristic is usually
right, and overriding it without human review risks silent
mislabels.

Usage:
  uv run python scripts/verify_speaker_attribution.py \\
      --stem 15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635 \\
      --titu-root outputs/real_eval_titu_excellent
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--titu-root", type=Path,
                    default=REPO / "outputs"
                            / "real_eval_titu_excellent")
    ap.add_argument("--wiki-dir", type=Path,
                    default=REPO / "data" / "survey_wiki")
    ap.add_argument("--anchor-window", type=float, default=600.0,
                    help="seconds from start to look in (consent "
                         "intro is typically the first 5-10 min)")
    args = ap.parse_args()

    keyfile = args.wiki_dir / "consent_keywords.txt"
    if not keyfile.exists():
        sys.exit(f"missing {keyfile} — run build_survey_wiki.py first")
    keywords = [k.strip().lower()
                for k in keyfile.read_text(
                    encoding="utf-8").splitlines() if k.strip()]
    print(f"loaded {len(keywords)} consent keywords")

    stem_dir = args.titu_root / args.stem
    chunks_files = sorted(stem_dir.glob("*.chunks.jsonl"))
    if not chunks_files:
        sys.exit(f"no *.chunks.jsonl in {stem_dir}")

    # Use first part only (consent is in part 1)
    first = chunks_files[0]
    partstem = first.name[: -len(".chunks.jsonl")]
    diar_file = stem_dir / f"{partstem}.diar.jsonl"
    if not diar_file.exists():
        sys.exit(f"missing {diar_file} — run diarize_parts.py first")
    print(f"analyzing first part: {partstem}")

    # Read titu chunks for first part
    titu_chunks: list[dict] = []
    for ln in first.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            titu_chunks.append(json.loads(ln))

    # Read diarization turns
    turns: list[dict] = []
    for ln in diar_file.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            turns.append(json.loads(ln))

    # Map each chunk to its dominant pyannote speaker (re-implement
    # the same dominant-overlap logic that assemble_transcript uses)
    def dominant_speaker(c_start, c_end):
        ov = defaultdict(float)
        for t in turns:
            o = min(c_end, t["end"]) - max(c_start, t["start"])
            if o > 0:
                ov[t["speaker"]] += o
        if not ov:
            return None
        return max(ov.items(), key=lambda x: x[1])[0]

    # Count keyword hits per pyannote speaker in the anchor window
    hits_by_spk: dict[str, int] = defaultdict(int)
    speech_by_spk: dict[str, float] = defaultdict(float)
    for c in titu_chunks:
        if c["start"] > args.anchor_window:
            continue
        spk = dominant_speaker(c["start"], c["end"])
        if spk is None:
            continue
        speech_by_spk[spk] += c["end"] - c["start"]
        text = c.get("text", "").lower()
        for k in keywords:
            if k in text:
                hits_by_spk[spk] += 1

    if not hits_by_spk:
        print("\nNO keyword hits found in the anchor window. Possible "
              "reasons: titu output too noisy, consent in a "
              "different part, or keywords need calibrating.")
        return

    print(f"\nAnchor window: first {args.anchor_window:.0f}s")
    print(f"Per-speaker stats in this window:")
    print(f"{'speaker':<12} {'speech_s':>10} "
          f"{'keyword_hits':>13} {'verdict':>20}")
    by_speech = sorted(speech_by_spk.items(),
                       key=lambda x: -x[1])
    enum_by_speech = by_speech[0][0] if by_speech else None
    by_hits = sorted(hits_by_spk.items(), key=lambda x: -x[1])
    enum_by_hits = by_hits[0][0] if by_hits else None
    for spk, sec in by_speech:
        verdict = []
        if spk == enum_by_speech:
            verdict.append("heuristic→ENUMERATOR")
        if spk == enum_by_hits:
            verdict.append("keywords→ENUMERATOR")
        print(f"{spk:<12} {sec:>10.1f} "
              f"{hits_by_spk.get(spk, 0):>13d} "
              f"{', '.join(verdict) or 'RESPONDENT/OTHER':>20}")

    if enum_by_speech == enum_by_hits:
        print(f"\n✓ AGREE: both heuristic and consent-keyword "
              f"matching identify {enum_by_speech} as ENUMERATOR")
    else:
        print(f"\n⚠ DISAGREE")
        print(f"  heuristic (most-speech-in-{args.anchor_window:.0f}s): "
              f"{enum_by_speech}")
        print(f"  consent keywords: {enum_by_hits}")
        print(f"  → spot-check the first ~10 min of audio to "
              f"verify which speaker reads the consent script")


if __name__ == "__main__":
    main()
