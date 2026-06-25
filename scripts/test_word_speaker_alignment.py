"""Unit test for word-level speaker attribution (Step 3, WhisperX-style).

Pure-python — no GPU, no NeMo, no audio. Fabricates one ASR chunk whose
words straddle a speaker change and checks that align_segments() splits it
into speaker-pure runs, tags each word, and handles the no-words fallback.

    uv run --no-sync python scripts/test_word_speaker_alignment.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from asr_pipeline.alignment import (  # noqa: E402
    align_segments,
    map_speakers_to_roles,
    merge_consecutive_segments,
    speaker_for_word,
)
from asr_pipeline.models import (  # noqa: E402
    AlignedSegment,
    ASRSegment,
    DiarizationResult,
    SpeakerSegment,
    WordSegment,
)


def _w(word: str, s: float, e: float) -> WordSegment:
    return WordSegment(word=word, start_s=s, end_s=e)


def main() -> int:
    fails = 0

    # Diarization: SPEAKER_00 owns 0–5 s (enumerator asks),
    # SPEAKER_01 owns 5–10 s (respondent answers).
    diar = DiarizationResult(
        num_speakers=2,
        segments=[
            SpeakerSegment(speaker_id="SPEAKER_00", start_s=0.0, end_s=5.0),
            SpeakerSegment(speaker_id="SPEAKER_01", start_s=5.0, end_s=10.0),
        ],
    )

    # One chunk, words crossing the 5 s boundary.
    seg = ASRSegment(
        segment_id=0, start_s=0.0, end_s=10.0, language="ben", confidence=0.0,
        text="ক খ গ ঘ ঙ",
        words=[
            _w("ক", 0.5, 1.0),   # SPEAKER_00
            _w("খ", 1.5, 2.0),   # SPEAKER_00
            _w("গ", 3.0, 3.5),   # SPEAKER_00
            _w("ঘ", 6.0, 6.5),   # SPEAKER_01
            _w("ঙ", 7.0, 7.5),   # SPEAKER_01
        ],
    )

    # ── word-level attribution ──
    assert speaker_for_word(seg.words[0], diar) == "SPEAKER_00"
    assert speaker_for_word(seg.words[3], diar) == "SPEAKER_01"
    # a word in a gap (10.5–11 s, past all turns) snaps to nearest (01)
    assert speaker_for_word(_w("z", 10.5, 11.0), diar) == "SPEAKER_01"
    print("✓ speaker_for_word: overlap + nearest-fallback correct")

    # ── splitting one mixed chunk into two speaker-pure segments ──
    aligned = align_segments([seg], diar)
    if len(aligned) != 2:
        print(f"✗ expected 2 speaker-pure segments, got {len(aligned)}")
        fails += 1
    else:
        a, b = aligned
        ok = (
            a.speaker_id == "SPEAKER_00" and a.raw_text == "ক খ গ"
            and b.speaker_id == "SPEAKER_01" and b.raw_text == "ঘ ঙ"
            and len(a.words) == 3 and len(b.words) == 2
            and a.start_s == 0.5 and b.end_s == 7.5
            and a.segment_id != b.segment_id
        )
        print(f"  seg A: {a.speaker_id} [{a.start_s}-{a.end_s}] {a.raw_text!r} "
              f"({len(a.words)} words)")
        print(f"  seg B: {b.speaker_id} [{b.start_s}-{b.end_s}] {b.raw_text!r} "
              f"({len(b.words)} words)")
        if ok:
            print("✓ align_segments: mixed chunk → 2 speaker-pure segments")
        else:
            print("✗ split produced wrong boundaries/text/ids")
            fails += 1

    # every word carries a speaker now
    if all(w.speaker_id for s in aligned for w in s.words):
        print("✓ every word tagged with a speaker")
    else:
        print("✗ some words left untagged")
        fails += 1

    # ── pure chunk (single speaker) stays one row, keeps original text ──
    pure = ASRSegment(
        segment_id=0, start_s=0.0, end_s=4.0, language="ben", confidence=0.0,
        text="ক খ গ",
        words=[_w("ক", 0.5, 1.0), _w("খ", 2.0, 2.5), _w("গ", 3.0, 3.5)],
    )
    pa = align_segments([pure], diar)
    if len(pa) == 1 and pa[0].raw_text == "ক খ গ" and pa[0].speaker_id == "SPEAKER_00":
        print("✓ single-speaker chunk → 1 row, original text preserved")
    else:
        print(f"✗ pure chunk mishandled: {len(pa)} rows")
        fails += 1

    # ── no-words fallback (transcription fallback path) ──
    nw = ASRSegment(
        segment_id=0, start_s=6.0, end_s=9.0, language="ben",
        confidence=0.0, text="কিছু", words=[],
    )
    nwa = align_segments([nw], diar)
    if len(nwa) == 1 and nwa[0].speaker_id == "SPEAKER_01":
        print("✓ no-words segment → whole-segment max-overlap fallback")
    else:
        print("✗ no-words fallback mishandled")
        fails += 1

    # ── merge across chunks: same speaker tail+head should rejoin ──
    c1 = ASRSegment(
        segment_id=0, start_s=0.0, end_s=3.0, language="ben", confidence=0.0,
        text="ক খ", words=[_w("ক", 0.5, 1.0), _w("খ", 2.0, 2.5)],
    )
    c2 = ASRSegment(
        segment_id=1, start_s=3.0, end_s=4.5, language="ben", confidence=0.0,
        text="গ", words=[_w("গ", 2.8, 3.3)],  # 0.3 s after c1's last word
    )
    merged = merge_consecutive_segments(align_segments([c1, c2], diar))
    if len(merged) == 1 and merged[0].speaker_id == "SPEAKER_00" \
            and len(merged[0].words) == 3:
        print("✓ same-speaker runs across chunks merge back together")
    else:
        print(f"✗ cross-chunk merge mishandled: {len(merged)} rows")
        fails += 1

    # ── Step 4: cluster → role mapping ──
    def _aseg(sid, spk, s, e, text):
        return AlignedSegment(
            segment_id=sid, start_s=s, end_s=e, speaker_id=spk,
            language="ben", raw_text=text, confidence=0.0,
            words=[_w(text, s, e)],
        )

    # SPEAKER_00 reads the consent script early (keyword 'consent'),
    # SPEAKER_01 answers. Keyword signal → 00 = ENUMERATOR.
    convo = [
        _aseg(0, "SPEAKER_00", 5.0, 9.0, "এই interview consent আপনি"),
        _aseg(1, "SPEAKER_01", 9.0, 30.0, "আমার পরিবার বন্যা"),
        _aseg(2, "SPEAKER_00", 30.0, 33.0, "আপনার বয়স কত"),
        _aseg(3, "SPEAKER_01", 33.0, 80.0, "আমি কৃষি কাজ করি অনেক বছর"),
    ]
    roled = map_speakers_to_roles(convo, consent_keywords=["consent", "interview"])
    rolemap = {s.segment_id: s.speaker_id for s in roled}
    if rolemap == {0: "ENUMERATOR", 1: "RESPONDENT",
                   2: "ENUMERATOR", 3: "RESPONDENT"}:
        print("✓ role mapping (keywords): consent-reader → ENUMERATOR")
    else:
        print(f"✗ role mapping (keywords) wrong: {rolemap}")
        fails += 1

    # words inherit the role too
    if all(w.speaker_id in ("ENUMERATOR", "RESPONDENT")
           for s in roled for w in s.words):
        print("✓ words re-tagged with role")
    else:
        print("✗ words not re-tagged with role")
        fails += 1

    # no keywords → fall back to most-speech-in-anchor-window.
    # Make SPEAKER_00 dominate the early window so it's ENUMERATOR.
    speech_convo = [
        _aseg(0, "SPEAKER_00", 1.0, 200.0, "ক"),    # 199 s early
        _aseg(1, "SPEAKER_01", 200.0, 230.0, "খ"),  # 30 s
    ]
    sr = map_speakers_to_roles(speech_convo, consent_keywords=[])
    srmap = {s.segment_id: s.speaker_id for s in sr}
    if srmap == {0: "ENUMERATOR", 1: "RESPONDENT"}:
        print("✓ role mapping (no keywords): most-speech → ENUMERATOR")
    else:
        print(f"✗ speech-fallback role mapping wrong: {srmap}")
        fails += 1

    # single cluster → all RESPONDENT (honest degradation)
    one = [_aseg(0, "SPEAKER_00", 1.0, 50.0, "ক খ গ")]
    om = map_speakers_to_roles(one)
    if len(om) == 1 and om[0].speaker_id == "RESPONDENT":
        print("✓ single cluster → RESPONDENT (no fabricated alternation)")
    else:
        print(f"✗ single-cluster mapping wrong: {[s.speaker_id for s in om]}")
        fails += 1

    # ── LLM cluster→role classifier (stubbed generate, no real model) ──
    try:
        from asr_pipeline.fusion import FusionTranslator
        ft = FusionTranslator.__new__(FusionTranslator)  # no heavy __init__
        ft._cluster_role_map = {}
        clusters_in = [
            _aseg(0, "SPEAKER_00", 5.0, 9.0, "এটা এক নম্বর ওয়ার্ড"),     # answer
            _aseg(1, "SPEAKER_01", 9.0, 12.0, "ওয়ার্ডের নাম কত ভাই"),     # question
            _aseg(2, "SPEAKER_00", 12.0, 40.0, "আমার পরিবারে ছয় জন"),     # answer
            _aseg(3, "SPEAKER_01", 40.0, 43.0, "আপনার বয়স কত"),           # question
        ]
        # Stub LLM: "B" → SPEAKER_01 (the question-asker) is ENUMERATOR.
        m = ft._decide_roles(clusters_in, lambda p: "B")
        if m == {"SPEAKER_00": "RESPONDENT", "SPEAKER_01": "ENUMERATOR"}:
            print("✓ LLM classifier: picks the named cluster as ENUMERATOR")
        else:
            print(f"✗ LLM classifier mapping wrong: {m}")
            fails += 1

        # Unparseable reply → heuristic fallback (no crash, returns a map)
        mf = ft._decide_roles(clusters_in, lambda p: "I'm not sure")
        if set(mf.values()) <= {"ENUMERATOR", "RESPONDENT"} and len(mf) == 2:
            print("✓ LLM classifier: garbage reply → heuristic fallback")
        else:
            print(f"✗ fallback mishandled: {mf}")
            fails += 1
    except Exception as ex:
        print(f"✗ LLM classifier test errored: {type(ex).__name__}: {ex}")
        fails += 1

    print("\n" + ("ALL PASS ✓" if fails == 0 else f"{fails} FAILURE(S) ✗"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
