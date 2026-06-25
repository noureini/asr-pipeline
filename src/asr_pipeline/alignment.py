"""
Segment alignment module (WhisperX-style word-level speaker attribution).

The old approach assigned ONE speaker per ~30 s ASR chunk by maximum
overlap. When a chunk spans a speaker change (enumerator asks → respondent
answers), the whole chunk got the dominant speaker's label and the other
speaker's words were silently mis-attributed. That mixing is exactly what
the downstream LLM "role attribution" stage was patching over.

With titu's native CTC word timestamps we instead tag EACH WORD with the
diarization turn it overlaps (`assign_word_speakers`), then split the chunk
into speaker-pure runs at every speaker-change boundary. The result is
segments that each belong to a single speaker — no LLM band-aid needed.

Segments that carry no word timestamps (e.g. a transcription fallback)
degrade gracefully to the old whole-segment maximum-overlap heuristic.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from asr_pipeline.models import (
    AlignedSegment,
    ASRSegment,
    DiarizationResult,
    SpeakerSegment,
    WordSegment,
)

logger = logging.getLogger("asr_pipeline")

_FALLBACK_SPEAKER = "SPEAKER_00"
ENUMERATOR = "ENUMERATOR"
RESPONDENT = "RESPONDENT"


def compute_overlap(
    seg_start: float,
    seg_end: float,
    spk_start: float,
    spk_end: float,
) -> float:
    """Compute the temporal overlap in seconds between two intervals."""
    overlap_start = max(seg_start, spk_start)
    overlap_end = min(seg_end, spk_end)
    return max(0.0, overlap_end - overlap_start)


def find_best_speaker(
    asr_segment: ASRSegment,
    diarization: DiarizationResult,
) -> str:
    """
    Find the speaker with the most temporal overlap with an ASR segment.

    Uses a simple maximum-overlap heuristic: the speaker whose
    diarization segment has the largest intersection with the ASR
    segment's time span is assigned ownership.
    """
    best_speaker = "SPEAKER_00"
    best_overlap = 0.0

    for spk_seg in diarization.segments:
        overlap = compute_overlap(
            asr_segment.start_s,
            asr_segment.end_s,
            spk_seg.start_s,
            spk_seg.end_s,
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = spk_seg.speaker_id

    return best_speaker


def speaker_for_word(
    word: WordSegment,
    diarization: DiarizationResult,
) -> str:
    """
    Attribute a single word to a diarization speaker.

    Uses maximum temporal overlap between the word's [start, end] interval
    and each diarization turn. If the word falls in a diarization gap
    (no overlap with any turn — common at turn boundaries), it is assigned
    to the *nearest* turn by edge distance (WhisperX `fill_nearest`),
    so every word always gets a speaker.
    """
    best_speaker = _FALLBACK_SPEAKER
    best_overlap = 0.0

    for spk in diarization.segments:
        ov = compute_overlap(word.start_s, word.end_s, spk.start_s, spk.end_s)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = spk.speaker_id

    if best_overlap > 0.0:
        return best_speaker

    # No overlap: snap to the nearest turn by gap distance.
    nearest_speaker = _FALLBACK_SPEAKER
    nearest_gap = float("inf")
    mid = 0.5 * (word.start_s + word.end_s)
    for spk in diarization.segments:
        if mid < spk.start_s:
            gap = spk.start_s - mid
        elif mid > spk.end_s:
            gap = mid - spk.end_s
        else:
            gap = 0.0
        if gap < nearest_gap:
            nearest_gap = gap
            nearest_speaker = spk.speaker_id
    return nearest_speaker


def _split_into_speaker_runs(
    seg: ASRSegment,
    diarization: DiarizationResult,
    next_id: int,
) -> tuple[list[AlignedSegment], int]:
    """
    Tag every word in a segment with its speaker, then cut the segment into
    speaker-pure runs at each speaker-change boundary.

    Returns the produced AlignedSegments and the next free segment_id.
    A segment with no words degrades to whole-segment max-overlap (one row).
    """
    if not seg.words:
        speaker = find_best_speaker(seg, diarization)
        out = [AlignedSegment(
            segment_id=next_id,
            start_s=seg.start_s,
            end_s=seg.end_s,
            speaker_id=speaker,
            language=seg.language,
            raw_text=seg.text,
            confidence=seg.confidence,
            words=[],
        )]
        return out, next_id + 1

    # Tag each word (copy so we never mutate the engine's objects).
    tagged = [
        w.model_copy(update={"speaker_id": speaker_for_word(w, diarization)})
        for w in seg.words
    ]

    runs: list[AlignedSegment] = []
    run: list[WordSegment] = [tagged[0]]
    for w in tagged[1:]:
        if w.speaker_id == run[-1].speaker_id:
            run.append(w)
        else:
            runs.append(_run_to_segment(run, seg, next_id))
            next_id += 1
            run = [w]
    runs.append(_run_to_segment(run, seg, next_id))
    next_id += 1
    return runs, next_id


def _run_to_segment(
    run: list[WordSegment],
    seg: ASRSegment,
    seg_id: int,
) -> AlignedSegment:
    """Build a speaker-pure AlignedSegment from a run of same-speaker words.

    If the run covers the whole original segment (no split happened), keep
    the engine's original text to preserve its exact form; otherwise
    reconstruct the text by joining the run's words (titu words ARE the
    text — CTC emits no punctuation, so a space-join is faithful)."""
    whole = len(run) == len(seg.words)
    text = seg.text if whole else " ".join(w.word for w in run)
    return AlignedSegment(
        segment_id=seg_id,
        start_s=run[0].start_s,
        end_s=run[-1].end_s,
        speaker_id=run[0].speaker_id or _FALLBACK_SPEAKER,
        language=seg.language,
        raw_text=text,
        confidence=seg.confidence,
        words=run,
    )


def align_segments(
    asr_segments: list[ASRSegment],
    diarization: DiarizationResult,
) -> list[AlignedSegment]:
    """
    Merge ASR segments with speaker labels via word-level attribution.

    When a segment carries word timestamps (titu CTC), each word is tagged
    with its diarization speaker and the segment is split into speaker-pure
    runs at every speaker change. Segments without words fall back to the
    whole-segment maximum-overlap heuristic.

    Args:
        asr_segments: Transcription segments from ASR engine.
        diarization: Speaker diarization results.

    Returns:
        List of speaker-pure aligned segments.
    """
    if not asr_segments:
        logger.warning("No ASR segments to align")
        return []

    if not diarization.segments:
        logger.warning(
            "No diarization segments available, "
            "assigning all to SPEAKER_00"
        )
        return [
            AlignedSegment(
                segment_id=seg.segment_id,
                start_s=seg.start_s,
                end_s=seg.end_s,
                speaker_id=_FALLBACK_SPEAKER,
                language=seg.language,
                raw_text=seg.text,
                confidence=seg.confidence,
                words=seg.words,
            )
            for seg in asr_segments
        ]

    aligned: list[AlignedSegment] = []
    next_id = 0
    for seg in asr_segments:
        produced, next_id = _split_into_speaker_runs(
            seg, diarization, next_id
        )
        aligned.extend(produced)

    # Log alignment statistics
    speaker_counts: dict[str, int] = {}
    for a in aligned:
        speaker_counts[a.speaker_id] = speaker_counts.get(a.speaker_id, 0) + 1

    n_split = len(aligned) - len(asr_segments)
    logger.info(
        f"  ✓ Aligned {len(asr_segments)} chunks → {len(aligned)} "
        f"speaker-pure segments across {len(speaker_counts)} speakers"
        + (f" (+{n_split} from speaker-change splits)" if n_split > 0 else "")
    )
    for spk, count in sorted(speaker_counts.items()):
        logger.debug(f"    {spk}: {count} segments")

    return aligned


def merge_consecutive_segments(
    segments: list[AlignedSegment],
    max_gap_s: float = 0.5,
) -> list[AlignedSegment]:
    """
    Merge consecutive segments from the same speaker.

    If two adjacent segments have the same speaker and are within
    max_gap_s of each other, they are combined into a single segment.
    This reduces fragmentation from chunk boundaries.
    """
    if not segments:
        return []

    merged: list[AlignedSegment] = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]

        same_speaker = seg.speaker_id == prev.speaker_id
        same_language = seg.language == prev.language
        close_in_time = (seg.start_s - prev.end_s) <= max_gap_s

        if same_speaker and same_language and close_in_time:
            # Merge: extend the previous segment
            merged[-1] = AlignedSegment(
                segment_id=prev.segment_id,
                start_s=prev.start_s,
                end_s=seg.end_s,
                speaker_id=prev.speaker_id,
                language=prev.language,
                raw_text=f"{prev.raw_text} {seg.raw_text}",
                confidence=min(prev.confidence, seg.confidence),
                words=prev.words + seg.words,
            )
        else:
            merged.append(seg)

    if len(merged) < len(segments):
        logger.info(
            f"  ✓ Merged {len(segments)} → {len(merged)} segments "
            f"(combined same-speaker consecutive segments)"
        )

    return merged


# Bengali question words (whole-token match) — the enumerator asks; the
# respondent rarely does. কি/কী also appears as a copula, so it's weighted
# with the pronoun signal rather than relied on alone.
_BN_QUESTION_WORDS = {
    "কি", "কী", "কেন", "কোথায়", "কিভাবে", "কীভাবে", "কত", "কতটা", "কতটুকু",
    "কখন", "কোন", "কোনটা", "কে", "কারা", "কীসের", "কিসের",
}


def decide_role_map(
    segments: list[AlignedSegment],
    consent_keywords: list[str] | None = None,
    anchor_window_s: float = 600.0,
) -> dict[str, str]:
    """HEURISTIC cluster→role decision (fallback when the LLM classifier is
    unavailable). Returns {SPEAKER_xx: ENUMERATOR|RESPONDENT}.

    Signal (Bengali-native, validated 2026-06-25): the ENUMERATOR addresses the
    respondent formally as আপনি/আপনার/আপনাকে and ASKS questions; the RESPONDENT
    speaks of আমি/আমরা/আমার. So the enumerator cluster is the one with the most
    আপন- tokens + question words. (The old "most speech = enumerator" was
    BACKWARDS — respondents usually speak more — and mislabelled in testing.)

    Last resort when the signal is absent: the cluster that owns the EARLIEST
    segment (the enumerator opens with the greeting/consent script). Single
    cluster → all RESPONDENT. The LLM classifier is preferred; this catches
    its failures.
    """
    speakers = sorted({s.speaker_id for s in segments})
    if not speakers:
        return {}
    if len(speakers) == 1:
        logger.warning(
            "  Role mapping: only 1 speaker cluster — diarization did not "
            "separate enumerator/respondent. Labelling all RESPONDENT."
        )
        return {speakers[0]: RESPONDENT}

    kw = [k.lower() for k in (consent_keywords or []) if k.strip()]
    enum_score: dict[str, int] = defaultdict(int)
    for s in segments:
        text = (s.raw_text or "")
        toks = text.split()
        # আপন- = formal "you" (enumerator addressing respondent) — strongest cue
        enum_score[s.speaker_id] += 2 * sum(1 for t in toks if "আপন" in t)
        enum_score[s.speaker_id] += sum(1 for t in toks if t in _BN_QUESTION_WORDS)
        if kw:  # optional legacy consent-keyword boost
            low = text.lower()
            enum_score[s.speaker_id] += sum(1 for k in kw if k in low)

    best = max(speakers, key=lambda sp: enum_score.get(sp, 0))
    if enum_score.get(best, 0) > 0:
        enum, basis = best, f"আপন-/question signal (score={enum_score[best]})"
    else:
        # No linguistic signal — the enumerator opens the interview.
        first_seg = min(segments, key=lambda s: s.start_s)
        enum, basis = first_seg.speaker_id, "earliest speaker (no signal)"

    logger.info(
        "  Roles (heuristic fallback): %s → ENUMERATOR (%s); %d → RESPONDENT",
        enum, basis, len(speakers) - 1,
    )
    return {sp: (ENUMERATOR if sp == enum else RESPONDENT) for sp in speakers}


def map_speakers_to_roles(
    segments: list[AlignedSegment],
    consent_keywords: list[str] | None = None,
    anchor_window_s: float = 600.0,
) -> list[AlignedSegment]:
    """Apply the heuristic `decide_role_map` to segments (see that function).
    Kept for the offline test and as a standalone heuristic entry point."""
    if not segments:
        return segments
    return relabel_segments(
        segments, decide_role_map(segments, consent_keywords, anchor_window_s))


def relabel_segments(
    segments: list[AlignedSegment],
    mapping: dict[str, str],
) -> list[AlignedSegment]:
    """Rewrite each segment's (and its words') speaker_id via `mapping`."""
    out: list[AlignedSegment] = []
    for s in segments:
        role = mapping.get(s.speaker_id, RESPONDENT)
        words = [w.model_copy(update={"speaker_id": role}) for w in s.words]
        out.append(s.model_copy(update={"speaker_id": role, "words": words}))
    return out
