"""
Segment alignment: word-level speaker attribution (WhisperX-style).

Used by the general multilingual path, where speaker labels come from
voice diarization. Each word (titu / engine word timestamps) is tagged with
the diarization turn it overlaps, and a chunk is split into speaker-pure
runs at every speaker change. Segments without word timestamps fall back to
the whole-segment maximum-overlap heuristic.

The Bengali fusion path does not use diarization: the fusion LLM labels
ENUMERATOR / RESPONDENT from what is said (see fusion.py).
"""
from __future__ import annotations

import logging

from asr_pipeline.models import (
    AlignedSegment,
    ASRSegment,
    DiarizationResult,
    SpeakerSegment,
    WordSegment,
)

logger = logging.getLogger("asr_pipeline")

_FALLBACK_SPEAKER = "SPEAKER_00"


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
        logger.info(
            "  No diarization segments — one segment per ASR chunk "
            "(SPEAKER_00)"
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
