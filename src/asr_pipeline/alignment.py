"""
Segment alignment module.

Merges ASR transcription segments with speaker diarization results
using temporal intersection. Each ASR segment is assigned to the
speaker who has the most overlap with it.
"""

from __future__ import annotations

import logging

from asr_pipeline.models import (
    AlignedSegment,
    ASRSegment,
    DiarizationResult,
    SpeakerSegment,
)

logger = logging.getLogger("asr_pipeline")


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


def align_segments(
    asr_segments: list[ASRSegment],
    diarization: DiarizationResult,
) -> list[AlignedSegment]:
    """
    Merge ASR segments with speaker labels from diarization.

    For each ASR segment, finds the speaker with maximum temporal
    overlap and creates an AlignedSegment combining both.

    Args:
        asr_segments: Transcription segments from ASR engine.
        diarization: Speaker diarization results.

    Returns:
        List of aligned segments with speaker IDs assigned.
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
                speaker_id="SPEAKER_00",
                language=seg.language,
                raw_text=seg.text,
                confidence=seg.confidence,
                words=seg.words,
            )
            for seg in asr_segments
        ]

    aligned: list[AlignedSegment] = []

    for seg in asr_segments:
        speaker = find_best_speaker(seg, diarization)

        aligned.append(
            AlignedSegment(
                segment_id=seg.segment_id,
                start_s=seg.start_s,
                end_s=seg.end_s,
                speaker_id=speaker,
                language=seg.language,
                raw_text=seg.text,
                confidence=seg.confidence,
                words=seg.words,
            )
        )

    # Log alignment statistics
    speaker_counts: dict[str, int] = {}
    for a in aligned:
        speaker_counts[a.speaker_id] = speaker_counts.get(a.speaker_id, 0) + 1

    logger.info(
        f"  ✓ Aligned {len(aligned)} segments across "
        f"{len(speaker_counts)} speakers"
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
