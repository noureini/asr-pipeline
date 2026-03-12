"""
Batch processing for Survey Solutions interview folder structures.

Handles folder discovery, audio filename parsing, and transcript merging
for the interview-based folder layout:

    root/{interview_key}/AudioAudit/{uuid}-audio-audit-{YYYYMMDD}_{hhmmssfff}.m4a
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from asr_pipeline.models import (
    AudioFileInfo,
    InterviewMetadata,
    NonSpeechSegment,
    ProcessedSegment,
    TranscriptMetadata,
    TranscriptResult,
    TranscriptionStyle,
)

logger = logging.getLogger("asr_pipeline")

# Regex for Survey Solutions audio audit filenames
# e.g., 02c852cd6b574aa69968ff51b48e2a8d-audio-audit-20231014_141450489.m4a
_FILENAME_RE = re.compile(
    r"^([0-9a-f]+)-audio-audit-(\d{8})_(\d{9})\.m4a$",
    re.IGNORECASE,
)


def parse_audio_filename(filename: str) -> tuple[str, datetime]:
    """
    Parse a Survey Solutions audio audit filename.

    Extracts the interview UUID and the recording start datetime from
    the filename format: {uuid}-audio-audit-{YYYYMMDD}_{hhmmssfff}.m4a

    Args:
        filename: The audio filename (not full path).

    Returns:
        Tuple of (uuid, recording_start_datetime).

    Raises:
        ValueError: If the filename doesn't match the expected pattern.
    """
    match = _FILENAME_RE.match(filename)
    if not match:
        raise ValueError(
            f"Filename does not match expected pattern: {filename}\n"
            f"Expected: {{uuid}}-audio-audit-{{YYYYMMDD}}_{{hhmmssfff}}.m4a"
        )

    uuid_str = match.group(1)
    date_str = match.group(2)  # YYYYMMDD
    time_str = match.group(3)  # hhmmssfff (9 digits)

    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    hour = int(time_str[:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])
    millisecond = int(time_str[6:9])

    recording_start = datetime(
        year, month, day, hour, minute, second, millisecond * 1000
    )

    return uuid_str, recording_start


def discover_interviews(root_dir: Path) -> dict[str, list[AudioFileInfo]]:
    """
    Scan a root directory for Survey Solutions interview folders.

    Expects the structure:
        root/{interview_key}/AudioAudit/*.m4a

    Args:
        root_dir: Path to the root folder containing interview subfolders.

    Returns:
        Dict mapping interview_key -> list of AudioFileInfo,
        sorted by recording start time within each interview.
    """
    interviews: dict[str, list[AudioFileInfo]] = {}

    if not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    for interview_dir in sorted(root_dir.iterdir()):
        if not interview_dir.is_dir():
            continue

        audio_dir = interview_dir / "AudioAudit"
        if not audio_dir.is_dir():
            # Also check without the exact case
            for subdir in interview_dir.iterdir():
                if subdir.is_dir() and subdir.name.lower() == "audioaudit":
                    audio_dir = subdir
                    break
            else:
                continue

        interview_key = interview_dir.name
        files: list[AudioFileInfo] = []

        for audio_file in sorted(audio_dir.glob("*.m4a")):
            try:
                uuid_str, recording_start = parse_audio_filename(audio_file.name)
                files.append(
                    AudioFileInfo(
                        filename=audio_file.name,
                        recording_start=recording_start.isoformat(
                            timespec="milliseconds"
                        ),
                        file_path=str(audio_file),
                    )
                )
            except ValueError as e:
                logger.warning(f"  Skipping unrecognized file: {e}")
                continue

        if files:
            # Sort by recording start time
            files.sort(key=lambda f: f.recording_start)
            interviews[interview_key] = files

    return interviews


def merge_interview_results(
    file_results: list[tuple[AudioFileInfo, TranscriptResult]],
    interview_key: str,
    project_name: str = "",
) -> TranscriptResult:
    """
    Merge multiple single-file TranscriptResults into one consolidated
    interview transcript.

    - Re-numbers segment_ids sequentially
    - Stamps each segment with source_file and absolute timestamps
    - Prefixes speaker_ids with file index to avoid cross-file collisions
    - Sorts all segments chronologically by absolute time
    - Builds combined metadata with InterviewMetadata

    Args:
        file_results: List of (AudioFileInfo, TranscriptResult) pairs,
            sorted by recording start time.
        interview_key: The interview identifier (folder name).
        project_name: Optional project name for metadata.

    Returns:
        A single consolidated TranscriptResult.
    """
    if not file_results:
        raise ValueError("No file results to merge")

    # Parse the interview start time (earliest recording)
    interview_start_dt = min(
        datetime.fromisoformat(fi.recording_start) for fi, _ in file_results
    )

    all_segments: list[ProcessedSegment] = []
    all_non_speech: list[NonSpeechSegment] = []
    all_languages: set[str] = set()
    all_engines: set[str] = set()
    all_postprocessing: set[str] = set()
    total_duration = 0.0
    total_speakers = 0

    for file_idx, (file_info, result) in enumerate(file_results, start=1):
        file_start_dt = datetime.fromisoformat(file_info.recording_start)
        file_offset_s = (file_start_dt - interview_start_dt).total_seconds()
        file_prefix = f"F{file_idx}"

        # Update file duration from result metadata
        file_info.duration_s = result.metadata.duration_s
        total_duration += result.metadata.duration_s
        total_speakers += result.metadata.num_speakers

        # Collect metadata
        all_languages.update(result.metadata.languages_detected)
        all_engines.update(result.metadata.asr_engines_used)
        all_postprocessing.update(result.metadata.postprocessing_stages)

        # Process speech segments
        for seg in result.segments:
            abs_start_dt = file_start_dt + timedelta(seconds=seg.start_s)
            abs_end_dt = file_start_dt + timedelta(seconds=seg.end_s)

            all_segments.append(
                ProcessedSegment(
                    segment_id=0,  # Re-numbered later
                    start_s=file_offset_s + seg.start_s,
                    end_s=file_offset_s + seg.end_s,
                    speaker_id=f"SPEAKER_{file_prefix}_{seg.speaker_id.replace('SPEAKER_', '')}",
                    language=seg.language,
                    raw_text=seg.raw_text,
                    corrected_text=seg.corrected_text,
                    english_translation=seg.english_translation,
                    refined_translation=seg.refined_translation,
                    confidence=seg.confidence,
                    source_file=file_info.filename,
                    absolute_start=abs_start_dt.isoformat(
                        timespec="milliseconds"
                    ),
                    absolute_end=abs_end_dt.isoformat(timespec="milliseconds"),
                )
            )

        # Process non-speech segments
        for ns in result.non_speech_segments:
            abs_start_dt = file_start_dt + timedelta(seconds=ns.start_s)
            abs_end_dt = file_start_dt + timedelta(seconds=ns.end_s)

            all_non_speech.append(
                NonSpeechSegment(
                    start_s=file_offset_s + ns.start_s,
                    end_s=file_offset_s + ns.end_s,
                    region_type=ns.region_type,
                    duration_s=ns.duration_s,
                    source_file=file_info.filename,
                    absolute_start=abs_start_dt.isoformat(
                        timespec="milliseconds"
                    ),
                    absolute_end=abs_end_dt.isoformat(timespec="milliseconds"),
                )
            )

    # Sort by absolute time and re-number
    all_segments.sort(key=lambda s: s.absolute_start or "")
    for i, seg in enumerate(all_segments):
        seg.segment_id = i

    all_non_speech.sort(key=lambda ns: ns.absolute_start or "")

    # Compute interview end time
    interview_end_dt = interview_start_dt
    for file_info, result in file_results:
        file_start_dt = datetime.fromisoformat(file_info.recording_start)
        file_end_dt = file_start_dt + timedelta(seconds=result.metadata.duration_s)
        if file_end_dt > interview_end_dt:
            interview_end_dt = file_end_dt

    # Extract interview_id from the first file's UUID
    first_uuid: Optional[str] = None
    try:
        first_uuid, _ = parse_audio_filename(file_results[0][0].filename)
    except ValueError:
        pass

    interview_meta = InterviewMetadata(
        interview_key=interview_key,
        interview_id=first_uuid,
        source_files=[fi for fi, _ in file_results],
        total_duration_s=total_duration,
        recording_start=interview_start_dt.isoformat(timespec="milliseconds"),
        recording_end=interview_end_dt.isoformat(timespec="milliseconds"),
    )

    # Use the first result's transcription style as representative
    style = file_results[0][1].metadata.transcription_style

    metadata = TranscriptMetadata(
        project_name=project_name,
        audio_file=interview_key,
        duration_s=(interview_end_dt - interview_start_dt).total_seconds(),
        recording_date=interview_start_dt.strftime("%Y-%m-%d"),
        languages_detected=sorted(all_languages),
        num_speakers=total_speakers,
        transcription_style=style,
        asr_engines_used=sorted(all_engines),
        postprocessing_stages=sorted(all_postprocessing),
        interview=interview_meta,
    )

    return TranscriptResult(
        metadata=metadata,
        segments=all_segments,
        non_speech_segments=all_non_speech,
    )
