"""
Output formatting module.

Generates the final transcript in standard qualitative research format.
Supports TXT, JSON, and SRT output formats.

Non-speech regions (silence, noise) are rendered as inline placeholders
in the TXT format to provide a complete timeline for research analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from asr_pipeline.models import (
    NonSpeechSegment,
    OutputFormat,
    ProcessedSegment,
    TranscriptMetadata,
    TranscriptResult,
    TranscriptionStyle,
)

logger = logging.getLogger("asr_pipeline")


# =============================================================================
# Timestamp formatting
# =============================================================================


def format_timestamp(seconds: float, fmt: str = "HH:MM:SS") -> str:
    """
    Format seconds into a human-readable timestamp.

    Args:
        seconds: Time in seconds.
        fmt: "HH:MM:SS" or "HH:MM:SS.mmm"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if fmt == "HH:MM:SS.mmm":
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int((secs % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millis:03d}"


# =============================================================================
# Language code to tag mapping
# =============================================================================


_LANG_TAGS: dict[str, str] = {
    "eng": "en", "spa": "es", "fra": "fr", "deu": "de",
    "por": "pt", "rus": "ru", "zho": "zh", "jpn": "ja",
    "kor": "ko", "ita": "it", "nld": "nl", "pol": "pl",
    "tur": "tr", "ces": "cs", "swe": "sv", "ukr": "uk",
    "ron": "ro", "ara": "ar", "hin": "hi", "ben": "bn",
    "nep": "ne", "swa": "sw", "amh": "am", "hau": "ha",
    "yor": "yo", "ibo": "ig", "tgl": "tl", "mya": "my",
    "khm": "km", "kin": "rw", "som": "so", "tir": "ti",
    "orm": "om",
}


def get_lang_tag(code: str) -> str:
    """Get a short language tag for display."""
    return _LANG_TAGS.get(code, code)


# =============================================================================
# TXT Formatter (Standard Qualitative Research Format)
# =============================================================================


def format_txt(
    result: TranscriptResult,
    timestamp_fmt: str = "HH:MM:SS",
    include_raw: bool = True,
    include_translation: bool = True,
) -> str:
    """
    Format transcript as a standard qualitative research text file.

    Follows professional transcription conventions:
    - Header block with full metadata + audio quality metrics
    - Timestamps at each speaker turn
    - Language tags per segment
    - Original + English translation on separate lines
    - Non-speech regions as inline placeholders (e.g., [NON-SPEECH — 3.2s])
    """
    lines: list[str] = []
    meta = result.metadata

    # ── Header ────────────────────────────────────────────────────
    sep = "=" * 72
    lines.append(sep)
    lines.append("TRANSCRIPT")
    lines.append(sep)

    # Interview metadata (batch mode)
    if meta.interview:
        iv = meta.interview
        lines.append(f"Interview:      {iv.interview_key}")
        lines.append(f"Time Range:     {iv.recording_start} — {iv.recording_end}")
        lines.append(f"Source Files:   {len(iv.source_files)}")

    lines.append(f"Project:        {meta.project_name or 'N/A'}")
    lines.append(f"Date:           {meta.recording_date or datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"Duration:       {format_timestamp(meta.duration_s)}")
    lines.append(f"Audio File:     {meta.audio_file}")
    lines.append(f"Languages:      {', '.join(meta.languages_detected)}")
    lines.append(f"Speakers:       {meta.num_speakers} identified")
    lines.append(f"Transcription:  {meta.transcription_style.value.replace('_', ' ').title()}")
    lines.append(f"ASR Engines:    {', '.join(meta.asr_engines_used)}")
    lines.append(f"Post-processed: {', '.join(meta.postprocessing_stages)}")

    # Audio quality metrics
    if meta.audio_quality:
        aq = meta.audio_quality
        lines.append(
            f"Audio Quality:  {aq.speech_ratio:.0%} speech | "
            f"{1 - aq.speech_ratio:.0%} non-speech | "
            f"Longest gap: {aq.longest_silence_s:.1f}s"
        )

    lines.append(sep)
    lines.append("")

    # ── Build sorted timeline of speech + non-speech ──────────────
    # Merge speech segments and non-speech regions into a single
    # time-ordered list for rendering.
    timeline: list[tuple[float, str, object]] = []

    for seg in result.segments:
        timeline.append((seg.start_s, "speech", seg))

    for ns in result.non_speech_segments:
        timeline.append((ns.start_s, "non_speech", ns))

    timeline.sort(key=lambda x: x[0])

    # ── Render timeline ──────────────────────────────────────────
    is_english_only = (
        len(meta.languages_detected) == 1
        and meta.languages_detected[0] in ("English", "eng")
    )

    for start_s, entry_type, entry in timeline:
        if entry_type == "non_speech":
            ns = entry  # type: NonSpeechSegment
            duration = ns.duration_s if ns.duration_s > 0 else (ns.end_s - ns.start_s)
            label = ns.region_type.upper().replace("_", "-")
            if ns.absolute_start:
                lines.append(f"        [{ns.absolute_start}] [{label} \u2014 {duration:.1f}s]")
            else:
                lines.append(f"        [{label} \u2014 {duration:.1f}s]")
            lines.append("")
        else:
            seg = entry  # type: ProcessedSegment
            # Use absolute timestamp if available, otherwise relative
            if seg.absolute_start:
                ts = seg.absolute_start
            else:
                ts = format_timestamp(seg.start_s, timestamp_fmt)
            lang_tag = get_lang_tag(seg.language)

            # Timestamp and speaker
            lines.append(f"[{ts}] {seg.speaker_id}:")

            if is_english_only:
                # English-only: just show corrected text
                lines.append(f"{seg.corrected_text}")
            else:
                # Multilingual: show original language + translation
                if include_raw:
                    lines.append(f"[{lang_tag}] {seg.corrected_text}")

                if include_translation and seg.language != "eng":
                    translation = seg.refined_translation or seg.english_translation
                    if translation:
                        lines.append(f"[en] {translation}")

            lines.append("")  # Blank line between segments

    # ── Footer ────────────────────────────────────────────────────
    lines.append(sep)
    lines.append("END OF TRANSCRIPT")
    lines.append(sep)

    return "\n".join(lines)


# =============================================================================
# JSON Formatter
# =============================================================================


def format_json(result: TranscriptResult) -> str:
    """Format transcript as structured JSON."""
    return result.model_dump_json(indent=2)


# =============================================================================
# SRT Formatter (Subtitle format)
# =============================================================================


def format_srt(
    result: TranscriptResult,
    use_translation: bool = True,
) -> str:
    """
    Format transcript as SRT subtitles.

    Uses refined English translation if available, otherwise
    corrected text in original language. Non-speech segments are
    excluded from SRT output (subtitles don't need silence markers).
    """
    lines: list[str] = []

    for i, seg in enumerate(result.segments, start=1):
        start_ts = format_timestamp_srt(seg.start_s)
        end_ts = format_timestamp_srt(seg.end_s)

        text = seg.corrected_text
        if use_translation and seg.refined_translation and seg.language != "eng":
            text = seg.refined_translation

        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(f"[{seg.speaker_id}] {text}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Write output files
# =============================================================================


def write_transcript(
    result: TranscriptResult,
    output_dir: Path,
    base_name: str,
    output_format: str = "txt",
    timestamp_fmt: str = "HH:MM:SS",
    include_raw: bool = True,
    include_translation: bool = True,
) -> list[Path]:
    """
    Write transcript to one or more output files.

    Args:
        result: Complete pipeline result.
        output_dir: Directory to write files to.
        base_name: Base filename (without extension).
        output_format: "txt", "json", "srt", or "all".
        timestamp_fmt: Timestamp format for TXT output.
        include_raw: Include raw (pre-correction) text in TXT.
        include_translation: Include English translation in TXT.

    Returns:
        List of paths to written files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    formats_to_write: list[str] = []
    if output_format == "all":
        formats_to_write = ["txt", "json", "srt"]
    else:
        formats_to_write = [output_format]

    for fmt in formats_to_write:
        output_path = output_dir / f"{base_name}.{fmt}"

        if fmt == "txt":
            content = format_txt(
                result, timestamp_fmt, include_raw, include_translation
            )
        elif fmt == "json":
            content = format_json(result)
        elif fmt == "srt":
            content = format_srt(result)
        else:
            logger.warning(f"Unknown output format: {fmt}")
            continue

        output_path.write_text(content, encoding="utf-8")
        written.append(output_path)
        logger.info(f"  ✓ Written: [file]{output_path}[/file]")

    return written
