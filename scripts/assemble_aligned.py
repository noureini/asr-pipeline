"""WhisperX-style assembler — consumes word-aligned + speaker-pure
segments from word_align_and_split.py and emits the final TXT/JSON/SRT
deliverable.

Key differences vs assemble_transcript.py:
  - Reads <partstem>.aligned_segments.jsonl (per-speaker segments,
    each containing word-level timestamps + per-word speakers).
  - Speaker labels are exact at word boundaries — no "?"
    turn-straddle ambiguity.
  - The cleaned-text variant maps each aligned segment to whichever
    cleaned chunks overlap it in time, then concatenates their
    cleaned text (chunks are still chunk-level since cleanup runs
    per-chunk). The CLEANED variant therefore has speaker-pure
    boundaries with chunk-level cleaned text inside each.

Inputs (in <stem-dir>):
  <partstem>.aligned_segments.jsonl   required (from word_align_and_split.py)
  <partstem>.chunks.jsonl             required (for chunk→time mapping)
  <partstem>.corrected.jsonl          optional (per-chunk cleaned BN)
  <partstem>.translated.jsonl         optional (per-chunk EN)

Speaker role mapping is unchanged from assemble_transcript.py:
  - Anchor part: dominant speaker in first --anchor-window seconds
    = ENUMERATOR, next-most-speech = RESPONDENT.
  - Later parts: most-speech = RESPONDENT, next = ENUMERATOR.

Usage:
  uv run python scripts/assemble_aligned.py \\
      --stem 15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0,
                 str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_OUT = Path("outputs/real_eval_titu_excellent")
DEFAULT_AUDIO = Path("for testing (Noureini)")


def ffprobe_duration(audio: Path) -> Optional[float]:
    if not audio.exists():
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=nk=1:nw=1",
             str(audio)],
            capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception:
        return None


def map_roles_per_part(turns: list[dict], is_anchor_part: bool,
                         anchor_window: float) -> dict[str, str]:
    """Reuse the role-mapping logic from assemble_transcript.py:
    speaker dominating anchor window (first part) = ENUMERATOR.
    Later parts: most-speech = RESPONDENT, next = ENUMERATOR."""
    total: dict[str, float] = defaultdict(float)
    anchor: dict[str, float] = defaultdict(float)
    for t in turns:
        total[t["speaker"]] += t["end"] - t["start"]
        if t["start"] < anchor_window:
            anchor[t["speaker"]] += (
                min(t["end"], anchor_window) - t["start"])
    by_speech = sorted(total, key=lambda s: -total[s])
    role: dict[str, str] = {}
    if is_anchor_part and anchor:
        enum = max(anchor, key=lambda s: anchor[s])
        rest = [s for s in by_speech if s != enum]
        role[enum] = "ENUMERATOR"
        if rest:
            role[rest[0]] = "RESPONDENT"
        for s in rest[1:]:
            role[s] = "OTHER"
    else:
        if by_speech:
            role[by_speech[0]] = "RESPONDENT"
        if len(by_speech) > 1:
            role[by_speech[1]] = "ENUMERATOR"
        for s in by_speech[2:]:
            role[s] = "OTHER"
    return role


def chunks_overlapping(seg_start: float, seg_end: float,
                         chunks: list[dict]) -> list[dict]:
    """Return chunks whose [start,end] overlaps the segment range."""
    return [c for c in chunks
            if c["end"] > seg_start and c["start"] < seg_end]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--audio-folder", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--asr-model",
                    default="titu_stt_bn_fastconformer")
    ap.add_argument("--anchor-window", type=float, default=360.0)
    ap.add_argument("--use-cleaned", action="store_true",
                    help="emit cleaned (BN) + translated (EN) when "
                         "the *.corrected.jsonl / *.translated.jsonl "
                         "exist")
    args = ap.parse_args()

    stem_dir = args.out_dir / args.stem
    aligned_files = sorted(stem_dir.glob("*.aligned_segments.jsonl"))
    if not aligned_files:
        sys.exit(f"no *.aligned_segments.jsonl in {stem_dir}\n"
                 f"run scripts/word_align_and_split.py first")

    # Discover parts (in audio time order)
    parts: list[tuple[str, Path, Path]] = []
    for af in aligned_files:
        partstem = af.name[: -len(".aligned_segments.jsonl")]
        audio = args.audio_folder / f"{partstem}.m4a"
        parts.append((partstem, af, audio))

    # Global timeline: stack part durations
    cum_offset = 0.0
    part_offsets = {}
    for partstem, _, audio in parts:
        part_offsets[partstem] = cum_offset
        d = ffprobe_duration(audio) or 0.0
        cum_offset += d
    total_duration = cum_offset

    # Build per-part text maps (cleaned BN, translated EN) keyed by idx
    cleaned_text: dict[tuple[str, int], str] = {}
    translated_text: dict[tuple[str, int], str] = {}
    if args.use_cleaned:
        for partstem, _, _ in parts:
            cp = stem_dir / f"{partstem}.corrected.jsonl"
            if cp.exists():
                for ln in cp.read_text(
                        encoding="utf-8").splitlines():
                    ln = ln.strip()
                    if ln:
                        r = json.loads(ln)
                        cleaned_text[(partstem, r["idx"])] = (
                            r.get("text", ""))
            tp = stem_dir / f"{partstem}.translated.jsonl"
            if tp.exists():
                for ln in tp.read_text(
                        encoding="utf-8").splitlines():
                    ln = ln.strip()
                    if ln:
                        r = json.loads(ln)
                        translated_text[(partstem, r["idx"])] = (
                            r.get("text", ""))

    # Build the global segment list
    from asr_pipeline.formatter import write_transcript
    from asr_pipeline.models import (
        ProcessedSegment, TranscriptMetadata, TranscriptResult,
        TranscriptionStyle,
    )

    psegs: list[ProcessedSegment] = []
    all_speakers: set[str] = set()
    seg_global_id = 0

    for part_idx, (partstem, ap_path, audio) in enumerate(parts):
        offset = part_offsets[partstem]

        # Load aligned segments + chunks + diar turns
        aligned: list[dict] = []
        for ln in ap_path.read_text(
                encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                aligned.append(json.loads(ln))

        chunks_path = stem_dir / f"{partstem}.chunks.jsonl"
        chunks: list[dict] = []
        for ln in chunks_path.read_text(
                encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                chunks.append(json.loads(ln))

        # Build diar turns from aligned-segment speakers if no
        # separate diar.jsonl (the regrouping already collapsed it)
        diar_turns = [{
            "speaker": s["speaker"],
            "start": s["start_s"],
            "end": s["end_s"],
        } for s in aligned]

        # Role mapping per part (anchor on first part)
        role = map_roles_per_part(diar_turns,
                                    is_anchor_part=(part_idx == 0),
                                    anchor_window=args.anchor_window)

        # Convert each aligned segment → ProcessedSegment
        for seg in aligned:
            spk = seg["speaker"]
            label = role.get(spk, "OTHER")
            raw_text = seg.get("text", "")
            corr_text = ""
            en_text = ""
            if args.use_cleaned:
                # Find chunks overlapping this segment, concat
                # their cleaned text in order
                ov_chunks = chunks_overlapping(
                    seg["start_s"], seg["end_s"], chunks)
                ov_chunks.sort(key=lambda c: c["start"])
                corr_text = " ".join(
                    cleaned_text.get((partstem, c["idx"]), "")
                    for c in ov_chunks).strip()
                en_text = " ".join(
                    translated_text.get((partstem, c["idx"]), "")
                    for c in ov_chunks).strip()
            psegs.append(ProcessedSegment(
                segment_id=seg_global_id,
                start_s=offset + seg["start_s"],
                end_s=offset + seg["end_s"],
                speaker_id=label,
                language="ben",
                raw_text=raw_text,
                corrected_text=corr_text,
                english_translation=en_text,
                refined_translation="",
                confidence=0.0,
                source_file=audio.name,
            ))
            seg_global_id += 1
            all_speakers.add(label)

    # Metadata
    meta = TranscriptMetadata(
        project_name="MMI Bangladesh",
        audio_file=parts[0][2].name if parts else "",
        duration_s=total_duration,
        languages_detected=["ben"],
        num_speakers=len(all_speakers),
        transcription_style=TranscriptionStyle.VERBATIM,
        asr_engines_used=[args.asr_model],
        postprocessing_stages=(
            ["LLM cleanup (TigerLLM)", "translation (TigerLLM)"]
            if args.use_cleaned
            else ["pyannote 3.1 + word-level forced alignment "
                  "(MMS_FA) + per-word speaker assignment"]
        ),
    )
    result = TranscriptResult(metadata=meta, segments=psegs)
    out_dir = stem_dir
    written = write_transcript(
        result, out_dir, args.stem,
        output_format="all", timestamp_fmt="HH:MM:SS",
        include_raw=True,
        include_translation=args.use_cleaned,
    )
    print(f"wrote {len(written)} files:")
    for f in written:
        print(f"  {f}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
