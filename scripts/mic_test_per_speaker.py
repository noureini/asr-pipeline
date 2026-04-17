"""
Per-speaker audio quality analysis for the mic-test workflow.

For each audio file in the test folder we:
  1. Convert to 16 kHz mono WAV (re-uses the pipeline's preprocessor).
  2. Run VAD to find global speech regions.
  3. Run speaker diarization (pyannote 3.1 by default).
  4. For each detected speaker, compute the same acoustic metrics
     (SNR, bandwidth, rolloff, crosstalk, RMS, plosives, clipping) on
     just that speaker's audio.
  5. Label each speaker as ENUMERATOR or RESPONDENT using simple
     conversational heuristics (who speaks first, number of turns,
     average turn length).

Outputs in the chosen output directory:
  - per-speaker-report.csv
  - per-speaker-report.json   (consumed by mic_test_per_folder_report.py
                               to add a "Per speaker" tab to the XLSX)

Usage:
    uv run python scripts/mic_test_per_speaker.py FOLDER -l ben \
        -o outputs/bangladesh_phone_mic_test
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio


logger = logging.getLogger("mic_test_per_speaker")


# ───────────────────────────────────────────────────────────────────────
# Quality classification (matches the per-folder script)
# ───────────────────────────────────────────────────────────────────────

def classify_quality_snr_only(snr: float, talk_time_s: float) -> tuple[str, str]:
    """Per-speaker quality flag.

    Speech ratio is meaningless per speaker (it's just their share of the
    file), so we classify on SNR alone, with a duration sanity check.
    """
    if talk_time_s < 1.0:
        return ("BAD", "speaker barely audible (< 1 s of talk time)")
    if snr < 8:
        return ("BAD", "near-silent / no usable speech")
    if snr < 12:
        return ("POOR", "very noisy, expect high WER")
    if snr >= 20:
        return ("GOOD", "clean voice")
    return ("OK", "usable, moderate noise")


# ───────────────────────────────────────────────────────────────────────
# Per-speaker metric computation (re-uses MicTester static methods)
# ───────────────────────────────────────────────────────────────────────

def _build_speaker_mask(
    waveform_len: int,
    sr: int,
    segments: list[tuple[float, float]],
) -> np.ndarray:
    mask = np.zeros(waveform_len, dtype=bool)
    for start, end in segments:
        s = max(0, int(start * sr))
        e = min(waveform_len, int(end * sr))
        if e > s:
            mask[s:e] = True
    return mask


def compute_per_speaker_metrics(
    waveform: np.ndarray,
    sr: int,
    speaker_segments: dict[str, list[tuple[float, float]]],
    global_vad_regions: list[tuple[float, float]],
) -> dict[str, dict]:
    """Compute acoustic metrics for every speaker in a file.

    Returns a dict keyed by raw speaker_id (e.g. SPEAKER_00).
    """
    from asr_pipeline.mic_test import MicTester

    total_samples = len(waveform)

    # Global non-speech mask (silence / background noise) — used as the
    # noise floor reference when computing per-speaker SNR.
    speech_mask_all = _build_speaker_mask(total_samples, sr, global_vad_regions)
    noise_samples = waveform[~speech_mask_all]
    noise_power = (
        float(np.mean(noise_samples ** 2)) if len(noise_samples) > 0 else 1e-12
    )

    out: dict[str, dict] = {}
    speaker_ids = list(speaker_segments.keys())
    speaker_masks = {
        spk: _build_speaker_mask(total_samples, sr, segs)
        for spk, segs in speaker_segments.items()
    }

    for spk, segs in speaker_segments.items():
        my_mask = speaker_masks[spk]
        my_audio = waveform[my_mask]
        talk_time = float(np.sum(my_mask)) / sr

        if len(my_audio) == 0:
            continue

        # SNR vs global noise floor
        speech_power = float(np.mean(my_audio ** 2))
        if noise_power < 1e-12:
            snr_db = 60.0
        else:
            snr_db = float(10.0 * np.log10(max(speech_power, 1e-12) / noise_power))

        # Bandwidth + rolloff on speaker's concatenated audio
        rolloff = MicTester._compute_spectral_rolloff(my_audio, sr)
        bandwidth = MicTester._compute_effective_bandwidth(my_audio, sr)

        # Plosives + clipping on speaker's audio
        plosives = MicTester._detect_plosive_spikes(my_audio, sr)
        clipped, clip_ratio = MicTester._count_clipping(my_audio)

        # RMS
        rms = float(np.sqrt(np.mean(my_audio ** 2) + 1e-10))
        rms_dbfs = float(20.0 * np.log10(rms + 1e-10))

        # Crosstalk: energy of OTHER speakers leaking through
        # (= power(other_speakers' segments) / power(my segments))
        other_mask = np.zeros(total_samples, dtype=bool)
        for other_spk, other_m in speaker_masks.items():
            if other_spk == spk:
                continue
            other_mask |= other_m
        other_audio = waveform[other_mask]
        if len(other_audio) > 0 and speech_power > 1e-12:
            crosstalk = float(np.mean(other_audio ** 2)) / speech_power
        else:
            crosstalk = 0.0

        # Conversational stats
        turn_lengths = [end - start for start, end in segs]
        n_turns = len(turn_lengths)
        avg_turn_s = float(np.mean(turn_lengths)) if turn_lengths else 0.0
        first_speaks_at = float(min(start for start, _ in segs)) if segs else 0.0

        out[spk] = {
            "speaker_id": spk,
            "talk_time_s": round(talk_time, 2),
            "n_turns": n_turns,
            "avg_turn_s": round(avg_turn_s, 2),
            "first_speaks_at_s": round(first_speaks_at, 2),
            "snr_db": round(snr_db, 2),
            "clip_pct": round(clip_ratio * 100.0, 6),
            "plosives": int(plosives),
            "rolloff_hz": round(rolloff, 1),
            "bandwidth_hz": round(bandwidth, 1),
            "crosstalk": round(crosstalk, 4),
            "rms_dbfs": round(rms_dbfs, 2),
        }

    return out


# ───────────────────────────────────────────────────────────────────────
# Role labelling (enumerator vs respondent)
# ───────────────────────────────────────────────────────────────────────

def label_roles(speakers: dict[str, dict]) -> dict[str, str]:
    """Assign each speaker_id a role string (ENUMERATOR / RESPONDENT / OTHER).

    Heuristic — works best for 2-speaker phone interviews:
      * Whoever speaks **first** is most likely the enumerator.
      * Among the top-2 speakers, the one with **shorter average turns**
        and **more turns** is the enumerator (Q-A pattern).
      * If only one speaker, it's labelled UNKNOWN.
      * Speakers beyond the top 2 are labelled OTHER.
    """
    if not speakers:
        return {}

    # Rank speakers by total talk time (descending) to find the top 2
    ranked = sorted(speakers.values(), key=lambda s: s["talk_time_s"], reverse=True)

    if len(ranked) == 1:
        return {ranked[0]["speaker_id"]: "UNKNOWN"}

    a, b = ranked[0], ranked[1]
    extras = ranked[2:]

    # Score how "enumerator-like" each candidate is
    def enumerator_score(cand: dict, other: dict) -> float:
        score = 0.0
        # First speaker bonus
        if cand["first_speaks_at_s"] < other["first_speaks_at_s"]:
            score += 2.0
        # More turns
        if cand["n_turns"] > other["n_turns"]:
            score += 1.0
        # Shorter average turns
        if cand["avg_turn_s"] < other["avg_turn_s"]:
            score += 1.0
        return score

    a_score = enumerator_score(a, b)
    b_score = enumerator_score(b, a)

    if a_score >= b_score:
        roles = {a["speaker_id"]: "ENUMERATOR", b["speaker_id"]: "RESPONDENT"}
    else:
        roles = {a["speaker_id"]: "RESPONDENT", b["speaker_id"]: "ENUMERATOR"}

    for x in extras:
        roles[x["speaker_id"]] = "OTHER"

    return roles


# ───────────────────────────────────────────────────────────────────────
# Pipeline — analyze a single file
# ───────────────────────────────────────────────────────────────────────

def analyze_file(
    audio_path: Path,
    folder_key: str,
    mic_name: str,
    cfg,
    diarizer,
    work_dir: Path,
) -> list[dict]:
    """Return a list of per-speaker rows for one audio file."""
    from asr_pipeline.preprocessor import AudioPreprocessor

    file_work_dir = work_dir / folder_key / audio_path.stem
    file_work_dir.mkdir(parents=True, exist_ok=True)
    prep = AudioPreprocessor(cfg.preprocessing, file_work_dir)

    wav_path = prep._convert_to_wav(audio_path)
    waveform, sr = torchaudio.load(str(wav_path))
    wav_np = waveform.squeeze().numpy().astype(np.float64)

    # Global VAD (used as the noise-floor reference)
    vad_regions = prep._detect_speech(wav_path)

    # Free VAD model before loading diarizer-internal models
    prep._vad_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    diar_result = diarizer.diarize(wav_path)

    # Group diarization segments by speaker_id
    speaker_segments: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for seg in diar_result.segments:
        speaker_segments[seg.speaker_id].append((seg.start_s, seg.end_s))

    if not speaker_segments:
        logger.warning(f"  No speakers detected in {audio_path.name}")
        return []

    metrics_by_spk = compute_per_speaker_metrics(
        wav_np, sr, dict(speaker_segments), vad_regions
    )
    roles = label_roles(metrics_by_spk)

    rows: list[dict] = []
    for spk, m in metrics_by_spk.items():
        flag, comment = classify_quality_snr_only(m["snr_db"], m["talk_time_s"])
        rows.append({
            "folder": folder_key,
            "filename": audio_path.name,
            "mic": mic_name,
            "speaker_id": spk,
            "role": roles.get(spk, "UNKNOWN"),
            "talk_time_s": m["talk_time_s"],
            "n_turns": m["n_turns"],
            "avg_turn_s": m["avg_turn_s"],
            "snr_db": m["snr_db"],
            "clip_pct": m["clip_pct"],
            "plosives": m["plosives"],
            "rolloff_hz": m["rolloff_hz"],
            "bandwidth_hz": m["bandwidth_hz"],
            "crosstalk": m["crosstalk"],
            "rms_dbfs": m["rms_dbfs"],
            "quality_flag": flag,
            "comment": comment,
        })

    return rows


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path,
                   help="Mic-test root folder (same as `asr-pipeline test-mics`)")
    p.add_argument("-l", "--language", default="ben",
                   help="Language code (only used for the report header)")
    p.add_argument("-o", "--output-dir", type=Path, required=True,
                   help="Output dir (typically the same one used by test-mics)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N files (for quick smoke tests)")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load HF_TOKEN from .env (required for pyannote)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from asr_pipeline.config import load_config
    from asr_pipeline.diarization import create_diarizer
    from asr_pipeline.mic_test import discover_test_audio, parse_mic_mapping

    cfg = load_config()
    if args.device:
        cfg.pipeline.device = args.device

    readme = args.folder / "README.txt"
    if not readme.exists():
        raise SystemExit(f"No README.txt in {args.folder}")

    folder_mic_map = parse_mic_mapping(readme)
    file_map = discover_test_audio(args.folder)

    # Flatten + optionally limit
    all_files: list[tuple[str, Path]] = []
    for folder_key, files in sorted(file_map.items()):
        for f in files:
            all_files.append((folder_key, f))
    if args.limit:
        all_files = all_files[: args.limit]

    logger.info(f"Will process {len(all_files)} files")

    # Load diarizer once
    logger.info("Loading diarization pipeline...")
    diarizer = create_diarizer(cfg.diarization, device=cfg.pipeline.device)
    diarizer.load()

    work_dir = Path(tempfile.mkdtemp(prefix="asr_mic_speaker_"))
    rows: list[dict] = []
    for i, (folder_key, audio_path) in enumerate(all_files, start=1):
        mic_name = folder_mic_map.get(folder_key, f"Unknown ({folder_key})")
        logger.info(f"[{i}/{len(all_files)}] {folder_key} / {audio_path.name}")
        try:
            file_rows = analyze_file(
                audio_path, folder_key, mic_name, cfg, diarizer, work_dir
            )
            rows.extend(file_rows)
        except Exception as e:
            logger.error(f"  Failed: {e}")

    diarizer.unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per-speaker-report.csv"
    json_path = args.output_dir / "per-speaker-report.json"

    write_csv(rows, csv_path)
    json_path.write_text(
        json.dumps(
            {
                "test_date": datetime.now().isoformat(),
                "language": args.language,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(f"Wrote {csv_path}")
    logger.info(f"Wrote {json_path}")
    logger.info(
        f"Done. {len(rows)} speaker rows from {len(all_files)} files."
    )


if __name__ == "__main__":
    main()
