"""
One-off helper: run mic-test acoustic + per-speaker analysis on a hand-picked
list of audio files (any path, any container) and *append* the results into
existing report JSONs so the per-folder XLSX rebuild picks them up.

Usage:
    uv run python scripts/mic_test_extra_files.py \
        outputs/bangladesh_phone_mic_test \
        --file PHONE-Call-ZHD "Phone (call rec)" \
            "test_data/.../PHONE_Call recording ZHD_260416_141356.m4a" \
        --file TABLET1-Int01 "Tablet1" \
            "test_data/.../TABLET1_Int01__1_1776336795332.mp3" \
        --file TABLET2-Int04 "Tablet2" \
            "test_data/.../TABLET2_Int04_8801711593288_0_1776337813401.mp3"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torchaudio


logger = logging.getLogger("mic_test_extra")


def analyze_acoustics(audio_path: Path, mic_name: str, folder_key: str, cfg) -> dict:
    """Returns a dict matching MicAudioMetrics shape."""
    from asr_pipeline.mic_test import MicTester

    tester = MicTester(cfg)
    metrics = tester.analyze_file(audio_path, mic_name, folder_key)
    return metrics.model_dump()


def analyze_per_speaker(
    audio_path: Path,
    folder_key: str,
    mic_name: str,
    cfg,
    diarizer,
    work_dir: Path,
) -> list[dict]:
    """Re-uses the per-speaker logic from mic_test_per_speaker.py."""
    sys.path.insert(0, str(Path(__file__).parent))
    from mic_test_per_speaker import analyze_file
    return analyze_file(audio_path, folder_key, mic_name, cfg, diarizer, work_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", type=Path,
                   help="Existing output dir holding mic-test-report.json + per-speaker-report.json")
    p.add_argument("--file", action="append", nargs=3, required=True,
                   metavar=("FOLDER_KEY", "MIC_NAME", "PATH"),
                   help="A file to add (repeat the flag once per file)")
    p.add_argument("--language", default="ben")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from asr_pipeline.config import load_config
    from asr_pipeline.diarization import create_diarizer

    cfg = load_config()

    # ── 1. Acoustic per-file metrics ──────────────────────────────────
    per_file_results: list[tuple[str, str, dict]] = []  # (folder, mic, metrics)
    for folder_key, mic_name, path_str in args.file:
        path = Path(path_str)
        if not path.exists():
            logger.error(f"  Missing file: {path}")
            continue
        logger.info(f"Acoustic: {folder_key} / {mic_name} / {path.name}")
        try:
            metrics = analyze_acoustics(path, mic_name, folder_key, cfg)
            per_file_results.append((folder_key, mic_name, metrics))
        except Exception as e:
            logger.error(f"  Failed: {e}")

    # ── 2. Per-speaker (diarization) ──────────────────────────────────
    logger.info("Loading diarization pipeline...")
    diarizer = create_diarizer(cfg.diarization, device=cfg.pipeline.device)
    diarizer.load()

    work_dir = Path(tempfile.mkdtemp(prefix="asr_extra_"))
    speaker_rows: list[dict] = []
    for folder_key, mic_name, path_str in args.file:
        path = Path(path_str)
        if not path.exists():
            continue
        logger.info(f"Per-speaker: {folder_key} / {path.name}")
        try:
            rows = analyze_per_speaker(path, folder_key, mic_name, cfg,
                                        diarizer, work_dir)
            speaker_rows.extend(rows)
        except Exception as e:
            logger.error(f"  Failed: {e}")
    diarizer.unload()

    # ── 3. Merge into the existing report JSONs ───────────────────────
    main_json = args.output_dir / "mic-test-report.json"
    spk_json = args.output_dir / "per-speaker-report.json"

    report = json.loads(main_json.read_text(encoding="utf-8"))
    # Group new acoustic metrics into existing mic_summaries (or create one)
    summaries_by_name = {s["mic_name"]: s for s in report["mic_summaries"]}
    for folder_key, mic_name, metrics in per_file_results:
        # avoid duplicate if rerun
        if mic_name not in summaries_by_name:
            new_summary = {
                "mic_name": mic_name,
                "num_files": 0,
                "avg_snr_db": 0.0, "avg_clipping_ratio": 0.0,
                "total_plosive_spikes": 0, "avg_spectral_rolloff_hz": 0.0,
                "avg_effective_bandwidth_hz": 0.0, "avg_crosstalk_ratio": 0.0,
                "avg_rms_dbfs": 0.0, "avg_speech_ratio": 0.0,
                "score": 0.0, "files": [],
            }
            report["mic_summaries"].append(new_summary)
            summaries_by_name[mic_name] = new_summary
        summary = summaries_by_name[mic_name]
        # remove existing entry with same file_path if present
        summary["files"] = [
            f for f in summary["files"] if f["file_path"] != metrics["file_path"]
        ]
        summary["files"].append(metrics)
        summary["num_files"] = len(summary["files"])
        # Re-compute aggregates for this mic
        files = summary["files"]
        n = max(1, len(files))
        summary["avg_snr_db"] = round(sum(f["snr_db"] for f in files) / n, 2)
        summary["avg_clipping_ratio"] = round(
            sum(f["clipping_ratio"] for f in files) / n, 6
        )
        summary["total_plosive_spikes"] = sum(f["plosive_spike_count"] for f in files)
        summary["avg_spectral_rolloff_hz"] = round(
            sum(f["spectral_rolloff_hz"] for f in files) / n, 1
        )
        summary["avg_effective_bandwidth_hz"] = round(
            sum(f["effective_bandwidth_hz"] for f in files) / n, 1
        )
        summary["avg_crosstalk_ratio"] = round(
            sum(f["crosstalk_ratio"] for f in files) / n, 4
        )
        summary["avg_rms_dbfs"] = round(sum(f["rms_dbfs"] for f in files) / n, 2)
        summary["avg_speech_ratio"] = round(
            sum(f["speech_ratio"] for f in files) / n, 3
        )

    main_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Updated {main_json}")

    # Per-speaker JSON
    if spk_json.exists():
        spk_data = json.loads(spk_json.read_text(encoding="utf-8"))
    else:
        spk_data = {"test_date": datetime.now().isoformat(),
                    "language": args.language, "rows": []}
    # Drop any prior rows for the same (folder, filename) to avoid dups
    new_keys = {(r["folder"], r["filename"]) for r in speaker_rows}
    spk_data["rows"] = [
        r for r in spk_data["rows"]
        if (r["folder"], r["filename"]) not in new_keys
    ] + speaker_rows
    spk_json.write_text(json.dumps(spk_data, indent=2), encoding="utf-8")
    logger.info(f"Updated {spk_json}")

    logger.info(
        f"Done. Added {len(per_file_results)} files, "
        f"{len(speaker_rows)} speaker rows."
    )
    logger.info(
        "Now re-run scripts/mic_test_per_folder_report.py on "
        f"{main_json} to refresh the XLSX/CSV/PNG."
    )


if __name__ == "__main__":
    main()
