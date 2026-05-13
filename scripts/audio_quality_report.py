"""
Simple audio quality assessment — point at a folder, get an Excel report.

Walks the folder recursively for audio files, computes acoustic metrics per
file, classifies quality, writes a single Excel workbook with color-coded
results.

No setup, no JSON intermediate, no language flag, no mic-name mapping required.

Usage:
    uv run python scripts/audio_quality_report.py /path/to/audio/folder

    # Custom output location
    uv run python scripts/audio_quality_report.py /path/to/audio \
        -o ./my_quality_report.xlsx

    # Restrict to certain extensions
    uv run python scripts/audio_quality_report.py /path/to/audio \
        --extensions .m4a .wav

Output:
    quality_report.xlsx in the input folder (or wherever -o points)
    Contains: filename, duration, SNR, RMS, clipping %, speech %,
              quality flag (GOOD / OK / LOW SPEECH / POOR / BAD),
              and a comment column explaining the flag.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ─── Audio loading ───────────────────────────────────────────────────────

def load_audio_mono16k(path: Path) -> tuple[np.ndarray, int]:
    """Load any audio file as mono float32 at 16 kHz. Uses librosa."""
    import librosa
    wav, sr = librosa.load(str(path), sr=16000, mono=True)
    return wav.astype(np.float32), sr


# ─── Quality metrics ─────────────────────────────────────────────────────

def compute_metrics(wav: np.ndarray, sr: int = 16000) -> dict:
    """Compute per-file acoustic quality metrics.
    Returns dict with keys: duration_s, snr_db, rms_dbfs, clipping_pct,
                            speech_pct, peak_dbfs.
    """
    if len(wav) == 0:
        return {
            "duration_s": 0.0, "snr_db": 0.0, "rms_dbfs": -120.0,
            "clipping_pct": 0.0, "speech_pct": 0.0, "peak_dbfs": -120.0,
        }

    duration_s = len(wav) / sr

    # RMS (loudness)
    rms = np.sqrt(np.mean(wav.astype(np.float64) ** 2))
    rms_dbfs = 20 * np.log10(max(rms, 1e-12))

    # Peak level
    peak = np.max(np.abs(wav))
    peak_dbfs = 20 * np.log10(max(peak, 1e-12))

    # Clipping detection (samples within 0.5dB of full scale)
    clipping_threshold = 0.997
    clipping_pct = float(np.mean(np.abs(wav) >= clipping_threshold) * 100)

    # SNR via frame-energy percentile method
    # Frame the signal at 20ms hops
    frame_samples = int(0.02 * sr)
    n_frames = len(wav) // frame_samples
    if n_frames < 5:
        snr_db = 0.0
        speech_pct = 0.0
    else:
        frames = wav[: n_frames * frame_samples].reshape(n_frames, frame_samples)
        frame_energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        # Avoid log(0)
        frame_energy = np.maximum(frame_energy, 1e-9)
        frame_db = 20 * np.log10(frame_energy)

        # Signal = upper 25th percentile, noise = lower 25th percentile
        signal_db = np.percentile(frame_db, 75)
        noise_db = np.percentile(frame_db, 25)
        snr_db = float(signal_db - noise_db)

        # Speech % = frames significantly above noise floor (15 dB margin)
        speech_threshold_db = noise_db + 15
        speech_pct = float(np.mean(frame_db > speech_threshold_db) * 100)

    return {
        "duration_s": float(duration_s),
        "snr_db": float(snr_db),
        "rms_dbfs": float(rms_dbfs),
        "clipping_pct": float(clipping_pct),
        "speech_pct": float(speech_pct),
        "peak_dbfs": float(peak_dbfs),
    }


# ─── Quality classification ──────────────────────────────────────────────

def classify_quality(snr: float, speech_pct: float,
                     clipping_pct: float, rms_dbfs: float) -> tuple[str, str]:
    """Return (flag, comment). Thresholds tuned for phone-mic interview audio."""
    if clipping_pct > 1.0:
        return "CLIPPED", f"distortion: {clipping_pct:.1f}% clipped samples"
    if rms_dbfs < -50:
        return "BAD", "very quiet — mic likely muted or unplugged"
    if snr < 8:
        return "BAD", "near-silent / no usable speech"
    if snr < 12:
        return "POOR", "very noisy, expect high WER"
    if snr >= 20 and speech_pct >= 40:
        return "GOOD", "clean, speech-rich"
    if snr >= 15 and speech_pct < 25:
        return "LOW SPEECH", "high SNR but mostly silent — mic far / muted"
    if snr >= 12 and speech_pct >= 30:
        return "OK", "usable, moderate noise"
    return "OK", "marginal — review recommended"


# ─── Excel writer ────────────────────────────────────────────────────────

def write_excel(rows: list[dict], out_path: Path):
    """Write quality report to Excel with color-coded quality column."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("ERROR: openpyxl not installed. Run: uv pip install openpyxl")
        sys.exit(1)

    wb = Workbook()
    ws = wb.active
    ws.title = "Audio Quality"

    columns = [
        ("file", 50),
        ("folder", 30),
        ("duration_s", 12),
        ("snr_db", 10),
        ("rms_dbfs", 10),
        ("peak_dbfs", 11),
        ("clipping_pct", 13),
        ("speech_pct", 12),
        ("quality", 12),
        ("comment", 50),
    ]
    headers = [c[0] for c in columns]
    ws.append(headers)
    for i, h in enumerate(headers, 1):
        ws.cell(1, i).font = Font(bold=True)
        ws.cell(1, i).alignment = Alignment(horizontal="center")
        ws.column_dimensions[get_column_letter(i)].width = columns[i-1][1]
    ws.freeze_panes = "A2"

    # Color map
    quality_colors = {
        "GOOD":       "C6EFCE",  # green
        "OK":         "FFEB9C",  # light yellow
        "LOW SPEECH": "FFD18C",  # orange
        "POOR":       "FFC7CE",  # light red
        "BAD":        "FF6B6B",  # red
        "CLIPPED":    "B19CD9",  # purple
    }

    for r in rows:
        ws.append([
            r["file"],
            r["folder"],
            round(r["duration_s"], 2),
            round(r["snr_db"], 1),
            round(r["rms_dbfs"], 1),
            round(r["peak_dbfs"], 1),
            round(r["clipping_pct"], 3),
            round(r["speech_pct"], 1),
            r["quality"],
            r["comment"],
        ])
        # Color the quality cell
        quality_col_idx = headers.index("quality") + 1
        cell = ws.cell(row=ws.max_row, column=quality_col_idx)
        color = quality_colors.get(r["quality"])
        if color:
            cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")

    # Summary footer
    n = len(rows)
    summary_row = ws.max_row + 2
    ws.cell(summary_row, 1).value = "SUMMARY"
    ws.cell(summary_row, 1).font = Font(bold=True, size=12)

    summary_row += 1
    ws.cell(summary_row, 1).value = "Total files"
    ws.cell(summary_row, 2).value = n
    ws.cell(summary_row, 1).font = Font(bold=True)

    # Quality distribution
    from collections import Counter
    quality_counts = Counter(r["quality"] for r in rows)
    summary_row += 1
    ws.cell(summary_row, 1).value = "Quality distribution:"
    ws.cell(summary_row, 1).font = Font(bold=True)
    for flag in ["GOOD", "OK", "LOW SPEECH", "POOR", "BAD", "CLIPPED"]:
        count = quality_counts.get(flag, 0)
        if count == 0:
            continue
        summary_row += 1
        ws.cell(summary_row, 1).value = f"  {flag}"
        ws.cell(summary_row, 2).value = count
        ws.cell(summary_row, 3).value = f"{100*count/n:.0f}%"
        color = quality_colors.get(flag)
        if color:
            ws.cell(summary_row, 1).fill = PatternFill(
                start_color=color, end_color=color, fill_type="solid")

    # Mean metrics
    if n > 0:
        summary_row += 2
        ws.cell(summary_row, 1).value = "Average metrics:"
        ws.cell(summary_row, 1).font = Font(bold=True)
        for label, key in [("Mean SNR (dB)", "snr_db"),
                           ("Mean RMS (dBFS)", "rms_dbfs"),
                           ("Mean speech %", "speech_pct"),
                           ("Mean clipping %", "clipping_pct"),
                           ("Total duration (min)", None)]:
            summary_row += 1
            ws.cell(summary_row, 1).value = f"  {label}"
            if key:
                vals = [r[key] for r in rows]
                ws.cell(summary_row, 2).value = round(sum(vals) / len(vals), 2)
            else:
                total_dur = sum(r["duration_s"] for r in rows) / 60
                ws.cell(summary_row, 2).value = round(total_dur, 1)

    # Auto-filter on header row
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{1 + n}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Simple audio quality assessment to Excel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("folder", type=Path,
                   help="Folder containing audio files (searched recursively)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output Excel path (default: <folder>/quality_report.xlsx)")
    p.add_argument("--extensions", nargs="+",
                   default=[".m4a", ".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac"],
                   help="Audio file extensions to scan")
    p.add_argument("--max-files", type=int, default=-1,
                   help="Cap number of files to process (-1 = all)")
    args = p.parse_args()

    if not args.folder.exists():
        print(f"ERROR: folder not found: {args.folder}")
        sys.exit(1)

    if args.output is None:
        args.output = args.folder / "quality_report.xlsx"

    # Find audio files
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}"
            for e in args.extensions}
    audio_files = []
    for f in sorted(args.folder.rglob("*")):
        if f.is_file() and f.suffix.lower() in exts:
            audio_files.append(f)
            if args.max_files > 0 and len(audio_files) >= args.max_files:
                break

    if not audio_files:
        print(f"No audio files found in {args.folder} "
              f"(searched for: {sorted(exts)})")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio files in {args.folder}")
    print(f"Output: {args.output}\n")

    # Process each file
    import time
    rows = []
    n_errors = 0
    t0 = time.time()
    for i, path in enumerate(audio_files, 1):
        try:
            wav, sr = load_audio_mono16k(path)
            metrics = compute_metrics(wav, sr)
            flag, comment = classify_quality(
                metrics["snr_db"], metrics["speech_pct"],
                metrics["clipping_pct"], metrics["rms_dbfs"],
            )
            rel_path = path.relative_to(args.folder)
            row = {
                "file": str(rel_path.name),
                "folder": str(rel_path.parent) if rel_path.parent != Path(".") else "",
                "quality": flag,
                "comment": comment,
                **metrics,
            }
            rows.append(row)

            # Print short progress
            if i <= 5 or i % 10 == 0 or i == len(audio_files):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(audio_files) - i) / max(rate, 1e-6)
                print(f"  [{i:>3}/{len(audio_files)}] {flag:<10} "
                      f"SNR={metrics['snr_db']:>5.1f}dB  "
                      f"speech={metrics['speech_pct']:>4.0f}%  "
                      f"{rel_path.name[:50]}",
                      f"({rate:.1f}/s, ETA {eta:.0f}s)" if i < len(audio_files) else "")
        except Exception as e:
            n_errors += 1
            print(f"  [{i:>3}/{len(audio_files)}] ERROR ({type(e).__name__}): {path.name}")
            continue

    # Write Excel
    print(f"\nWriting Excel report → {args.output}")
    write_excel(rows, args.output)

    # Print summary to stdout
    from collections import Counter
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(rows)} files processed ({n_errors} errors)")
    print(f"{'=' * 60}")
    quality_counts = Counter(r["quality"] for r in rows)
    for flag in ["GOOD", "OK", "LOW SPEECH", "POOR", "BAD", "CLIPPED"]:
        count = quality_counts.get(flag, 0)
        if count > 0:
            pct = 100 * count / len(rows)
            print(f"  {flag:<12} {count:>4}  ({pct:.0f}%)")

    if rows:
        avg_snr = sum(r["snr_db"] for r in rows) / len(rows)
        avg_speech = sum(r["speech_pct"] for r in rows) / len(rows)
        total_dur = sum(r["duration_s"] for r in rows) / 60
        print(f"\n  Mean SNR:       {avg_snr:.1f} dB")
        print(f"  Mean speech %:  {avg_speech:.1f} %")
        print(f"  Total duration: {total_dur:.1f} min")

    print(f"\n  ✓ Open: {args.output}")


if __name__ == "__main__":
    main()
