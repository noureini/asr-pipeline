"""Prepare the qualitative-review source materials per interview.

Copies the audio file(s) and extracts the ground-truth docx
transcripts and translations into a self-contained source/ folder
alongside the model outputs.

Resulting structure (used for qualitative evaluation):

  outputs/v1_deliverable/<stem>/
    source/
      <stem>.m4a                       (copy or symlink)
      <stem>_1.m4a                     (if usable second part)
      gt_transcription.txt             (extracted from .docx)
      gt_transcription.docx            (copy of original)
      gt_translation.txt               (extracted, if exists)
      gt_translation.docx              (copy of original, if exists)
      audio_quality.txt                (line copied from the quality
                                        report — SNR, duration, etc.)
    raw/                ← Pipeline 1 output
    cleaned_tigerllm/   ← Pipeline 2A output
    cleaned_claude/     ← Pipeline 2B placeholder

The intent: a reviewer can open outputs/v1_deliverable/<stem>/
and have everything needed to qualitatively evaluate transcription
quality without hunting across the repo.

Usage:
  uv run python scripts/prepare_deliverable.py \\
      --stem 06_Shatkhira_CA-MMI_RURAL_Male_06_0202_1150 \\
      --audio-folder "for testing (Noureini)" \\
      --skip-bad-parts   # default: drops parts marked BAD in quality
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def extract_docx_text(p: Path) -> str:
    """Plain text extraction, preserving paragraph breaks."""
    try:
        import docx
    except ImportError:
        return f"[python-docx not installed: cannot extract {p.name}]"
    try:
        d = docx.Document(str(p))
    except Exception as e:
        return f"[could not open {p.name}: {e}]"
    out = []
    for para in d.paragraphs:
        t = (para.text or "").strip()
        if t:
            out.append(t)
    return "\n\n".join(out)


def read_quality_for(stem: str, quality_xlsx: Path) -> str:
    """Pull the audio quality lines matching this stem from the
    quality_report.xlsx (one line per audio part)."""
    if not quality_xlsx.exists():
        return ""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(quality_xlsx)
        ws = wb["Audio Quality"] if "Audio Quality" in wb.sheetnames \
            else wb.active
        headers = [c.value for c in ws[1]]
        lines = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            fname = row[0] or ""
            if not str(fname).startswith(stem):
                continue
            kv = dict(zip(headers, row))
            lines.append(
                f"{kv.get('file','?')}  "
                f"dur={kv.get('duration_s','?')}s  "
                f"SNR={kv.get('snr_db','?')}dB  "
                f"speech={kv.get('speech_pct','?')}%  "
                f"quality={kv.get('quality','?')}  "
                f"score={kv.get('score','?')}\n"
                f"  comment: {kv.get('comment','')}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"[could not read quality report: {e}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--audio-folder", type=Path,
                    default=Path("for testing (Noureini)"))
    ap.add_argument("--quality-xlsx", type=Path,
                    default=Path("for testing (Noureini)")
                            / "quality_report.xlsx")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: outputs/v1_deliverable/<stem>/source")
    ap.add_argument("--copy-audio", action="store_true", default=True,
                    help="copy audio (default) vs --symlink")
    ap.add_argument("--symlink", action="store_true",
                    help="symlink audio instead of copying "
                         "(saves disk)")
    ap.add_argument("--skip-bad-parts", action="store_true",
                    default=True,
                    help="if quality says BAD for a part, skip it")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = (REPO / "outputs" / "v1_deliverable"
                        / args.stem / "source")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Find audio files for this stem
    audio_files = sorted(
        p for p in args.audio_folder.iterdir()
        if p.suffix.lower() == ".m4a"
        and p.name.startswith(args.stem))
    if not audio_files:
        sys.exit(f"no .m4a for stem {args.stem} in "
                 f"{args.audio_folder}")

    # Quality table — used to skip BAD parts
    quality_text = read_quality_for(args.stem, args.quality_xlsx)
    bad_names: set[str] = set()
    if args.skip_bad_parts and quality_text:
        for line in quality_text.splitlines():
            if "quality=BAD" in line:
                # First token is the filename
                bad_names.add(line.split()[0])

    # Copy/symlink audio
    print(f"AUDIO ({len(audio_files)} part(s) found):", flush=True)
    for src in audio_files:
        dst = args.out_dir / src.name
        if src.name in bad_names:
            print(f"  skip (BAD quality): {src.name}")
            continue
        if dst.exists():
            print(f"  exists: {dst.name}")
            continue
        if args.symlink:
            dst.symlink_to(src.resolve())
            print(f"  symlink {src.name} -> {dst}")
        else:
            shutil.copy(src, dst)
            print(f"  copy    {src.name} -> {dst}  "
                  f"({src.stat().st_size//1024} KB)")

    # GT docx — both transcription and translation if present
    docx_files = sorted(
        p for p in args.audio_folder.iterdir()
        if p.suffix.lower() == ".docx"
        and p.name.startswith(args.stem))
    print(f"\nGROUND TRUTH ({len(docx_files)} docx file(s) found):",
          flush=True)
    for src in docx_files:
        dst = args.out_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)
            print(f"  copy {src.name} -> {dst}")
        # Also extract plain text
        text = extract_docx_text(src)
        if "transcription" in src.name.lower() \
                or "transcript" in src.name.lower():
            txt_path = args.out_dir / "gt_transcription.txt"
        elif "translation" in src.name.lower():
            txt_path = args.out_dir / "gt_translation.txt"
        else:
            txt_path = args.out_dir / (src.stem + ".txt")
        txt_path.write_text(text, encoding="utf-8")
        print(f"  -> {txt_path.name}  ({len(text)} chars)")

    # Quality summary
    if quality_text:
        (args.out_dir / "audio_quality.txt").write_text(
            f"=== Audio quality (from quality_report.xlsx) ===\n"
            f"Stem: {args.stem}\n\n{quality_text}",
            encoding="utf-8")
        print(f"\n  -> audio_quality.txt")

    # README inside source/ orienting the reviewer
    readme = args.out_dir.parent / "README.md"
    readme.write_text(
        f"# Deliverable — {args.stem}\n\n"
        f"## Folders\n\n"
        f"- `source/` — original audio + ground-truth docx + "
        f"audio quality summary. For qualitative review.\n"
        f"- `raw/` — Pipeline 1 deliverable: titu CTC + pyannote "
        f"speaker diarization, no LLM cleanup.\n"
        f"- `cleaned_tigerllm/` — Pipeline 2A deliverable: TigerLLM "
        f"cleanup + English translation.\n"
        f"- `cleaned_claude/` — Pipeline 2B (manual). Populate by "
        f"pasting `raw/transcript.txt` into the Claude web app "
        f"within your secure infrastructure.\n\n"
        f"## Qualitative review workflow\n\n"
        f"1. Play `source/*.m4a` while reading "
        f"`source/gt_transcription.txt` (the human reference).\n"
        f"2. Compare against `raw/transcript.txt` (titu output) — "
        f"identifies the ASR's faithful but rough capture.\n"
        f"3. Compare against `cleaned_tigerllm/transcript.txt` — "
        f"identifies what the local LLM cleanup adds (and breaks).\n"
        f"4. Compare against `cleaned_claude/transcript.txt` — "
        f"identifies what the premium LLM cleanup adds.\n",
        encoding="utf-8")
    print(f"\n  -> {readme}")

    # Placeholders for the Claude variant
    (args.out_dir.parent / "cleaned_claude").mkdir(
        parents=True, exist_ok=True)
    (args.out_dir.parent / "cleaned_claude" / ".gitkeep").touch()
    print(f"  -> cleaned_claude/ (placeholder)")

    print(f"\ndone. source folder ready at:\n  {args.out_dir}")


if __name__ == "__main__":
    main()
