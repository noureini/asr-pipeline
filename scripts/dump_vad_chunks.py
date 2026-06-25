"""
Stage-2 bridge (part 1/2) — MAIN venv.

Runs the pipeline's OWN VAD chunker (same config as the Omnilingual
baseline) and persists each chunk WAV + a manifest, so an external
engine (titu, isolated venv) can transcribe the IDENTICAL chunks,
idx-aligned with outputs/real_eval_inproc. This is what makes the
three-way comparison controlled (only the ASR model varies).

Outputs (in <out-dir>/<stem>/):
  _chunkwav/<partstem>/<idx:05d>.wav     persisted chunk audio
  <partstem>.manifest.jsonl              {idx,start,end,wav} per line

Usage:
  uv run python scripts/dump_vad_chunks.py --stem <STEM>
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

DEFAULT_FOLDER = Path("for testing (Noureini)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/real_eval_titu"))
    ap.add_argument("--config", type=Path,
                    default=Path("omni_test_1b.yaml"),
                    help="SAME config as the Omnilingual baseline so "
                         "VAD chunks (and idx) are identical.")
    args = ap.parse_args()

    from asr_pipeline.config import load_config
    from asr_pipeline.preprocessor import AudioPreprocessor

    cfg = load_config(args.config)
    audio = sorted(
        f for f in args.folder.iterdir()
        if f.suffix.lower() == ".m4a" and f.name.startswith(args.stem)
    )
    if not audio:
        raise SystemExit(f"no .m4a parts for '{args.stem}' in {args.folder}")

    d = args.out_dir / args.stem
    d.mkdir(parents=True, exist_ok=True)

    for a in audio:
        partstem = a.stem
        wavdir = d / "_chunkwav" / partstem
        wavdir.mkdir(parents=True, exist_ok=True)
        man = d / f"{partstem}.manifest.jsonl"
        print(f"  VAD-chunking {a.name} ...", flush=True)
        with tempfile.TemporaryDirectory(prefix="dump_") as tmp:
            pp = AudioPreprocessor(cfg.preprocessing, Path(tmp))
            _, chunks, _ = pp.preprocess(a)
            rows = []
            for i, ch in enumerate(chunks):
                dst = wavdir / f"{i:05d}.wav"
                shutil.copyfile(str(ch.waveform_path), str(dst))
                rows.append({
                    "idx": i,
                    "start": round(float(ch.start_s), 2),
                    "end": round(float(ch.end_s), 2),
                    "wav": str(dst),
                })
        with man.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {partstem}: {len(rows)} chunks -> {man.name} "
              f"(+ {wavdir})", flush=True)

    print("done. Next: run scripts/titu_transcribe_chunks.py in the "
          "isolated titu venv.")


if __name__ == "__main__":
    main()
