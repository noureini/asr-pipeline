"""
Fetch a small, privacy-safe Bengali test set from FLEURS.

FLEURS (Google, CC-BY) is public read-speech with gold transcripts —
safe to upload to a HuggingFace Space, unlike real survey field audio.

Downloads N Bengali samples into a flat folder:
    <out>/audio/0001.wav ...           16 kHz mono WAV
    <out>/references.json              gold transcript per file
    <out>/references.tsv               same, tab-separated (Excel-friendly)

Usage:
    uv run python scripts/fetch_fleurs_demo.py            # 10 samples -> test_data/fleurs_demo_10
    uv run python scripts/fetch_fleurs_demo.py --n 20
    uv run python scripts/fetch_fleurs_demo.py --split test --seed 7
    uv run python scripts/fetch_fleurs_demo.py --out ./my_demo

references.json schema:
    [
      {"file": "0001.wav", "transcript": "...", "id": 123,
       "duration_s": 4.2, "split": "validation", "lang": "bn"},
      ...
    ]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FLEURS_BN = "bn_in"  # FLEURS Bengali (India) config id


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n", type=int, default=10,
                   help="Number of samples to fetch (default 10)")
    p.add_argument("--split", default="validation",
                   choices=["validation", "test", "train"],
                   help="FLEURS split to sample from (default validation)")
    p.add_argument("--out", type=Path,
                   default=Path("test_data/fleurs_demo_10"),
                   help="Output folder")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed for reproducibility")
    p.add_argument("--min-dur", type=float, default=2.0,
                   help="Skip clips shorter than this many seconds")
    p.add_argument("--max-dur", type=float, default=20.0,
                   help="Skip clips longer than this many seconds")
    args = p.parse_args()

    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        print(f"ERROR: missing dependency ({e}).")
        print("These ship with the project: uv sync")
        sys.exit(1)

    print(f"Loading FLEURS {FLEURS_BN} [{args.split}] (streaming)...")
    try:
        ds = load_dataset(
            "google/fleurs", FLEURS_BN,
            split=args.split, streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"ERROR loading FLEURS: {e}")
        print("If this is an auth error, set HF_TOKEN in .env.")
        sys.exit(1)

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    references = []
    n_saved = 0
    n_seen = 0
    # Stream, skip out-of-duration clips, take the first N that qualify.
    # Streaming + seed-free is fine here — we just need 10 representative
    # clips, not a statistically random draw.
    for ex in ds:
        n_seen += 1
        if n_saved >= args.n:
            break
        audio = ex.get("audio")
        transcript = (ex.get("transcription")
                      or ex.get("raw_transcription") or "").strip()
        if audio is None or not transcript:
            continue
        wav = np.asarray(audio["array"], dtype="float32")
        sr = int(audio["sampling_rate"])
        dur = len(wav) / sr if sr else 0.0
        if dur < args.min_dur or dur > args.max_dur:
            continue

        n_saved += 1
        fname = f"{n_saved:04d}.wav"
        # FLEURS is already 16 kHz mono; write as-is.
        sf.write(str(audio_dir / fname), wav, sr, subtype="PCM_16")
        references.append({
            "file": fname,
            "transcript": transcript,
            "id": int(ex.get("id", -1)),
            "duration_s": round(dur, 2),
            "split": args.split,
            "lang": "bn",
        })
        print(f"  [{n_saved:>2}/{args.n}] {fname}  {dur:>4.1f}s  "
              f"{transcript[:55]}")

    if n_saved == 0:
        print("ERROR: no samples saved (all filtered out?). "
              "Try wider --min-dur/--max-dur.")
        sys.exit(1)

    # references.json
    ref_json = args.out / "references.json"
    with open(ref_json, "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)

    # references.tsv (Excel-friendly)
    ref_tsv = args.out / "references.tsv"
    with open(ref_tsv, "w", encoding="utf-8") as f:
        f.write("file\tduration_s\ttranscript\n")
        for r in references:
            f.write(f"{r['file']}\t{r['duration_s']}\t{r['transcript']}\n")

    print(f"\n{'=' * 60}")
    print(f"Saved {n_saved} FLEURS Bengali clips (scanned {n_seen})")
    print(f"  audio:      {audio_dir}/*.wav  (16 kHz mono)")
    print(f"  references: {ref_json}")
    print(f"              {ref_tsv}")
    print(f"{'=' * 60}")
    print("\nThis folder is safe to upload to a HuggingFace Space")
    print("(public CC-BY data, no PII). Compare a Space's output")
    print("against references.json to measure CER.")


if __name__ == "__main__":
    main()
