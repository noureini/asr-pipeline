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
import os
import sys
from pathlib import Path


def _ensure_cudnn_on_ld_path():
    """torch (pulled in by datasets audio decoding) needs libcudnn.so.9.
    The venv ships nvidia-cudnn-cu12 but its lib dir isn't on
    LD_LIBRARY_PATH. The dynamic linker reads LD_LIBRARY_PATH at process
    start, so we set it and re-exec once."""
    if os.environ.get("_FLEURS_CUDNN_FIXED") == "1":
        return
    candidates = []
    for base in sys.path:
        p = Path(base) / "nvidia" / "cudnn" / "lib"
        if p.is_dir():
            candidates.append(str(p))
        # also probe site-packages siblings
    # Fallback: glob the venv
    if not candidates:
        for sp in Path(sys.prefix).rglob("nvidia/cudnn/lib"):
            candidates.append(str(sp))
            break
    if not candidates:
        return  # nothing to do; let it fail with the clear error
    new_ld = os.pathsep.join(candidates + [os.environ.get("LD_LIBRARY_PATH", "")])
    os.environ["LD_LIBRARY_PATH"] = new_ld.strip(os.pathsep)
    os.environ["_FLEURS_CUDNN_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cudnn_on_ld_path()


def _load_hf_token():
    """Pull HF token from env or .env files (8 common var names)."""
    keys = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
            "HF_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_TOKEN",
            "HF_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
    for k in keys:
        v = os.environ.get(k)
        if v:
            os.environ.setdefault("HF_TOKEN", v)
            return v
    for env_path in (Path(".env"),
                     Path(__file__).parent.parent / ".env",
                     Path.home() / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip().upper() in keys:
                tok = val.strip().strip('"').strip("'")
                os.environ["HF_TOKEN"] = tok
                os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
                return tok
    return None


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
        from datasets import load_dataset, Audio
        import soundfile as sf
        import numpy as np
        import io
    except ImportError as e:
        print(f"ERROR: missing dependency ({e}).")
        print("These ship with the project: uv sync")
        sys.exit(1)

    tok = _load_hf_token()
    print(f"HF token: {'found' if tok else 'NOT found (public dataset, ok)'}")
    print(f"Loading FLEURS {FLEURS_BN} [{args.split}] (streaming)...")

    ds = None
    errors = []

    # Attempt 1: parquet mirror (NO trust_remote_code -> no torchaudio
    # import -> no libcudnn dependency). Works on datasets>=2.18 since
    # google/fleurs was converted to parquet.
    try:
        ds = load_dataset(
            "google/fleurs", FLEURS_BN,
            split=args.split, streaming=True,
            token=tok,
        )
        print("  loaded via parquet (no remote code)")
    except Exception as e:
        errors.append(f"parquet path: {e}")

    # Attempt 2: legacy loader script (needs torchaudio -> libcudnn).
    if ds is None:
        try:
            ds = load_dataset(
                "google/fleurs", FLEURS_BN,
                split=args.split, streaming=True,
                trust_remote_code=True, token=tok,
            )
            print("  loaded via legacy loader script")
        except Exception as e:
            errors.append(f"remote-code path: {e}")

    if ds is None:
        print("ERROR loading FLEURS. Tried both paths:")
        for er in errors:
            print(f"  - {er}")
        print("\nFix: uv pip install nvidia-cudnn-cu12   "
              "(then re-run; the script auto-adds it to LD_LIBRARY_PATH)")
        sys.exit(1)

    # Don't let datasets decode audio (torchcodec -> libcudnn). We parse
    # raw bytes with soundfile ourselves below.
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass  # some versions already give undecoded dicts in streaming

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
        # decode=False -> audio is {"bytes": ..., "path": ...}
        try:
            if audio.get("bytes"):
                wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            elif audio.get("path"):
                wav, sr = sf.read(audio["path"], dtype="float32")
            else:
                continue
        except Exception as e:
            print(f"  skip (decode error: {e})")
            continue
        # Stereo -> mono
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype="float32")
        sr = int(sr)
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
