"""
Per-part speaker diarization — RESUMABLE.

Uses the SAME pyannote backend as the production pipeline
(asr_pipeline.diarization.create_diarizer + the pipeline's
DiarizationConfig), so this is not a re-implementation: identical
model, identical torch.load patch, identical SPEAKER_xx
normalisation and min-segment filtering as a real pipeline run.

For each audio part it caches the speaker turns (LOCAL part-relative
seconds) to:

    <out-dir>/<stem>/<partstem>.diar.jsonl
    one JSON object per line: {"speaker","start","end"}

Times are wall-clock seconds of the ORIGINAL part file. The ASR
preprocessor only does ffmpeg->mono->16k (no trim/pad), so these
share the exact same time reference as <partstem>.chunks.jsonl —
assemble_transcript.py then applies the same part-offset logic to
get GLOBAL interview times. (pyannote / libsndfile cannot read
.m4a, so each part is decoded to a temp 16k-mono WAV first — the
pipeline likewise diarizes a preprocessed WAV, not the raw m4a.)

Acoustic first pass ONLY. Cluster -> ENUMERATOR / RESPONDENT /
OTHER role mapping happens deterministically in
assemble_transcript.py (scripted consent-intro anchor). No LLM.

Resumable: a part whose .diar.jsonl already exists (non-empty) is
skipped (use --force to redo).

Usage:
  uv run python scripts/diarize_parts.py --stem <STEM>
  uv run python scripts/diarize_parts.py --stem <STEM> --num-speakers 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

OUT = Path("outputs/real_eval_inproc")
AUDIO = Path("for testing (Noureini)")


def _to_wav(src: Path, dst: Path) -> None:
    """Decode to mono 16 kHz PCM WAV. No trim/pad => identical
    timeline to the ASR chunk timestamps (libsndfile/pyannote
    cannot read .m4a/AAC directly; the pipeline diarizes a
    preprocessed WAV too)."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         str(dst)],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--audio-folder", type=Path, default=AUDIO)
    ap.add_argument("--config", type=Path,
                    default=Path("omni_test_1b.yaml"),
                    help="Pipeline config (provides DiarizationConfig; "
                         "auth_token auto-filled from HF_TOKEN env)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="force EXACTLY this many speakers (strongest "
                         "prior; a survey is enumerator+respondent -> "
                         "use 2 when pyannote collapses a part into "
                         "one cluster). Sets min=max=N.")
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-diarize even if .diar.jsonl exists")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    d = args.out_dir / args.stem
    cfiles = sorted(d.glob("*.chunks.jsonl"))
    if not cfiles:
        sys.exit(f"no *.chunks.jsonl in {d} — run eval_real_survey_"
                 f"inproc (ASR) first so the part list is known")

    # Same config object the pipeline uses; load_config() auto-fills
    # diarization.auth_token from $HF_TOKEN (see config.py).
    from asr_pipeline.config import load_config
    from asr_pipeline.diarization import create_diarizer

    cfg = load_config(args.config)
    dcfg = cfg.diarization
    if args.num_speakers is not None:
        dcfg.min_speakers = args.num_speakers
        dcfg.max_speakers = args.num_speakers
    else:
        if args.min_speakers is not None:
            dcfg.min_speakers = args.min_speakers
        if args.max_speakers is not None:
            dcfg.max_speakers = args.max_speakers
    if not dcfg.auth_token:
        sys.exit("diarization.auth_token unset and no HF_TOKEN in env "
                 "— needed for pyannote/speaker-diarization-3.1")
    print(f"  backend={dcfg.backend} model={dcfg.model} "
          f"min={dcfg.min_speakers} max={dcfg.max_speakers}")

    diarizer = None  # lazy: skip model load if everything cached
    for cf in cfiles:
        partstem = cf.name[: -len(".chunks.jsonl")]
        out = d / f"{partstem}.diar.jsonl"
        if out.exists() and out.stat().st_size > 0 and not args.force:
            n = sum(1 for _ in out.open(encoding="utf-8"))
            print(f"  SKIP {partstem} ({n} turns cached)")
            continue
        audio = args.audio_folder / f"{partstem}.m4a"
        if not audio.exists():
            print(f"  WARN: {audio} missing — skipping {partstem}")
            continue

        if diarizer is None:
            diarizer = create_diarizer(dcfg, device=args.device)
            diarizer.load()

        print(f"  diarizing {partstem} ({audio.name}) ...", flush=True)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "part.wav"
            _to_wav(audio, wav)
            result = diarizer.diarize(wav)

        turns = sorted(
            ({"speaker": s.speaker_id,
              "start": round(float(s.start_s), 3),
              "end": round(float(s.end_s), 3)}
             for s in result.segments),
            key=lambda t: (t["start"], t["end"]),
        )
        spk = sorted({t["speaker"] for t in turns})
        tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for t in turns:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")
        tmp.replace(out)  # atomic -> resume-safe
        print(f"  OK {partstem}: {len(turns)} turns, "
              f"{result.num_speakers} speakers {spk} -> {out.name}")

    if diarizer is None:
        print("All parts already diarized (nothing to do). "
              "Use --force to redo.")
    else:
        diarizer.unload()


if __name__ == "__main__":
    main()
