"""
Whole-file long-form transcription with the Unlimited model — the
way the model is DESIGNED to be used.

This bypasses our VAD chunker entirely and hands each audio PART to
`ASRInferencePipeline.transcribe([whole_part_wav], lang=[...])` in a
single call, so Omnilingual's internal N=15s / M=1 cross-segment
conditioning spans the ENTIRE part (not reset every chunk/window
like eval_real_survey_inproc). This is the only configuration that
actually tests Meta's "long-form reduces hallucinations" claim.

Honest limits (by design of `transcribe()`):
  * Returns ONE string per part — NO timestamps. (Recover later via
    forced alignment ONLY if the text quality proves worth it.)
  * One call per part — NOT resumable mid-part; a crash = redo part.
  * Peak GPU mem unknown a priori — measured & printed; may OOM on
    8 GB. If so, this strategy needs a bigger GPU or large windows.

Resumable ACROSS parts: a part whose .wholefile.txt already exists
(non-empty) is skipped (use --force to redo).

Output (in <out-dir>/<stem>/):
  <partstem>.wholefile.txt    raw transcript (one line)
  <stem>.wholefile.json       per-part stats (secs, gpu_mb, loop diag)

Usage:
  uv run python scripts/transcribe_wholefile_unlimited.py --stem <STEM>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

OUT = Path("outputs/real_eval_wholefile")
AUDIO = Path("for testing (Noureini)")


def _to_wav(src: Path, dst: Path) -> None:
    """Decode to mono 16 kHz PCM WAV. No trim/pad => whole part, true
    timeline (libsndfile/fairseq2 read this fine; .m4a it does not)."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def _loop_diag(text: str) -> dict:
    """Cheap degenerate-loop signal so quality is judged at a glance:
    longest run of one repeated token, and the most frequent token's
    share. High values => repetition collapse."""
    toks = (text or "").split()
    n = len(toks)
    if n == 0:
        return {"tokens": 0, "max_run": 0, "top_tok_share": 0.0}
    max_run = run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        max_run = max(max_run, run)
    top_tok, top_cnt = Counter(toks).most_common(1)[0]
    return {
        "tokens": n,
        "max_run": max_run,
        "top_tok": top_tok,
        "top_tok_share": round(top_cnt / n, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--folder", type=Path, default=AUDIO)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--model-card", default="omniASR_LLM_Unlimited_1B_v2")
    ap.add_argument("--omni-lang", default="ben_Beng")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--force", action="store_true",
                    help="re-transcribe a part even if .wholefile.txt exists")
    args = ap.parse_args()

    # Part discovery identical to eval_real_survey_inproc.find_assets
    # (sorted .m4a starting with stem) so parts align with diarize/
    # assemble: index 0 = base, 1 = _1, ...
    files = list(args.folder.iterdir()) if args.folder.is_dir() else []
    audio = sorted(
        f for f in files
        if f.suffix.lower() == ".m4a" and f.name.startswith(args.stem)
    )
    if not audio:
        sys.exit(f"no .m4a parts for stem '{args.stem}' in {args.folder}")

    d = args.out_dir / args.stem
    d.mkdir(parents=True, exist_ok=True)

    todo = []
    for a in audio:
        outtxt = d / f"{a.stem}.wholefile.txt"
        if outtxt.exists() and outtxt.stat().st_size > 0 and not args.force:
            print(f"  SKIP {a.stem} (.wholefile.txt cached)")
        else:
            todo.append(a)
    if not todo:
        print("All parts already transcribed (--force to redo).")
        return

    import torch
    from omnilingual_asr.models.inference.pipeline import (
        ASRInferencePipeline,
    )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"Loading [{args.model_card}] on {args.device} "
          f"dtype={args.dtype} ...", flush=True)
    t0 = time.time()
    pipe = ASRInferencePipeline(model_card=args.model_card,
                                device=args.device, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    stats_path = d / f"{args.stem}.wholefile.json"
    stats = []
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            stats = []

    for a in todo:
        print(f"\n  PART {a.name}  (whole file, single transcribe call)",
              flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "part.wav"
            _to_wav(a, wav)
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            try:
                texts = pipe.transcribe(
                    [str(wav)], lang=[args.omni_lang], batch_size=1)
                txt = (texts[0] if texts else "").strip()
                err = None
            except Exception as e:  # OOM / other — report, keep going
                txt, err = "", repr(e)
            secs = time.time() - t1
        gpu_mb = 0.0
        if args.device == "cuda" and torch.cuda.is_available():
            gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        if err:
            print(f"  FAILED in {secs:.1f}s | peak GPU {gpu_mb:.0f} MiB"
                  f"\n  {err}", flush=True)
            stats = [s for s in stats if s.get("part") != a.stem]
            stats.append({"part": a.stem, "ok": False, "error": err,
                          "secs": round(secs, 1), "gpu_mb": round(gpu_mb)})
            stats_path.write_text(
                json.dumps(stats, ensure_ascii=False, indent=2),
                encoding="utf-8")
            continue

        (d / f"{a.stem}.wholefile.txt").write_text(
            txt + "\n", encoding="utf-8")
        diag = _loop_diag(txt)
        st = {"part": a.stem, "ok": True, "secs": round(secs, 1),
              "gpu_mb": round(gpu_mb), **diag}
        stats = [s for s in stats if s.get("part") != a.stem]
        stats.append(st)
        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"  OK {secs:.1f}s | peak GPU {gpu_mb:.0f} MiB | "
              f"{diag['tokens']} tok | max_run={diag['max_run']} | "
              f"top '{diag.get('top_tok')}' {diag['top_tok_share']:.0%}",
              flush=True)
        print(f"  head: {txt[:240]}", flush=True)
        print(f"  tail: {txt[-240:]}", flush=True)

    print(f"\nDone. Transcripts + {stats_path.name} in {d}/")
    print("max_run / top_tok_share high => repetition collapse "
          "(compare vs B-lite windows). No timestamps yet by design.")


if __name__ == "__main__":
    main()
