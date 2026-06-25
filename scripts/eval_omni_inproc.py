"""
In-process Omnilingual CER eval on the 10 FLEURS Bengali clips.

Loads the Omnilingual model ONCE and transcribes all clips in the same
process — no per-clip subprocess, no diarization, no forced alignment,
no LLM post-processing. This is the fast path for measuring RAW ASR CER
and for A/B-ing model cards (300M vs 1B) on an 8 GB GPU.

It deliberately bypasses ASRPipeline (which always runs pyannote +
MMS forced alignment — both irrelevant to CER and the cause of the
multi-minute stalls in the subprocess-per-clip harness).

CER normalization (base_norm / num_norm / cer) is copied verbatim from
scripts/eval_omni_fleurs10.py so the numbers are directly comparable to
the older results.

Usage:
    uv run python scripts/eval_omni_inproc.py --model-card omniASR_LLM_300M_v2 \
        --results results/omni_inproc_300m.json
    uv run python scripts/eval_omni_inproc.py --model-card omniASR_LLM_1B_v2 \
        --results results/omni_inproc_1b.json

Output:
    results/<file>.json   (n, mean_cer_m1, mean_cer_m2, rows, model_card,
                           device, load_s, gpu_mem_mb)
    console: device actually used + per-clip + mean CER table
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

# ── CER normalization — VERBATIM from scripts/eval_omni_fleurs10.py ──
_BN = {"০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
       "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"}
_WN = {"একশ": "100", "একশো": "100", "এক হাজার": "1000", "একহাজার": "1000"}


def base_norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).strip()
    s = re.sub(r"[।,.!?‌‍]", "", s)
    return re.sub(r"\s+", " ", s)


def num_norm(s: str) -> str:
    for w, d in _WN.items():
        s = s.replace(w, d)
    for b, e in _BN.items():
        s = s.replace(b, e)
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    n, m = len(r), len(h)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        p = d[0]
        d[0] = i
        for j in range(1, m + 1):
            c = d[j]
            co = 0 if r[i - 1] == h[j - 1] else 1
            d[j] = min(d[j] + 1, d[j - 1] + 1, p + co)
            p = c
    return d[m] / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--demo-dir", type=Path,
                   default=Path("test_data/fleurs_demo_10"))
    p.add_argument("--model-card", default="omniASR_LLM_1B_v2",
                   help="Omnilingual model card (e.g. omniASR_LLM_300M_v2, "
                        "omniASR_LLM_1B_v2)")
    p.add_argument("--omni-lang", default="ben_Beng",
                   help="Omnilingual language code (LLM variants need "
                        "{code}_{script}, e.g. ben_Beng)")
    p.add_argument("--device", default="cuda",
                   help='"cuda" or "cpu"')
    p.add_argument("--results", type=Path, default=None,
                   help="Output JSON (default results/omni_inproc_"
                        "<card>.json)")
    p.add_argument("--max", type=int, default=0,
                   help="Only first N clips (0 = all)")
    args = p.parse_args()

    refs_path = args.demo_dir / "references.json"
    audio_dir = args.demo_dir / "audio"
    if not refs_path.exists() or not audio_dir.is_dir():
        print(f"ERROR: {args.demo_dir} missing references.json or audio/")
        sys.exit(1)

    refs = {r["file"]: r["transcript"]
            for r in json.loads(refs_path.read_text(encoding="utf-8"))}
    files = sorted(refs)
    if args.max and args.max > 0:
        files = files[:args.max]

    # Heavy imports AFTER arg parsing so --help is instant.
    import soundfile as sf
    import torch

    from asr_pipeline.config import OmnilingualConfig
    from asr_pipeline.engines.omnilingual_engine import OmnilingualEngine
    from asr_pipeline.models import AudioChunk

    print(f"torch {torch.__version__} | cuda available: "
          f"{torch.cuda.is_available()}")

    # max_audio_length_s set generously high — FLEURS clips are short
    # read-speech sentences; we feed each whole file as one chunk.
    cfg = OmnilingualConfig(
        model_card=args.model_card,
        zero_shot_model_card=args.model_card,
        max_audio_length_s=600,
    )

    print(f"Loading Omnilingual [{args.model_card}] on {args.device} ...",
          flush=True)
    t0 = time.time()
    engine = OmnilingualEngine(cfg, device=args.device)
    engine.load()
    load_s = time.time() - t0

    # Definitively report where the model actually landed.
    pipe = engine._pipeline  # noqa: SLF001 — diagnostic
    actual_device = str(getattr(pipe, "device", "unknown"))
    gpu_mem_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"  loaded in {load_s:.1f}s | pipeline.device = {actual_device} "
          f"| torch GPU mem allocated = {gpu_mem_mb:.0f} MiB")
    if actual_device.startswith("cpu"):
        print("  WARNING: model is on CPU — inference will be very slow.")

    rows = []
    t_inf0 = time.time()
    for i, fname in enumerate(files, 1):
        wav = audio_dir / fname
        if not wav.exists():
            print(f"  [{i}/{len(files)}] {fname}: MISSING audio, skip")
            continue
        info = sf.info(str(wav))
        dur = float(info.frames) / float(info.samplerate)
        chunk = AudioChunk(
            chunk_id=i,
            start_s=0.0,
            end_s=dur,
            duration_s=dur,
            waveform_path=wav,
        )
        ct0 = time.time()
        segs = engine.transcribe_chunk(chunk, args.omni_lang)
        ct = time.time() - ct0
        hyp = " ".join(s.text for s in segs).strip()
        ref = refs[fname]
        m1 = cer(base_norm(ref), base_norm(hyp)) * 100
        m2 = cer(num_norm(base_norm(ref)),
                 num_norm(base_norm(hyp))) * 100
        rows.append({"file": fname, "ref": ref, "hyp": hyp,
                     "cer_m1": m1, "cer_m2": m2,
                     "audio_s": round(dur, 2),
                     "infer_s": round(ct, 2)})
        print(f"  [{i}/{len(files)}] {fname}  {dur:5.1f}s audio  "
              f"{ct:5.1f}s infer  M1={m1:5.1f}%  M2={m2:5.1f}%  "
              f"| {hyp[:50]}", flush=True)

    engine.unload()

    if not rows:
        print("No clips scored.")
        sys.exit(1)

    n = len(rows)
    mean_m1 = sum(r["cer_m1"] for r in rows) / n
    mean_m2 = sum(r["cer_m2"] for r in rows) / n
    infer_total = time.time() - t_inf0

    print(f"\n{'=' * 64}")
    print(f"Omnilingual {args.model_card} — FLEURS Bengali, n={n}")
    print(f"  load {load_s:.1f}s | inference {infer_total:.1f}s "
          f"| device {actual_device} | GPU {gpu_mem_mb:.0f} MiB")
    print(f"{'=' * 64}")
    print(f"{'file':>10} {'M1':>8} {'M2(+num)':>10}")
    for r in rows:
        print(f"{r['file']:>10} {r['cer_m1']:7.1f}% {r['cer_m2']:9.1f}%")
    print(f"{'MEAN':>10} {mean_m1:7.1f}% {mean_m2:9.1f}%")

    out = args.results or Path(
        f"results/omni_inproc_{args.model_card}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"n": n, "mean_cer_m1": mean_m1, "mean_cer_m2": mean_m2,
         "model_card": args.model_card, "device": actual_device,
         "load_s": load_s, "gpu_mem_mb": gpu_mem_mb,
         "infer_total_s": infer_total, "rows": rows},
        open(out, "w", encoding="utf-8"),
        ensure_ascii=False, indent=2,
    )
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
