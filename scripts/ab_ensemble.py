#!/usr/bin/env python3
"""
3-witness ensemble A/B on the SAME shared spine.

Builds the production Stage-1 shared spine ONCE, then transcribes those identical
chunks with all three candidate witnesses:

    titu-fast   = hishab/titu_stt_bn_fastconformer    (CTC, standard, 1024-tok)
    titu-large  = hishab/titu_stt_bn_conformer_large  (CTC, dialectal, 128-tok)
    whisper     = tugstugi regional Whisper (seq2seq, dialectal)  [models/tugstugi_ct2]

Purpose: qualitatively assess the ENSEMBLE. The key question per chunk is NOT
"who is best" but "what does each witness UNIQUELY contribute, and does any pair
already cover the third?" Watch survey-critical tokens:
  * flood months (ভাদ্র / আষাঢ় / শ্রাবণ) — seasonal timing = core climate data
  * numbers (ward #, counts, years), entities (সুরমা নদী)
  * dialectal fidelity vs hallucination (whisper's �, English leaks, repeats)

If titu-large catches survey-critical tokens that BOTH titu-fast AND whisper miss,
that argues for including it. If whisper already covers titu-large's wins, the
fast+whisper pair stands.

Run:
    uv run --no-sync python scripts/ab_ensemble.py test_data/15_part1_mid5min.m4a
Flags:
    --whisper-path models/tugstugi_ct2   (override CT2 location)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

TITU_MODELS = {
    "titu-fast":  "hishab/titu_stt_bn_fastconformer",
    "titu-large": "hishab/titu_stt_bn_conformer_large",
}


def titu_transcribe(model_id: str, wav_paths: list[Path]) -> list[str]:
    import torch
    import nemo.collections.asr as nemo_asr

    m = nemo_asr.models.ASRModel.from_pretrained(model_id)
    if torch.cuda.is_available():
        m = m.to(torch.device("cuda"))
    m.eval()
    outs = m.transcribe([str(p) for p in wav_paths], batch_size=8)
    texts = [
        (o if isinstance(o, str) else getattr(o, "text", str(o))).strip()
        for o in outs
    ]
    del m
    torch.cuda.empty_cache()
    return texts


def whisper_transcribe(ct2_path: str, wav_paths: list[Path]) -> list[str]:
    import torch
    from faster_whisper import WhisperModel

    m = WhisperModel(ct2_path, device="cuda", compute_type="int8_float16")
    texts = []
    for p in wav_paths:
        segs, _ = m.transcribe(str(p), language="bn", beam_size=5)
        texts.append(" ".join(s.text.strip() for s in segs).strip())
    del m
    torch.cuda.empty_cache()
    return texts


def toks(s: str) -> int:
    return len(s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--whisper-path", default="models/tugstugi_ct2")
    args = ap.parse_args()
    if not args.audio.exists():
        print(f"!! not found: {args.audio}")
        return 1

    from asr_pipeline.config import PreprocessingConfig
    from asr_pipeline.preprocessor import AudioPreprocessor

    work = Path(tempfile.mkdtemp(prefix="ab_ens_"))
    pp = AudioPreprocessor(PreprocessingConfig(), work)
    _wav, chunks, _ns = pp.preprocess(args.audio)
    print(f"shared spine: {len(chunks)} chunks "
          f"({len(pp.speech_timestamps)} speech regions)")
    wav_paths = [c.waveform_path for c in chunks]

    results: dict[str, list[str]] = {}
    for name, mid in TITU_MODELS.items():
        print(f"[{name}] transcribing {len(chunks)} chunks ...")
        results[name] = titu_transcribe(mid, wav_paths)
    print(f"[whisper] transcribing {len(chunks)} chunks ...")
    results["whisper"] = whisper_transcribe(args.whisper_path, wav_paths)

    order = ["titu-fast", "titu-large", "whisper"]
    rpt = args.audio.with_suffix(".ab_ensemble.txt")
    L = [
        f"3-witness ensemble A/B (same shared spine)  —  {args.audio.name}",
        f"spine chunks: {len(chunks)}",
        "=" * 74, "", "SUMMARY",
    ]
    for name in order:
        L.append(f"  {name:12} {sum(toks(t) for t in results[name]):>5} tokens")
    L += ["", "=" * 74,
          "PER-CHUNK   (what does each witness uniquely contribute?)",
          "=" * 74]
    for i, c in enumerate(chunks):
        L.append(f"[{i:03d}] {c.start_s:7.2f}-{c.end_s:7.2f}s")
        for name in order:
            L.append(f"   {name:11}: {results[name][i]}")
        L.append("")

    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ report: {rpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
