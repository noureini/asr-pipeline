#!/usr/bin/env python3
"""
A/B: titu fastconformer vs conformer_large on the SAME shared spine.

Builds the production Stage-1 shared spine ONCE (the real AudioPreprocessor,
so it's byte-identical to what the pipeline feeds titu), then transcribes those
identical chunks with both titu CTC variants. Apples-to-apples — the only
variable is the model.

No gold set exists, so this is for eyeballing. Focus on SURVEY-CRITICAL tokens:
  * numbers (ward numbers, counts, years, months, money)
  * named entities (places, rivers — e.g. সুরমা নদী)
  * dialectal/Sylheti forms (খরি, কুনু, অইলে, টেখা, গর, ফরে)
Count where each model gets a high-value token right vs drops/garbles it.

Run it yourself (no venv auto-sync):
    uv run --no-sync python scripts/ab_titu_variants.py test_data/15_part1_mid5min.m4a
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

# Both are NeMo EncDecCTC models; from_pretrained auto-resolves the class.
MODELS = {
    "fastconformer":   "hishab/titu_stt_bn_fastconformer",    # 18,336 hr YouTube
    "conformer_large": "hishab/titu_stt_bn_conformer_large",  # 4.4K hr (non-telephony)
}


def transcribe_all(model_id: str, wav_paths: list[Path]) -> list[str]:
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


def toks(s: str) -> int:
    return len(s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    args = ap.parse_args()
    if not args.audio.exists():
        print(f"!! not found: {args.audio}")
        return 1

    from asr_pipeline.config import PreprocessingConfig
    from asr_pipeline.preprocessor import AudioPreprocessor

    # Build the real production shared spine once.
    work = Path(tempfile.mkdtemp(prefix="ab_titu_"))
    pp = AudioPreprocessor(PreprocessingConfig(), work)
    _wav, chunks, _ns = pp.preprocess(args.audio)
    print(f"shared spine: {len(chunks)} chunks "
          f"({len(pp.speech_timestamps)} speech regions)")
    wav_paths = [c.waveform_path for c in chunks]

    results: dict[str, list[str]] = {}
    for name, mid in MODELS.items():
        print(f"[{name}] transcribing {len(chunks)} chunks ...")
        results[name] = transcribe_all(mid, wav_paths)

    rpt = args.audio.with_suffix(".ab_titu.txt")
    L = [
        f"titu A/B (same shared spine)  —  {args.audio.name}",
        f"spine chunks: {len(chunks)}",
        "=" * 72, "", "SUMMARY",
    ]
    for name in MODELS:
        L.append(f"  {name:16} {sum(toks(t) for t in results[name]):>5} tokens")
    L += ["", "=" * 72,
          "PER-CHUNK   fast = fastconformer   large = conformer_large",
          "=" * 72]
    for i, c in enumerate(chunks):
        L.append(f"[{i:03d}] {c.start_s:7.2f}-{c.end_s:7.2f}s")
        L.append(f"   fast : {results['fastconformer'][i]}")
        L.append(f"   large: {results['conformer_large'][i]}")
        L.append("")

    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ report: {rpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
