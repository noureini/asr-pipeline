#!/usr/bin/env python3
"""
Stage 6b evidence: the REAL production fusion on one file, Whisper N-best 3 vs 5.

Builds the production pipeline through alignment (spine → titu → community-1 →
speaker-pure segments), then runs the actual FusionTranslator.process() twice —
whisper_nbest=3 and =5 — and dumps, per segment:
    role | LITERAL (titu) | FLUENT (whisper) | [bn] | [en] | review flags | GER dist

What to judge:
  * reconciliation quality of [bn] (does it pick titu's content + whisper's structure?)
  * Gemma [en] faithfulness (decides the IndicTrans2-vs-Gemma question)
  * number handling ([অস্পষ্ট] abstains, review flags)
  * does N-best=5 give Gemma materially better options than 3?

Heavy: titu + community-1 + Whisper + Gemma-3-12B, run twice.

    uv run --no-sync python scripts/ab_fusion.py test_data/15_part1_mid5min.m4a
"""

from __future__ import annotations

import argparse
import gc
import tempfile
from pathlib import Path


def build_aligned(audio: Path, work: Path, token):
    import torch
    from asr_pipeline.config import PreprocessingConfig, DiarizationConfig
    from asr_pipeline.preprocessor import AudioPreprocessor
    from asr_pipeline.engines.titu_engine import TituEngine
    from asr_pipeline.diarization import create_diarizer
    from asr_pipeline.alignment import align_segments, merge_consecutive_segments

    pp = AudioPreprocessor(PreprocessingConfig(), work)
    wav_path, chunks, _ = pp.preprocess(audio)
    print(f"spine: {len(chunks)} chunks")

    eng = TituEngine()
    eng.load()
    asr = eng.transcribe_batch(chunks, "ben")
    eng.unload()
    torch.cuda.empty_cache()

    dcfg = DiarizationConfig(
        model="pyannote/speaker-diarization-community-1",
        min_speakers=2, max_speakers=4, auth_token=token)
    diar = create_diarizer(dcfg, device="cuda", work_dir=work)
    diar.load()
    res = diar.diarize(wav_path)
    diar.unload()
    torch.cuda.empty_cache()

    aligned = merge_consecutive_segments(align_segments(asr, res))
    print(f"aligned: {len(aligned)} speaker-pure segments")
    return aligned, wav_path


def run_fusion(aligned, wav_path, nbest):
    import torch
    from asr_pipeline.fusion import FusionTranslator
    ft = FusionTranslator(whisper_nbest=nbest, prompt_lookup_tokens=0)
    processed = ft.process(aligned, wav_path)
    abstains = sum(1 for p in processed if "[অস্পষ্ট]" in (p.corrected_text or ""))
    reviews = sum(1 for p in processed if p.needs_review)
    fallbacks = sum(1 for p in processed if p.fusion_fallback)
    dists = [p.ger_min_dist for p in processed if p.ger_min_dist is not None]
    summary = {
        "segments": len(processed), "needs_review": reviews,
        "number_abstain": abstains, "fusion_fallback": fallbacks,
        "mean_ger_dist": (sum(dists) / len(dists)) if dists else None,
    }
    del ft
    gc.collect()
    torch.cuda.empty_cache()
    return processed, summary


def dump(processed, nbest, L):
    L += ["", "=" * 78, f"### Whisper N-best = {nbest}", "=" * 78]
    for p in processed:
        flags = []
        if p.needs_review:
            flags.append(f"review:{p.review_reason}")
        if p.fusion_fallback:
            flags.append("FALLBACK")
        if p.ger_min_dist is not None:
            flags.append(f"gerdist={p.ger_min_dist:.2f}")
        L.append(f"\n[{p.segment_id:03d}] {p.start_s:7.2f}-{p.end_s:7.2f}s  "
                 f"{p.speaker_id}  {'  '.join(flags)}")
        L.append(f"   LITERAL(titu) : {p.witness_literal}")
        L.append(f"   FLUENT(whisp) : {p.witness_fluent}")
        L.append(f"   [bn]          : {p.corrected_text}")
        L.append(f"   [en]          : {p.english_translation}")
        if p.reasoning:
            L.append(f"   think         : {p.reasoning}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    args = ap.parse_args()
    if not args.audio.exists():
        print(f"!! not found: {args.audio}")
        return 1

    import os
    from dotenv import load_dotenv
    load_dotenv()
    token = os.environ.get("HF_TOKEN")

    work = Path(tempfile.mkdtemp(prefix="ab_fusion_"))
    aligned, wav_path = build_aligned(args.audio, work, token)

    print("[fusion] N-best=3 ...")
    p3, s3 = run_fusion(aligned, wav_path, 3)
    print("[fusion] N-best=5 ...")
    p5, s5 = run_fusion(aligned, wav_path, 5)

    L = [f"Stage 6b fusion A/B — {args.audio.name}", ""]
    L.append(f"N-best=3  summary: {s3}")
    L.append(f"N-best=5  summary: {s5}")
    dump(p3, 3, L)
    dump(p5, 5, L)

    rpt = args.audio.with_suffix(".ab_fusion.txt")
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ report: {rpt}")
    print(f"N-best=3: {s3}")
    print(f"N-best=5: {s5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
