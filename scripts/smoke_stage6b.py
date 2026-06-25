#!/usr/bin/env python3
"""
Stage 6b end-to-end smoke test: the ASSEMBLED production fusion on one file.

Exercises everything we just locked — CT2 batched Whisper witness, gemma-rec
sampling (temp=1.0/top_k=64/top_p=0.95/min_p=0.0) + fixed seed, Gemma [en]
translation, cluster→role (greedy), GER guard — via the real
FusionTranslator.process(), and dumps role / [bn] / [en] / flags per segment.

Confirms: CT2 path ran, no crash, सुরমা নদী recovered ([002]-type), no number
hallucination, roles assigned (ENUMERATOR/RESPONDENT). Reproducible (fixed seed).

    uv run --no-sync python scripts/smoke_stage6b.py test_data/15_part1_mid5min.m4a
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    args = ap.parse_args()

    import os
    import torch
    from dotenv import load_dotenv
    load_dotenv()
    token = os.environ.get("HF_TOKEN")

    from asr_pipeline.config import PreprocessingConfig, DiarizationConfig
    from asr_pipeline.preprocessor import AudioPreprocessor
    from asr_pipeline.engines.titu_engine import TituEngine
    from asr_pipeline.diarization import create_diarizer
    from asr_pipeline.alignment import align_segments, merge_consecutive_segments
    from asr_pipeline.fusion import FusionTranslator

    work = Path(tempfile.mkdtemp(prefix="smoke6b_"))
    pp = AudioPreprocessor(PreprocessingConfig(), work)
    wav_path, chunks, _ = pp.preprocess(args.audio)
    print(f"spine: {len(chunks)} chunks")

    eng = TituEngine(); eng.load()
    asr = eng.transcribe_batch(chunks, "ben")
    eng.unload(); torch.cuda.empty_cache()

    dcfg = DiarizationConfig(model="pyannote/speaker-diarization-community-1",
                             min_speakers=2, max_speakers=4, auth_token=token)
    diar = create_diarizer(dcfg, device="cuda", work_dir=work)
    diar.load(); res = diar.diarize(wav_path); diar.unload()
    torch.cuda.empty_cache()

    aligned = merge_consecutive_segments(align_segments(asr, res))
    print(f"aligned: {len(aligned)} speaker-pure segments")

    # The REAL production fusion (all defaults = CT2 + gemma-rec + seed).
    ft = FusionTranslator()
    processed = ft.process(aligned, wav_path)

    reviews = sum(1 for p in processed if p.needs_review)
    abstains = sum(1 for p in processed if "[অস্পষ্ট]" in (p.corrected_text or ""))
    roles = {}
    for p in processed:
        roles[p.speaker_id] = roles.get(p.speaker_id, 0) + 1

    L = [f"Stage 6b smoke — {args.audio.name}",
         f"segments: {len(processed)} | needs_review: {reviews} | "
         f"number_abstain: {abstains} | roles: {roles}",
         "=" * 78]
    for p in processed:
        flags = []
        if p.needs_review:
            flags.append(f"review:{p.review_reason}")
        if p.fusion_fallback:
            flags.append("FALLBACK")
        L.append(f"\n[{p.segment_id:03d}] {p.start_s:7.2f}-{p.end_s:7.2f}s  "
                 f"{p.speaker_id}  {'  '.join(flags)}")
        L.append(f"   LITERAL: {p.witness_literal}")
        L.append(f"   [bn]   : {p.corrected_text}")
        L.append(f"   [en]   : {p.english_translation}")

    rpt = args.audio.with_suffix(".smoke6b.txt")
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ report: {rpt}")
    print(f"segments: {len(processed)} | needs_review: {reviews} | "
          f"number_abstain: {abstains} | roles: {roles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
