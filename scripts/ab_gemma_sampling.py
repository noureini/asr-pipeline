#!/usr/bin/env python3
"""
A/B Gemma-3 sampling config for the fusion task, on the REAL fusion prompts:

   GREEDY (current)      : temperature=0.0, top_p=1.0
   GEMMA-REC (Gemma team): temperature=1.0, top_k=64, top_p=0.95, repeat_penalty=1.0

Same shared spine, same witnesses, same fusion prompt (built via the production
FusionTranslator._fmt_fluent + _fusion_prompt, parsed with _parse_fusion). Only
the decoding config differs. A fixed seed makes the sampled arm reproducible.

Judge: does GEMMA-REC produce MORE FAITHFUL reconciliation (fewer over-trust
errors, recovers more correct words) — or does temp=1.0 drift/hallucinate on a
survey faithfulness task? And how often does it even differ from greedy?

NOTE: GEMMA-REC is non-deterministic in production (seed varies); greedy is
reproducible. Weigh quality gain vs reproducibility for a survey deliverable.

    uv run --no-sync python scripts/ab_gemma_sampling.py test_data/15_part1_mid5min.m4a
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from string import Template

# Clean temperature-only isolation: GREEDY is identical to GEMMA_REC except
# temperature=0 (same top_k/top_p/min_p/repeat_penalty), so the ONLY variable
# is the temperature. (Previously GREEDY omitted those → llama-cpp defaults
# like repeat_penalty=1.1 leaked in, muddying the comparison.)
GREEDY = dict(temperature=0.0, top_k=64, top_p=0.95, min_p=0.0, repeat_penalty=1.0)
# Exact Gemma-team config — min_p=0.0 matters (llama-cpp defaults it to 0.05).
GEMMA_REC = dict(temperature=1.0, top_k=64, top_p=0.95, min_p=0.0, repeat_penalty=1.0)
SEED = 1234


def build_inputs(audio, work, token):
    import torch
    from asr_pipeline.config import PreprocessingConfig, DiarizationConfig
    from asr_pipeline.preprocessor import AudioPreprocessor
    from asr_pipeline.engines.titu_engine import TituEngine
    from asr_pipeline.diarization import create_diarizer
    from asr_pipeline.alignment import align_segments, merge_consecutive_segments
    from asr_pipeline.fusion import FusionTranslator

    pp = AudioPreprocessor(PreprocessingConfig(), work)
    wav_path, chunks, _ = pp.preprocess(audio)
    eng = TituEngine(); eng.load()
    asr = eng.transcribe_batch(chunks, "ben")
    eng.unload(); torch.cuda.empty_cache()
    dcfg = DiarizationConfig(model="pyannote/speaker-diarization-community-1",
                             min_speakers=2, max_speakers=4, auth_token=token)
    diar = create_diarizer(dcfg, device="cuda", work_dir=work)
    diar.load(); res = diar.diarize(wav_path); diar.unload()
    torch.cuda.empty_cache()
    aligned = merge_consecutive_segments(align_segments(asr, res))

    ft = FusionTranslator(prompt_lookup_tokens=0)   # whisper_nbest=3
    whisper_nbest = ft._run_whisper(aligned, wav_path)
    titu_texts = [(s.raw_text or "").strip() for s in aligned]
    return ft, aligned, titu_texts, whisper_nbest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    args = ap.parse_args()

    import os
    from dotenv import load_dotenv
    load_dotenv()
    token = os.environ.get("HF_TOKEN")

    from asr_pipeline.fusion import _parse_fusion
    from asr_pipeline.postprocessor import GGUFProcessor

    work = Path(tempfile.mkdtemp(prefix="ab_gsamp_"))
    ft, aligned, titu_texts, whisper_nbest = build_inputs(args.audio, work, token)

    proc = GGUFProcessor(
        gguf_repo=ft._gguf_repo, gguf_file=ft._gguf_file,
        n_ctx=ft._n_ctx, n_gpu_layers=ft._gguf_n_gpu_layers,
        prompt_lookup_tokens=0, flash_attn=ft._flash_attn, kv_quant=ft._kv_quant)
    if not proc.load():
        print("!! Gemma failed to load"); return 1

    tpl = Template(ft._fusion_prompt)
    maxtok = ft._fusion_max_tokens

    def gen(prompt, cfg):
        out = proc._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=maxtok, seed=SEED, **cfg)
        text = (out["choices"][0]["message"].get("content") or "")
        _think, turns = _parse_fusion(text)
        bn = turns[0][1] if turns else ""
        en = turns[0][2] if turns else ""
        return bn, en

    rows, differ = [], 0
    for i, (tt, whyps) in enumerate(zip(titu_texts, whisper_nbest)):
        whyps = [h.strip() for h in (whyps or []) if h and h.strip()]
        wt = whyps[0] if whyps else ""
        if not tt and not wt:
            continue
        if not wt:
            continue
        prompt = tpl.safe_substitute(literal=tt or "(none)",
                                     fluent=ft._fmt_fluent(whyps))
        g_bn, g_en = gen(prompt, GREEDY)
        r_bn, r_en = gen(prompt, GEMMA_REC)
        if g_bn != r_bn:
            differ += 1
        rows.append((i, aligned[i], tt, whyps, g_bn, g_en, r_bn, r_en))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1} segments done")

    L = [f"Gemma sampling A/B — {args.audio.name}",
         f"GREEDY={GREEDY}   GEMMA_REC={GEMMA_REC}  seed={SEED}",
         f"segments fused: {len(rows)} | [bn] differ greedy-vs-rec: {differ}",
         "=" * 78]
    for i, seg, tt, whyps, g_bn, g_en, r_bn, r_en in rows:
        L.append(f"\n[{i:03d}] {seg.start_s:7.2f}-{seg.end_s:7.2f}s  {seg.speaker_id}")
        L.append(f"   LITERAL : {tt}")
        L.append(f"   FLUENT  : {' | '.join(whyps)}")
        L.append(f"   GREEDY   [bn]: {g_bn}")
        L.append(f"   GREEDY   [en]: {g_en}")
        L.append(f"   GEMMAREC [bn]: {r_bn}")
        L.append(f"   GEMMAREC [en]: {r_en}")

    rpt = args.audio.with_suffix(".ab_gemma_sampling.txt")
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ report: {rpt}")
    print(f"segments: {len(rows)} | differ: {differ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
