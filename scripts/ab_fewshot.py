#!/usr/bin/env python3
"""
A/B: fusion prompt WITHOUT vs WITH the few-shot examples — both at temp=0.

Isolates the few-shot block (the accuracy lever): same shared spine, same
witnesses, same Gemma config (temp=0, repeat_penalty=1.0). BASELINE = the
current prompt with the "WORKED EXAMPLES" block stripped; FEWSHOT = the full
current prompt. Judge whether the generic examples lift recovery (FLUENT-
consensus over garbled LITERAL, numbers, no fabrication) WITHOUT overfitting.

    uv run --no-sync python scripts/ab_fewshot.py test_data/15_part1_mid5min.m4a
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from string import Template

CFG = dict(temperature=0.0, top_k=64, top_p=0.95, min_p=0.0, repeat_penalty=1.0)
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
    ft = FusionTranslator(prompt_lookup_tokens=0)
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

    work = Path(tempfile.mkdtemp(prefix="ab_fs_"))
    ft, aligned, titu_texts, whisper_nbest = build_inputs(args.audio, work, token)

    full = ft._fusion_prompt
    baseline = re.sub(
        r"Two WORKED EXAMPLES.*?Now do the SAME for the real input:\s*",
        "", full, flags=re.DOTALL)
    assert "WORKED EXAMPLES" in full and "WORKED EXAMPLES" not in baseline, \
        "few-shot block markers not found — check the prompt"
    print(f"prompt: baseline {len(baseline)} chars | fewshot {len(full)} chars")

    proc = GGUFProcessor(
        gguf_repo=ft._gguf_repo, gguf_file=ft._gguf_file,
        n_ctx=4096, n_gpu_layers=ft._gguf_n_gpu_layers,
        prompt_lookup_tokens=0, flash_attn=ft._flash_attn, kv_quant=ft._kv_quant)
    if not proc.load():
        print("!! Gemma failed to load"); return 1

    def gen(tpl, tt, whyps):
        prompt = Template(tpl).safe_substitute(
            literal=tt or "(none)", fluent=ft._fmt_fluent(whyps))
        out = proc._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=ft._fusion_max_tokens, seed=SEED, **CFG)
        text = out["choices"][0]["message"].get("content") or ""
        _think, turns = _parse_fusion(text)
        return (turns[0][1] if turns else ""), (turns[0][2] if turns else "")

    rows, differ = [], 0
    for i, (tt, whyps) in enumerate(zip(titu_texts, whisper_nbest)):
        whyps = [h.strip() for h in (whyps or []) if h and h.strip()]
        wt = whyps[0] if whyps else ""
        if not tt and not wt:
            continue
        if not wt:
            continue
        b_bn, b_en = gen(baseline, tt, whyps)
        f_bn, f_en = gen(full, tt, whyps)
        if b_bn != f_bn:
            differ += 1
        rows.append((i, aligned[i], tt, whyps, b_bn, b_en, f_bn, f_en))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1} segments done")

    L = [f"Few-shot A/B (both temp=0) — {args.audio.name}",
         f"segments: {len(rows)} | [bn] differ baseline-vs-fewshot: {differ}",
         "=" * 78]
    for i, seg, tt, whyps, b_bn, b_en, f_bn, f_en in rows:
        L.append(f"\n[{i:03d}] {seg.start_s:7.2f}-{seg.end_s:7.2f}s  {seg.speaker_id}")
        L.append(f"   LITERAL : {tt}")
        L.append(f"   FLUENT  : {' | '.join(whyps)}")
        L.append(f"   BASELINE [bn]: {b_bn}")
        L.append(f"   BASELINE [en]: {b_en}")
        L.append(f"   FEWSHOT  [bn]: {f_bn}")
        L.append(f"   FEWSHOT  [en]: {f_en}")

    rpt = args.audio.with_suffix(".ab_fewshot.txt")
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ report: {rpt}")
    print(f"segments: {len(rows)} | differ: {differ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
