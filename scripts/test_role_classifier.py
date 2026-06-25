#!/usr/bin/env python3
"""
Validate Stage 6a: the one-shot LLM cluster→role classifier, on real data.

Runs the real pipeline through alignment (spine → titu → community-1 → speaker-
pure segments), then calls the ACTUAL FusionTranslator._decide_roles with the
production Gemma backend. Prints the cluster→role mapping next to each cluster's
aggregated Bengali so you can verify the ENUMERATOR pick by eye (the enumerator
cluster's text should be the QUESTIONS).

Heavy: loads titu + community-1 + the Gemma-3-12B GGUF (sequentially).

Run:
    uv run --no-sync python scripts/test_role_classifier.py test_data/15_part1_mid5min.m4a
"""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    args = ap.parse_args()
    if not args.audio.exists():
        print(f"!! not found: {args.audio}")
        return 1

    import os
    import torch
    from dotenv import load_dotenv
    load_dotenv()
    token = os.environ.get("HF_TOKEN")

    from asr_pipeline.config import PreprocessingConfig, DiarizationConfig
    from asr_pipeline.preprocessor import AudioPreprocessor
    from asr_pipeline.engines.titu_engine import TituEngine
    from asr_pipeline.diarization import create_diarizer
    from asr_pipeline.alignment import (
        align_segments, merge_consecutive_segments, decide_role_map)
    from asr_pipeline.fusion import FusionTranslator
    from asr_pipeline.postprocessor import GGUFProcessor

    work = Path(tempfile.mkdtemp(prefix="role_test_"))
    pp = AudioPreprocessor(PreprocessingConfig(), work)
    _wav, chunks, _ = pp.preprocess(args.audio)
    print(f"spine: {len(chunks)} chunks")

    # titu
    print("[titu] ...")
    eng = TituEngine()
    eng.load()
    asr_segments = eng.transcribe_batch(chunks, "ben")
    eng.unload()
    torch.cuda.empty_cache()

    # community-1 diarization (production settings)
    print("[diar] community-1 ...")
    dcfg = DiarizationConfig(
        model="pyannote/speaker-diarization-community-1",
        min_speakers=2, max_speakers=4, auth_token=token)
    diar = create_diarizer(dcfg, device="cuda", work_dir=work)
    diar.load()
    diar_res = diar.diarize(_wav)
    diar.unload()
    torch.cuda.empty_cache()

    # align → speaker-pure segments
    aligned = merge_consecutive_segments(align_segments(asr_segments, diar_res))
    print(f"aligned: {len(aligned)} speaker-pure segments, "
          f"{diar_res.num_speakers} speakers")

    # aggregated text per cluster (for the eyeball check)
    agg: dict[str, list[str]] = defaultdict(list)
    for s in aligned:
        if (s.raw_text or "").strip():
            agg[s.speaker_id].append(s.raw_text.strip())

    # ── the actual Stage-6a classifier, real Gemma backend ──────────────
    print("[gemma] loading for role classification ...")
    ft = FusionTranslator(prompt_lookup_tokens=0)  # match production FusionConfig default
    proc = GGUFProcessor(
        gguf_repo=ft._gguf_repo, gguf_file=ft._gguf_file,
        n_ctx=ft._n_ctx, n_gpu_layers=ft._gguf_n_gpu_layers,
        prompt_lookup_tokens=ft._prompt_lookup_tokens,
        flash_attn=ft._flash_attn, kv_quant=ft._kv_quant)
    if not proc.load():
        print("!! Gemma failed to load")
        return 1

    mapping = ft._decide_roles(
        aligned,
        lambda p: proc.generate(p, temperature=ft._temperature,
                                top_p=ft._top_p, max_tokens=8))

    # ── verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROLE CLASSIFIER RESULT")
    print("=" * 70)
    for spk in sorted(agg.keys()):
        role = mapping.get(spk, "??")
        head = " ".join(agg[spk])[:240]
        print(f"\n{spk}  ->  {role}")
        print(f"   text: {head}")
    # Heuristic fallback, run standalone — must agree with the LLM.
    heur = decide_role_map(aligned)
    print("\n" + "-" * 70)
    print("HEURISTIC FALLBACK (আপন-/question signal) — should match the LLM:")
    for spk in sorted(agg.keys()):
        print(f"   {spk} -> {heur.get(spk, '??')}")

    print("\n" + "=" * 70)
    enum = [s for s, r in mapping.items() if r == "ENUMERATOR"]
    heur_enum = [s for s, r in heur.items() if r == "ENUMERATOR"]
    print(f"LLM ENUMERATOR:       {enum}")
    print(f"heuristic ENUMERATOR: {heur_enum}")
    print(f"AGREE: {set(enum) == set(heur_enum)}")
    print("Verify: the ENUMERATOR's text above should be the QUESTIONS "
          "(আপনি বললেন / তার জন্য আপনার কোন ব্যবস্থা / কি বলেন).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
