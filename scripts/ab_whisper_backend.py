#!/usr/bin/env python3
"""
A/B the Whisper witness backend on the SAME shared spine:
    HF transformers (current production)  vs  CT2/faster-whisper (low-level N-best)

For each, generate Whisper N-best at N=3 and N=5 over the aligned spine spans,
and report: total wall-time, any failures/OOM, and per-segment N-best so we can
judge DIVERSITY (does CT2's sampled N-best give the same (a)/(b)/(c) spread the
fusion prompt relies on?).

The CT2 path uses ctranslate2's Whisper.generate(num_hypotheses=N,
sampling_temperature=...) — sampled, diverse hypotheses, the same behavior the HF
path gets from do_sample+num_return_sequences. If CT2 output looks garbled, that's
MY plumbing, not CT2 — judge by whether the Bengali is sensible.

    uv run --no-sync python scripts/ab_whisper_backend.py test_data/15_part1_mid5min.m4a
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

CT2_DIR = "models/tugstugi_ct2"


def build_aligned(audio, work, token):
    import torch
    from asr_pipeline.config import PreprocessingConfig, DiarizationConfig
    from asr_pipeline.preprocessor import AudioPreprocessor
    from asr_pipeline.engines.titu_engine import TituEngine
    from asr_pipeline.diarization import create_diarizer
    from asr_pipeline.alignment import align_segments, merge_consecutive_segments

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
    print(f"spine: {len(chunks)} chunks -> {len(aligned)} aligned spans")
    return aligned, wav_path


def hf_nbest(aligned, wav_path, nbest):
    """Exact production HF path."""
    from asr_pipeline.fusion import FusionTranslator
    ft = FusionTranslator(whisper_nbest=nbest, prompt_lookup_tokens=0)
    t0 = time.perf_counter()
    out = ft._run_whisper(aligned, wav_path)
    dt = time.perf_counter() - t0
    import torch, gc
    del ft; gc.collect(); torch.cuda.empty_cache()
    return out, dt


def ct2_nbest(aligned, wav_path, nbest):
    """CT2/faster-whisper low-level sampled N-best."""
    import numpy as np
    import torch
    import torchaudio
    import ctranslate2
    from faster_whisper import WhisperModel
    from faster_whisper.tokenizer import Tokenizer

    wm = WhisperModel(CT2_DIR, device="cuda", compute_type="int8_float16")
    fe = wm.feature_extractor
    tok = Tokenizer(wm.hf_tokenizer, multilingual=True,
                    task="transcribe", language="bn")
    prompt = list(tok.sot_sequence) + [tok.no_timestamps]
    n_frames_required = 3000  # Whisper encoder fixed 30 s

    wav, sr = torchaudio.load(str(wav_path))
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000); sr = 16000
    wav = wav[0].numpy().astype(np.float32)

    # 1) feature-extract every valid span (CPU, cheap), pad to 30 s once
    valid_idx, feats_list = [], []
    for i, seg in enumerate(aligned):
        s = max(0, int(seg.start_s * sr)); e = min(len(wav), int(seg.end_s * sr))
        if e - s < int(0.2 * sr):
            continue
        f = np.asarray(fe(wav[s:e]), dtype=np.float32)
        if f.shape[-1] < n_frames_required:
            f = np.pad(f, ((0, 0), (0, n_frames_required - f.shape[-1])))
        else:
            f = f[:, :n_frames_required]
        valid_idx.append(i); feats_list.append(f)

    out = [[""] for _ in aligned]
    items = list(zip(valid_idx, feats_list))   # (orig_idx, feats)

    def run_batch(batch):
        # whisperX-canonical: on CUDA OOM, halve the batch and retry.
        if not batch:
            return
        try:
            sv = ctranslate2.StorageView.from_array(
                np.ascontiguousarray(np.stack([f for _, f in batch])))
            res = wm.model.generate(
                sv, [prompt] * len(batch),
                num_hypotheses=nbest, beam_size=1,
                sampling_temperature=0.5 if nbest > 1 else 1.0,
                sampling_topk=0, return_scores=False)
            for (oi, _), r in zip(batch, res):
                hyps, seen = [], set()
                for ids in r.sequences_ids:
                    txt = tok.decode(ids).strip()
                    if txt and txt not in seen:
                        seen.add(txt); hyps.append(txt)
                out[oi] = hyps or [""]
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower() and len(batch) > 1:
                torch.cuda.empty_cache()
                mid = len(batch) // 2
                run_batch(batch[:mid]); run_batch(batch[mid:])
            else:
                raise

    t0 = time.perf_counter()
    BS = 16  # whisperX default; halves automatically on OOM (N-best)
    for b in range(0, len(items), BS):
        run_batch(items[b:b + BS])
    dt = time.perf_counter() - t0
    del wm; torch.cuda.empty_cache()
    return out, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--show", type=int, default=10, help="segments to print")
    args = ap.parse_args()

    import os
    from dotenv import load_dotenv
    load_dotenv()
    token = os.environ.get("HF_TOKEN")

    work = Path(tempfile.mkdtemp(prefix="ab_wbk_"))
    aligned, wav_path = build_aligned(args.audio, work, token)

    results = {}
    for label, fn in (("HF", hf_nbest), ("CT2", ct2_nbest)):
        for n in (3, 5):
            key = f"{label}-N{n}"
            print(f"[{key}] ...")
            try:
                out, dt = fn(aligned, wav_path, n)
                avg = sum(len(h) for h in out) / max(1, len(out))
                results[key] = {"out": out, "time": dt, "avg_hyps": avg, "err": None}
                print(f"   {dt:.1f}s, avg {avg:.2f} hyps/seg")
            except Exception as ex:
                results[key] = {"out": None, "time": None, "avg_hyps": 0, "err": repr(ex)}
                print(f"   FAILED: {ex!r}")

    L = [f"Whisper backend A/B — {args.audio.name}", "=" * 74, "", "SUMMARY"]
    for k, r in results.items():
        if r["err"]:
            L.append(f"  {k:8} FAILED: {r['err']}")
        else:
            L.append(f"  {k:8} {r['time']:6.1f}s   avg {r['avg_hyps']:.2f} hyps/seg")
    L += ["", "=" * 74, f"PER-SEGMENT N-best (first {args.show})", "=" * 74]
    for i in range(min(args.show, len(aligned))):
        seg = aligned[i]
        L.append(f"\n[{i:03d}] {seg.start_s:7.2f}-{seg.end_s:7.2f}s")
        for k, r in results.items():
            if not r["out"]:
                continue
            L.append(f"  {k}:")
            for j, h in enumerate(r["out"][i]):
                L.append(f"     ({chr(97+j)}) {h}")

    rpt = args.audio.with_suffix(".ab_whisper_backend.txt")
    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ report: {rpt}")
    for k, r in results.items():
        if r["err"]:
            print(f"  {k}: FAILED {r['err']}")
        else:
            print(f"  {k}: {r['time']:.1f}s, {r['avg_hyps']:.2f} hyps/seg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
