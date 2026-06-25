"""
A/B chunking isolation test — is our VAD merge heuristic the bug?

Same Omnilingual 1B, same audio region, three segmentation strategies:

  A CURRENT  : full pipeline AudioPreprocessor — loudness + spectral
               denoise + Silero-VAD-on-denoised + our merge rule
               (close chunk if gap>3s OR span>30s).  [the pipeline]
  B WHISPERX : raw 16k (no denoise) + Silero-VAD-on-raw + WhisperX
               merge_chunks (greedy pack to 30s, NO gap rule, never
               cut inside a segment).               [the proposed fix]
  C ISOLATE  : raw 16k + Silero-VAD-on-raw + OUR gap>3s merge rule.
               (B vs C isolates the merge rule; C vs A isolates raw
               vs denoised.)

Prints, per strategy: #chunks, chunk-duration spread (fragmentation),
and the concatenated transcript — so we can see whether the salad
becomes coherent.

Usage:
  uv run python scripts/ab_chunking_test.py --stem <STEM> \
      --region part1:3300-3520
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIO = Path("for testing (Noureini)")
OUT = Path("outputs/real_eval_inproc")


def whisperx_merge(segs, chunk_size=30.0):
    """Faithful port of whisperx/vads/vad.py:Vad.merge_chunks —
    greedy pack to chunk_size, close only when span would exceed it,
    at the previous segment boundary; never split inside a segment.
    Over-long single segments are hard-split (WhisperX does a
    low-score min-cut upstream; even split is a fine approximation)."""
    # pre-split any segment longer than chunk_size
    norm = []
    for s, e in segs:
        if e - s <= chunk_size:
            norm.append((s, e))
        else:
            n = math.ceil((e - s) / chunk_size)
            step = (e - s) / n
            for k in range(n):
                norm.append((s + k * step, min(e, s + (k + 1) * step)))
    if not norm:
        return []
    out = []
    cs, ce = norm[0]
    for s, e in norm[1:]:
        if e - cs > chunk_size and ce - cs > 0:
            out.append((cs, ce))
            cs = s
        ce = e
    out.append((cs, ce))
    return out


def our_merge(segs, max_dur=30.0):
    """Our current heuristic: merge while span<=max_dur AND gap<3s,
    else close the chunk early (the suspected fragmenter)."""
    if not segs:
        return []
    out = []
    gs, ge = segs[0]
    for s, e in segs[1:]:
        gap = s - ge
        if (e - gs) <= max_dur and gap < 3.0:
            ge = e
        else:
            out.append((gs, ge))
            gs, ge = s, e
    out.append((gs, ge))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--region", default="part1:3300-3520",
                    help="partN:START-END seconds (within that part)")
    ap.add_argument("--omni-lang", default="ben_Beng")
    args = ap.parse_args()

    import torch  # noqa
    import torchaudio

    from asr_pipeline.config import load_config
    from asr_pipeline.engines.omnilingual_engine import OmnilingualEngine
    from asr_pipeline.models import AudioChunk
    from asr_pipeline.preprocessor import AudioPreprocessor

    cfg = load_config(Path("omni_test_1b.yaml"))
    cfg.omnilingual.max_audio_length_s = 60

    part, rng = args.region.split(":")
    rs, re_ = (float(x) for x in rng.split("-"))
    chunk_files = sorted((OUT / args.stem).glob("*.chunks.jsonl"))
    pidx = int(part.replace("part", "")) - 1
    stem = chunk_files[pidx].name[: -len(".chunks.jsonl")]
    m4a = AUDIO / f"{stem}.m4a"

    tmp = Path(tempfile.mkdtemp(prefix="abchunk_"))
    region = tmp / "region.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(rs),
                    "-to", str(re_), "-i", str(m4a), "-ac", "1",
                    "-ar", "16000", str(region)], check=True)
    print(f"Region {part} {rs:.0f}-{re_:.0f}s ({re_-rs:.0f}s) "
          f"from {m4a.name}")

    print("Loading Omnilingual 1B ...", flush=True)
    eng = OmnilingualEngine(cfg.omnilingual, device="cuda")
    eng.load()
    print(f"  device={getattr(eng._pipeline,'device','?')}", flush=True)

    wf, sr = torchaudio.load(str(region))
    wf1 = wf.squeeze()

    def silero_on(wave):
        from silero_vad import get_speech_timestamps, load_silero_vad
        m = load_silero_vad()
        ts = get_speech_timestamps(
            wave, m, threshold=cfg.preprocessing.vad.threshold,
            min_speech_duration_ms=cfg.preprocessing.vad
            .min_speech_duration_ms,
            min_silence_duration_ms=cfg.preprocessing.vad
            .min_silence_duration_ms, sampling_rate=16000)
        return [(t["start"] / 16000, t["end"] / 16000) for t in ts]

    def transcribe_spans(spans, src_wave):
        txt = []
        for i, (s, e) in enumerate(spans):
            seg = src_wave[int(s * 16000):int(e * 16000)].unsqueeze(0)
            p = tmp / f"c_{i}.wav"
            torchaudio.save(str(p), seg, 16000)
            ch = AudioChunk(chunk_id=i, start_s=s, end_s=e,
                            duration_s=e - s, waveform_path=p)
            segs = eng.transcribe_chunk(ch, args.omni_lang)
            txt.append(" ".join(x.text for x in segs).strip())
        return " ".join(t for t in txt if t).strip()

    def report(name, spans, transcript):
        ds = [e - s for s, e in spans]
        sp = (f"min{min(ds):.0f} med{sorted(ds)[len(ds)//2]:.0f} "
              f"max{max(ds):.0f}") if ds else "-"
        print("\n" + "=" * 72)
        print(f"[{name}]  {len(spans)} chunks  dur({sp})s")
        print(transcript[:1100])

    # A CURRENT pipeline (loudness+denoise+SileroVAD-on-denoised+ourmerge)
    ppA = AudioPreprocessor(cfg.preprocessing, tmp / "A")
    (tmp / "A").mkdir(exist_ok=True)
    _, A_chunks, _ = ppA.preprocess(region)
    A_txt = []
    for ch in A_chunks:
        segs = eng.transcribe_chunk(ch, args.omni_lang)
        A_txt.append(" ".join(x.text for x in segs).strip())
    report("A CURRENT pipeline",
           [(c.start_s, c.end_s) for c in A_chunks],
           " ".join(t for t in A_txt if t).strip())

    # raw-VAD shared by B and C
    vad_raw = silero_on(wf1)

    # B WhisperX merge on raw
    report("B WHISPERX (raw+greedy30, no gap rule)",
           whisperx_merge(vad_raw, 30.0),
           transcribe_spans(whisperx_merge(vad_raw, 30.0), wf1))

    # C isolation: raw + OUR gap rule
    report("C ISOLATE (raw + our gap>3s merge)",
           our_merge(vad_raw, 30.0),
           transcribe_spans(our_merge(vad_raw, 30.0), wf1))

    eng.unload()
    print("\nDONE. B vs C => merge-rule effect. C vs A => raw-vs-denoise.")


if __name__ == "__main__":
    main()
