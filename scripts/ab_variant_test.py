"""
A/B: ASR model-variant x chunk-length, WhisperX-clean preprocessing.

One model per process (avoids fairseq2 reload/gang issues). Region is
cut raw 16k mono (no loudness/denoise — matches what Omnilingual
expects; it normalizes internally), Silero-VAD on the raw waveform,
WhisperX greedy merge at a configurable chunk_size, then transcribed.

Run it once per condition:
  A  --model-card omniASR_LLM_1B_v2            --chunk-sec 30
  B  --model-card omniASR_LLM_1B_v2            --chunk-sec 15
  C  --model-card omniASR_LLM_Unlimited_1B_v2  --chunk-sec 15
(A vs B = chunk-length effect; B vs C = variant effect at same length.)
"""
from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

AUDIO = Path("for testing (Noureini)")
OUT = Path("outputs/real_eval_inproc")


def whisperx_merge(segs, chunk_size, wave=None, sr=16000):
    def split_quiet(s, e):
        if e - s <= chunk_size or wave is None:
            if e - s <= chunk_size:
                return [(s, e)]
            n = math.ceil((e - s) / chunk_size)        # fallback
            st = (e - s) / n
            return [(s + k * st, min(e, s + (k + 1) * st))
                    for k in range(n)]
        seg = wave[int(s * sr):int(e * sr)]
        win = max(1, int(0.05 * sr))
        lo, hi = int(len(seg) * 0.3), int(len(seg) * 0.7)
        bi, bv, i = None, None, lo
        while i + win <= hi:
            v = float(seg[i:i + win].pow(2).mean())
            if bv is None or v < bv:
                bv, bi = v, i
            i += win
        mid = (s + e) / 2 if bi is None else s + (bi + win / 2) / sr
        return split_quiet(s, mid) + split_quiet(mid, e)

    norm = []
    for s, e in segs:
        norm.extend(split_quiet(s, e))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--region", default="part1:3300-3520")
    ap.add_argument("--model-card", required=True)
    ap.add_argument("--chunk-sec", type=float, default=30.0)
    ap.add_argument("--omni-lang", default="ben_Beng")
    args = ap.parse_args()

    import torch  # noqa
    import torchaudio

    from asr_pipeline.config import load_config
    from asr_pipeline.engines.omnilingual_engine import OmnilingualEngine
    from asr_pipeline.models import AudioChunk

    cfg = load_config(Path("omni_test_1b.yaml"))
    cfg.omnilingual.model_card = args.model_card
    cfg.omnilingual.max_audio_length_s = max(45, args.chunk_sec + 5)

    part, rng = args.region.split(":")
    rs, re_ = (float(x) for x in rng.split("-"))
    cfiles = sorted((OUT / args.stem).glob("*.chunks.jsonl"))
    stem = cfiles[int(part.replace("part", "")) - 1].name[
        : -len(".chunks.jsonl")]
    m4a = AUDIO / f"{stem}.m4a"

    tmp = Path(tempfile.mkdtemp(prefix="abvar_"))
    region = tmp / "r.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(rs),
                    "-to", str(re_), "-i", str(m4a), "-ac", "1",
                    "-ar", "16000", str(region)], check=True)

    print(f"=== {args.model_card} | chunk<= {args.chunk_sec:.0f}s | "
          f"{part} {rs:.0f}-{re_:.0f}s ===", flush=True)
    eng = OmnilingualEngine(cfg.omnilingual, device="cuda")
    eng.load()

    wf, _ = torchaudio.load(str(region))
    wf1 = wf.squeeze()

    from silero_vad import get_speech_timestamps, load_silero_vad
    vm = load_silero_vad()
    ts = get_speech_timestamps(
        wf1, vm, threshold=cfg.preprocessing.vad.threshold,
        min_speech_duration_ms=cfg.preprocessing.vad
        .min_speech_duration_ms,
        min_silence_duration_ms=cfg.preprocessing.vad
        .min_silence_duration_ms, sampling_rate=16000)
    vad = [(t["start"] / 16000, t["end"] / 16000) for t in ts]
    spans = whisperx_merge(vad, args.chunk_sec, wf1)

    out = []
    for i, (s, e) in enumerate(spans):
        seg = wf1[int(s * 16000):int(e * 16000)].unsqueeze(0)
        p = tmp / f"c{i}.wav"
        torchaudio.save(str(p), seg, 16000)
        ch = AudioChunk(chunk_id=i, start_s=s, end_s=e,
                        duration_s=e - s, waveform_path=p)
        r = eng.transcribe_chunk(ch, args.omni_lang)
        out.append(" ".join(x.text for x in r).strip())
    eng.unload()

    ds = [e - s for s, e in spans]
    sp = (f"min{min(ds):.0f} med{sorted(ds)[len(ds)//2]:.0f} "
          f"max{max(ds):.0f}") if ds else "-"
    print(f"chunks={len(spans)} dur({sp})s")
    print((" ".join(t for t in out if t).strip())[:1400])

    # ── boundary-watch: show the join at each chunk edge ──
    print("\n-- chunk-boundary joins (…tail-of-N | head-of-N+1…) --")
    for i in range(len(out) - 1):
        tail = " ".join(out[i].split()[-6:])
        head = " ".join(out[i + 1].split()[:6])
        print(f"  ⟂{i}->{i+1} @{spans[i][1]:.1f}s: …{tail}  |  "
              f"{head}…")


if __name__ == "__main__":
    main()
