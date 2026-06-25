#!/usr/bin/env python3
"""
A/B test for the Stage-1 preprocessing rework (telephony-robust + shared spine).

Compares, on a real file, the CURRENT preprocessing against the PROPOSED one,
and demonstrates the COUPLED shared-spine fusion input (titu + Whisper on the
*same* chunks).

  Arm A  (current / baseline)
      ffmpeg default resample -> 16 kHz
      Silero VAD @ 16 kHz (512-win), threshold 0.50, min_silence 100 ms
      greedy-merge chunks <= 15 s
      titu transcribes each chunk

  Arm B  (proposed)
      rate-adaptive VAD: if source is 8 kHz, run Silero @ 8 kHz (256-win) on
          the NATIVE signal; otherwise @ 16 kHz (512-win)
      recall-leaning: threshold 0.45, min_silence 350 ms
      soxr high-quality resample -> 16 kHz
      greedy-merge chunks <= 15 s  ->  this is the SHARED SPINE
      titu AND Whisper both transcribe the SAME spine chunks (coupled)

What to look for in the report:
  * Arm A vs Arm B titu text  -> did the resample/VAD change help or hurt titu?
  * Arm B titu vs Whisper, per chunk -> this is what coupled fusion will see:
    two witnesses on identical audio, directly comparable token-for-token.

Run it yourself (do not let any tool auto-sync your venv):
    uv run --no-sync python scripts/ab_stage1_preprocess.py /path/to/audio.wav

Optional flags:
    --no-whisper      skip the Whisper arm (titu-only, faster)
    --max-seconds N   only process the first N seconds (quick smoke test)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — adjust if your local paths differ
# ─────────────────────────────────────────────────────────────────────────
TITU_MODEL = "hishab/titu_stt_bn_fastconformer"
WHISPER_CT2_PATH = "models/tugstugi_ct2"   # local CT2 conversion of tugstugi-medium
MAX_CHUNK_S = 15.0

# Arm A (current) VAD params
A_THRESHOLD = 0.50
A_MIN_SILENCE_MS = 100

# Arm B (proposed) VAD params — model-neutral, recall-leaning
B_THRESHOLD = 0.45
B_MIN_SILENCE_MS = 350


# ─────────────────────────────────────────────────────────────────────────
# ffmpeg / ffprobe helpers
# ─────────────────────────────────────────────────────────────────────────
def probe_native_sr(src: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(src)],
        check=True, capture_output=True, text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def decode(src: Path, out_wav: Path, sr: int, soxr: bool, max_seconds: float | None) -> None:
    """Decode src to mono PCM16 WAV at `sr`. soxr=True uses high-quality soxr."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1"]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]
    if soxr:
        cmd += ["-af", f"aresample=resampler=soxr:precision=28", "-ar", str(sr)]
    else:
        cmd += ["-ar", str(sr)]
    cmd += ["-c:a", "pcm_s16le", str(out_wav)]
    subprocess.run(cmd, check=True, capture_output=True)


def read_mono(path: Path) -> np.ndarray:
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("float32")
    return audio


# ─────────────────────────────────────────────────────────────────────────
# Silero VAD
# ─────────────────────────────────────────────────────────────────────────
def load_vad():
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def vad_segments(audio: np.ndarray, sr: int, model, get_ts, threshold: float,
                 min_silence_ms: int) -> list[tuple[float, float]]:
    """Return speech regions as (start_s, end_s). Window size is auto-selected
    by Silero from `sr` (256 @ 8k, 512 @ 16k)."""
    import torch
    wav = torch.from_numpy(audio)
    ts = get_ts(
        wav, model,
        sampling_rate=sr,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=250,
        speech_pad_ms=30,
        return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


# ─────────────────────────────────────────────────────────────────────────
# Greedy-merge chunker (shared spine)
# ─────────────────────────────────────────────────────────────────────────
def greedy_chunks(segs: list[tuple[float, float]], max_s: float) -> list[tuple[float, float]]:
    """Pack consecutive speech regions up to max_s, cutting only at silence.
    Over-long single regions are split evenly (production uses a quietest-point
    split; even split is fine for this A/B)."""
    chunks: list[tuple[float, float]] = []
    cur_start = cur_end = None
    for s, e in segs:
        # split an over-long single region
        while e - s > max_s:
            chunks.append((s, s + max_s))
            s = s + max_s
        if cur_start is None:
            cur_start, cur_end = s, e
        elif e - cur_start <= max_s:
            cur_end = e
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_start is not None:
        chunks.append((cur_start, cur_end))
    return chunks


def cut_chunks(audio16k: np.ndarray, chunks: list[tuple[float, float]],
               work: Path) -> list[Path]:
    paths = []
    for i, (s, e) in enumerate(chunks):
        a = int(s * 16000)
        b = int(e * 16000)
        clip = audio16k[a:b]
        p = work / f"chunk_{i:03d}.wav"
        sf.write(str(p), clip, 16000, subtype="PCM_16")
        paths.append(p)
    return paths


# ─────────────────────────────────────────────────────────────────────────
# ASR
# ─────────────────────────────────────────────────────────────────────────
def titu_transcribe(chunk_paths: list[Path]) -> list[str]:
    import torch
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(TITU_MODEL)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    model.eval()
    outs = model.transcribe([str(p) for p in chunk_paths], batch_size=8)
    texts = []
    for o in outs:
        texts.append((o if isinstance(o, str) else getattr(o, "text", str(o))).strip())
    del model
    torch.cuda.empty_cache()
    return texts


def whisper_transcribe(chunk_paths: list[Path]) -> list[str]:
    import torch
    from faster_whisper import WhisperModel
    model = WhisperModel(WHISPER_CT2_PATH, device="cuda", compute_type="int8_float16")
    texts = []
    for p in chunk_paths:
        segs, _ = model.transcribe(str(p), language="bn", beam_size=5)
        texts.append(" ".join(s.text.strip() for s in segs).strip())
    del model
    torch.cuda.empty_cache()
    return texts


# ─────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────
def toks(s: str) -> int:
    return len(s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--no-whisper", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    src = args.audio
    if not src.exists():
        print(f"!! not found: {src}", file=sys.stderr)
        return 1

    native_sr = probe_native_sr(src)
    print(f"native sample rate: {native_sr} Hz")

    vad_model, get_ts = load_vad()
    tmp = Path(tempfile.mkdtemp(prefix="ab_stage1_"))

    # ---- Arm A: current --------------------------------------------------
    print("\n[Arm A] ffmpeg-default resample -> 16k, VAD@16k (0.50 / 100ms)")
    a16 = tmp / "a_16k.wav"
    decode(src, a16, 16000, soxr=False, max_seconds=args.max_seconds)
    a_audio = read_mono(a16)
    a_segs = vad_segments(a_audio, 16000, vad_model, get_ts, A_THRESHOLD, A_MIN_SILENCE_MS)
    a_chunks = greedy_chunks(a_segs, MAX_CHUNK_S)
    a_dir = tmp / "a_chunks"; a_dir.mkdir()
    a_paths = cut_chunks(a_audio, a_chunks, a_dir)
    print(f"  {len(a_segs)} speech regions -> {len(a_chunks)} chunks")
    a_titu = titu_transcribe(a_paths)

    # ---- Arm B: proposed (shared spine) ----------------------------------
    print("\n[Arm B] rate-adaptive VAD + soxr -> 16k (0.45 / 350ms)  [SHARED SPINE]")
    # VAD on the native-rate signal (8k -> 256-win) so timing is model-correct;
    # timestamps are in SECONDS, so they transfer to the 16k spine unchanged.
    vad_sr = 8000 if native_sr == 8000 else 16000
    bvad = tmp / f"b_vad_{vad_sr}.wav"
    decode(src, bvad, vad_sr, soxr=(vad_sr != native_sr), max_seconds=args.max_seconds)
    b_audio_vad = read_mono(bvad)
    b_segs = vad_segments(b_audio_vad, vad_sr, vad_model, get_ts, B_THRESHOLD, B_MIN_SILENCE_MS)
    b_chunks = greedy_chunks(b_segs, MAX_CHUNK_S)
    # high-quality 16k signal for the actual ASR cut
    b16 = tmp / "b_16k.wav"
    decode(src, b16, 16000, soxr=True, max_seconds=args.max_seconds)
    b_audio = read_mono(b16)
    b_dir = tmp / "b_chunks"; b_dir.mkdir()
    b_paths = cut_chunks(b_audio, b_chunks, b_dir)
    print(f"  VAD@{vad_sr}Hz: {len(b_segs)} speech regions -> {len(b_chunks)} chunks")
    b_titu = titu_transcribe(b_paths)
    b_whisper = None if args.no_whisper else whisper_transcribe(b_paths)

    # ---- Write report ----------------------------------------------------
    rpt = src.with_suffix(".ab_stage1.txt")
    L = []
    L.append(f"A/B Stage-1 preprocessing  —  {src.name}")
    L.append(f"native sample rate: {native_sr} Hz")
    L.append("=" * 72)
    L.append("")
    L.append("SUMMARY")
    L.append(f"  Arm A (current):  {len(a_chunks):>3} chunks, "
             f"{sum(toks(t) for t in a_titu):>5} titu tokens")
    L.append(f"  Arm B (proposed): {len(b_chunks):>3} chunks, "
             f"{sum(toks(t) for t in b_titu):>5} titu tokens"
             + ("" if b_whisper is None
                else f", {sum(toks(t) for t in b_whisper):>5} whisper tokens"))
    L.append("")
    L.append("=" * 72)
    L.append("ARM A  (current preprocessing)  — titu per chunk")
    L.append("=" * 72)
    for i, ((s, e), t) in enumerate(zip(a_chunks, a_titu)):
        L.append(f"[A{i:03d}] {s:7.2f}-{e:7.2f}s  {t}")
    L.append("")
    L.append("=" * 72)
    L.append("ARM B  (proposed, SHARED SPINE)  — titu vs Whisper per chunk")
    L.append("  (this is exactly what coupled fusion will arbitrate)")
    L.append("=" * 72)
    for i, (s, e) in enumerate(b_chunks):
        L.append(f"[B{i:03d}] {s:7.2f}-{e:7.2f}s")
        L.append(f"   titu   : {b_titu[i]}")
        if b_whisper is not None:
            L.append(f"   whisper: {b_whisper[i]}")
        L.append("")

    rpt.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ report written: {rpt}")
    print("  (temp chunks left in:", tmp, ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
