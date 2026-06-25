"""Validate the BATCHED Whisper second witness on real audio.

Builds a handful of fixed windows over the start of the audio and runs
the real FusionTranslator._run_whisper at a given batch size — confirming
(a) no OOM, (b) it produces Bengali text, (c) the per-segment time vs
batch size. Run with --bs 1 then --bs 8 to see the speedup.

Usage:
    uv run python scripts/test_whisper_manual.py <audio.wav|.m4a> --n 24 --bs 8
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--n", type=int, default=24, help="windows to test")
    ap.add_argument("--win", type=float, default=8.0, help="window secs")
    ap.add_argument("--bs", type=int, default=8, help="batch size")
    args = ap.parse_args()

    import torchaudio
    from asr_pipeline.models import AlignedSegment
    from asr_pipeline.fusion import FusionTranslator

    # info().num_frames is unreliable for m4a via ffmpeg — load and
    # measure the real duration from the waveform.
    wav, sr = torchaudio.load(str(args.audio))
    dur = wav.shape[1] / sr
    print(f"audio duration: {dur:.0f}s ({dur/60:.1f} min)")
    segs = []
    for i in range(args.n):
        st = i * (args.win + 2.0)
        if st + args.win > dur:
            break
        segs.append(AlignedSegment(
            segment_id=i, start_s=st, end_s=st + args.win,
            speaker_id="S0", language="ben", raw_text="", confidence=1.0))
    if not segs:
        print("audio too short for the requested windows", file=sys.stderr)
        return 1

    ft = FusionTranslator(whisper_batch_size=args.bs)
    t0 = time.perf_counter()
    texts = ft._run_whisper(segs, args.audio)
    dt = time.perf_counter() - t0

    nonempty = [t for t in texts if t]
    print(f"\n=== {len(nonempty)}/{len(segs)} produced text in {dt:.1f}s "
          f"(bs={args.bs}) = {dt / len(segs):.2f}s/segment ===")
    for t in nonempty[:6]:
        print("  •", t[:90])
    return 0


if __name__ == "__main__":
    sys.exit(main())
