"""Probe whether titu (NeMo FastConformer-CTC) emits WORD-level timestamps.

Step 1 of the WhisperX-style re-architecture: confirm we can get word
timestamps from titu directly (cheapest source — no extra model) before
wiring it into the engine. Prints the exact NeMo timestamp structure so the
extraction can be written to reality, not a guess.

Usage (needs the NeMo venv):
    uv run --no-sync python scripts/probe_titu_timestamps.py
    uv run --no-sync python scripts/probe_titu_timestamps.py path/to/clip.wav
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")


def _make_wav(src: str, dst: str, secs: int = 20) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "0", "-t", str(secs), "-i", src,
         "-ar", "16000", "-ac", "1", dst],
        check=True, capture_output=True,
    )


def main() -> int:
    wav = sys.argv[1] if len(sys.argv) > 1 else None
    if not wav:
        wav = "/tmp/titu_probe.wav"
        src = "test_data/15_part1_mid5min.m4a"
        if not Path(src).exists():
            print(f"no test audio at {src}; pass a wav path as an argument",
                  file=sys.stderr)
            return 1
        _make_wav(src, wav, 20)
        print(f"made 20s probe clip: {wav}")

    from asr_pipeline.engines.titu_engine import TituEngine
    eng = TituEngine(device="cuda", batch_size=1)
    eng.load()
    m = eng._model

    print("\n=== plain transcribe (baseline) ===")
    plain = m.transcribe([wav], batch_size=1)
    h0 = plain[0]
    print("text:", (h0 if isinstance(h0, str) else getattr(h0, "text", h0)))

    print("\n=== transcribe(timestamps=True) ===")
    try:
        out = m.transcribe([wav], batch_size=1, timestamps=True)
    except Exception as e:
        print(f"timestamps=True NOT accepted ({type(e).__name__}: {e})")
        print("→ will need change_decoding_strategy(compute_timestamps=True) "
              "or a separate wav2vec2 aligner")
        return 2

    h = out[0]
    print("hyp type:", type(h).__name__)
    ts = getattr(h, "timestamp", None)
    print("timestamp container type:", type(ts).__name__)
    if isinstance(ts, dict):
        print("timestamp keys:", list(ts.keys()))
        words = ts.get("word") or ts.get("words")
        if words:
            print(f"\n✓ {len(words)} WORD entries — first 6 (this is the "
                  "structure I'll extract):")
            for w in words[:6]:
                print("   ", w)
        else:
            print("no 'word' key — keys present:", list(ts.keys()))
            for k in list(ts.keys())[:1]:
                print(f"sample of '{k}':", (ts[k][:3] if ts[k] else ts[k]))
    else:
        print("timestamp repr:", repr(ts)[:400])

    # ── verify the ENGINE path (transcribe_batch → ASRSegment.words) ──
    print("\n=== engine path: TituEngine.transcribe_batch → words ===")
    from asr_pipeline.models import AudioChunk
    chunk = AudioChunk(chunk_id=0, start_s=100.0, end_s=120.0,
                       duration_s=20.0, waveform_path=Path(wav))
    segs = eng.transcribe_batch([chunk], "ben")
    seg = segs[0]
    print(f"segment has {len(seg.words)} words")
    if seg.words:
        w0 = seg.words[0]
        print(f"first word: {w0.word!r}  start={w0.start_s:.2f}  "
              f"end={w0.end_s:.2f}")
        ok = abs(w0.start_s - 100.0) < 25.0 and w0.start_s >= 100.0
        print("absolute offset applied (start ≈ 100 + rel):",
              "✓ YES" if ok else "✗ NO — offset bug")
    else:
        print("✗ no words on the segment — engine extraction failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
