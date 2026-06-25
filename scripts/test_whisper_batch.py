"""Isolate the batched-Whisper ValueError so we can fix the input format.

Loads the regional Whisper once and tries several batched-input idioms on
synthetic audio, printing the FULL traceback for whichever fails.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    import numpy as np
    import torch
    from transformers import pipeline

    model = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"
    dev = 0 if torch.cuda.is_available() else -1
    print(f"loading {model} on device {dev} ...")
    asr = pipeline("automatic-speech-recognition", model=model,
                   device=dev, chunk_length_s=30)

    sr = 16000
    # MIX of lengths incl. one >30s clip — reproduces the real-run batch
    # that ValueError'd (2s, 35s, 1.5s batched together).
    arrays = [np.random.randn(int(t * sr)).astype("float32") * 0.01
              for t in (2.0, 35.0, 1.5)]

    def gen():
        for a in arrays:
            yield {"raw": a, "sampling_rate": sr}

    # Idiom A: generator of {"raw", "sampling_rate"} + batch_size
    print("\n--- A: generator of dict(raw,sampling_rate), batch_size=2 ---")
    try:
        for i, r in enumerate(asr(gen(), batch_size=2)):
            print(f"  A[{i}]:", repr((r.get('text') or '')[:40]))
    except Exception:
        traceback.print_exc()

    # Idiom B: generator of raw arrays (pipeline infers sampling_rate?)
    print("\n--- B: list of raw arrays, batch_size=2 ---")
    try:
        for i, r in enumerate(asr(arrays, batch_size=2)):
            print(f"  B[{i}]:", repr((r.get('text') or '')[:40]))
    except Exception:
        traceback.print_exc()

    # Idiom C: single dict baseline (known to work)
    print("\n--- C: single dict baseline ---")
    try:
        r = asr({"raw": arrays[0], "sampling_rate": sr})
        print("  C:", repr((r.get('text') or '')[:40]))
    except Exception:
        traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
