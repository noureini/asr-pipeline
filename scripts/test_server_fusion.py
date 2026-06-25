"""End-to-end smoke test of the server-backed (parallel) fusion path on a
small subset — validates whisper N-best → llama-server batched fusion →
batched role attribution, and times it. Run BEFORE a full pipeline run.

Usage:
    uv run python scripts/test_server_fusion.py <audio> --n 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    from asr_pipeline.config import load_config
    from asr_pipeline.models import AlignedSegment
    from asr_pipeline.fusion import FusionTranslator

    fc = load_config(Path("fusion_production.yaml")).postprocessing.fusion
    ft = FusionTranslator(
        device="cuda", whisper_model=fc.whisper_model,
        gguf_repo=fc.gguf_repo, gguf_file=fc.gguf_file,
        gguf_n_gpu_layers=fc.gguf_n_gpu_layers,
        whisper_batch_size=fc.whisper_batch_size,
        whisper_nbest=fc.whisper_nbest, n_ctx=fc.n_ctx,
        flash_attn=fc.flash_attn, kv_quant=fc.kv_quant,
        fusion_max_tokens=fc.fusion_max_tokens,
        role_max_tokens=fc.role_max_tokens, role_window=fc.role_window,
        prompts_file=fc.prompts_file, fusion_backend=fc.fusion_backend,
        n_parallel=fc.n_parallel, server_port=fc.server_port,
        prompt_lookup_tokens=fc.prompt_lookup_tokens)

    segs = [AlignedSegment(
        segment_id=i, start_s=i * 11.0, end_s=i * 11.0 + 9.0,
        speaker_id=f"SPEAKER_0{i % 2}", language="ben", raw_text="",
        confidence=1.0) for i in range(args.n)]

    print(f">>> backend={fc.fusion_backend} parallel={fc.n_parallel} "
          f"nbest={fc.whisper_nbest}")
    t0 = time.perf_counter()
    out = ft.process(segs, args.audio)
    dt = time.perf_counter() - t0

    print(f"\n=== {len(out)} segments out (from {args.n}) in {dt:.1f}s "
          f"= {dt / args.n:.1f}s/segment ===")
    roles = {}
    for s in out:
        roles[s.speaker_id] = roles.get(s.speaker_id, 0) + 1
    print("roles:", roles)
    for s in out[:5]:
        print(f"  [{s.speaker_id}] {s.english_translation[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
