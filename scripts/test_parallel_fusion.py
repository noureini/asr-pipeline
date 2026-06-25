"""Python-level PARALLEL fusion: N worker processes, each its own Gemma
instance, each fusing a slice of the segments concurrently on the GPU.

This is plain multiprocessing — NOT model-internal batching. The only
constraint is that N model copies must fit in 8GB VRAM (so use a small
model: 4B → 2 workers fit ~7GB; 1B → many fit). Measures wall-clock vs
the number of workers so we can SEE the speedup.

Usage:
    uv run python scripts/test_parallel_fusion.py \
        --gguf-file gemma-3-4b-it-UD-Q4_K_XL.gguf \
        --gguf-repo unsloth/gemma-3-4b-it-GGUF --workers 2 --n 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Real witness pairs (phonetic ‖ fluent) — replicated to simulate a batch.
PAIRS = [
    ("আমার ফ্যামিলিতে আমার ছয় জন মানুষ এই ছয় জন মানুষে আমার এক কৃষি কি "
     "কাজের উপরে নির্ভর করে কুনু বছর যদি মাইর যায় তখনি কষ্ট বেড়ে যায়",
     "আমার ফ্যামিলিতে ছয় জন মানুষ। এই ছয় জন মানুষ আমার কৃষি কাজের উপরে "
     "নির্ভর করে। কোনো বছর যদি ফসল নষ্ট হয়, তখনই কষ্ট বেড়ে যায়।"),
    ("আমার এখন বর্তমানে ডেইলি আমার দুইশ থেকে আড়াইশো টাকার সাউল লাগে",
     "আমার এখন ডেইলি দুইশো থেকে আড়াইশো টাকার চাল লাগে।"),
]


def _worker(args_tuple):
    """Runs in a separate process: load its OWN Gemma, fuse its slice."""
    repo, gfile, prompt_tpl, pairs = args_tuple
    from string import Template
    from asr_pipeline.postprocessor import GGUFProcessor
    proc = GGUFProcessor(gguf_repo=repo, gguf_file=gfile, n_ctx=2048,
                         n_gpu_layers=-1, flash_attn=True, kv_quant=True)
    if not proc.load():
        return [("LOAD_FAIL", "")]
    tpl = Template(prompt_tpl)
    out = []
    for ph, fl in pairs:
        ans = proc.generate(tpl.safe_substitute(phonetic=ph, fluent=fl),
                            temperature=0.0, max_tokens=384)
        out.append(ans)
    proc.unload()
    return out


def main() -> int:
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-repo", default="unsloth/gemma-3-4b-it-GGUF")
    ap.add_argument("--gguf-file",
                    default="gemma-3-4b-it-UD-Q4_K_XL.gguf")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n", type=int, default=16, help="total segments")
    args = ap.parse_args()

    from asr_pipeline.fusion import load_prompts
    prompt_tpl = load_prompts("config/prompts/bn_survey.yaml")["fusion_prompt"]

    # Build n segments by cycling the real pairs.
    segs = [PAIRS[i % len(PAIRS)] for i in range(args.n)]
    # Split into `workers` contiguous slices.
    W = max(1, args.workers)
    slices = [segs[i::W] for i in range(W)]
    tasks = [(args.gguf_repo, args.gguf_file, prompt_tpl, sl)
             for sl in slices]

    print(f">>> {args.n} segments across {W} parallel worker(s), "
          f"model={args.gguf_file}")
    ctx = mp.get_context("spawn")          # CUDA needs spawn, not fork
    t0 = time.perf_counter()
    with ctx.Pool(processes=W) as pool:
        results = pool.map(_worker, tasks)
    dt = time.perf_counter() - t0

    n_ok = sum(1 for r in results for x in r if x and "LOAD_FAIL" not in x)
    print(f"\n=== {args.n} segments, {W} workers: {dt:.1f}s wall "
          f"({dt / args.n:.1f}s/segment effective) ===")
    print(f"    {n_ok} segments produced output")
    # show one sample
    for r in results:
        if r and "LOAD_FAIL" not in r[0]:
            print("    sample:", r[0][:110].replace(chr(10), " "))
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
