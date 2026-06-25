"""Benchmark vLLM continuous batching for the fusion task on the 8GB GPU.

Runs in the ISOLATED .venv-vllm (AWQ Gemma-3-12B). Fires N fusion prompts
at once; vLLM batches them via PagedAttention. Reports AGGREGATE tok/s —
the number that decides whether vLLM's batching beats llama.cpp's 20 tok/s
single-stream on this hardware.

Usage (from the vLLM venv):
    .venv-vllm/bin/python scripts/bench_vllm.py \
        --model gaunernst/gemma-3-12b-it-int4-awq --n 32
"""
from __future__ import annotations

import argparse
import time

# Minimal fusion prompt (inlined so this stays independent of the main
# asr_pipeline env). Mirrors config/prompts/bn_survey.yaml.
PROMPT = """\
You are reconstructing ONE Bengali utterance from a climate-disaster \
survey in coastal Bangladesh. Reconcile the two ASR readings into one \
clean Bengali line, then translate it to natural English.
PHONETIC: {ph}
FLUENT: {fl}
Produce exactly:
[bn] <clean Bengali>
[en] <natural English, numbers exact>"""

PH = ("আমার ফ্যামিলিতে আমার ছয় জন মানুষ এই ছয় জন মানুষে আমার এক কৃষি কি "
      "কাজের উপরে নির্ভর করে কুনু বছর যদি মাইর যায় তখনি কষ্ট বেড়ে যায়")
FL = ("আমার ফ্যামিলিতে ছয় জন মানুষ। এই ছয় জন মানুষ আমার কৃষি কাজের উপরে "
      "নির্ভর করে। কোনো বছর যদি ফসল নষ্ট হয়, তখনই কষ্ট বেড়ে যায়।")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gaunernst/gemma-3-12b-it-int4-awq")
    ap.add_argument("--n", type=int, default=32, help="concurrent prompts")
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-tokens", type=int, default=384)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    t_load = time.perf_counter()
    llm = LLM(
        model=args.model,
        max_model_len=2048,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,          # save VRAM (no CUDA graphs)
        dtype="float16",
    )
    print(f"model loaded in {time.perf_counter() - t_load:.0f}s")

    prompts = [PROMPT.format(ph=PH, fl=FL) for _ in range(args.n)]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0

    total_out = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"\n=== {args.n} prompts, max_num_seqs={args.max_num_seqs} ===")
    print(f"wall: {dt:.1f}s | output tokens: {total_out} | "
          f"AGGREGATE: {total_out / dt:.1f} tok/s | "
          f"per-prompt: {dt / args.n:.1f}s")
    print(f"(llama.cpp single-stream baseline ~20 tok/s, ~15 s/segment)")
    print("\nsample output:", outs[0].outputs[0].text[:120].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
