"""Profile a Gemma fusion call: prefill vs decode time split.

Decides the right speedup: if PREFILL dominates, multi-segment batching
(amortize the instruction prompt) wins; if DECODE dominates, only
parallel decode (continuous batching) helps. Runs real fusion prompts
with llama.cpp verbose timings ON.

Usage:
    uv run python scripts/profile_gemma.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Real fused witnesses (phonetic ‖ fluent) from the validated segments.
PAIRS = [
    ("আমার ফ্যামিলিতে আমার ছয় জন মানুষ এই ছয় জন মানুষে আমার এক কৃষি কি "
     "কাজের উপরে নির্ভর করে কুনু বছর যদি মাইর যায় তখনি কষ্ট বেড়ে যায়",
     "আমার ফ্যামিলিতে ছয় জন মানুষ। এই ছয় জন মানুষ আমার কৃষি কাজের উপরে "
     "নির্ভর করে। কোনো বছর যদি ফসল নষ্ট হয়, তখনই কষ্ট বেড়ে যায়।"),
    ("বয়স কত মনে আছে কি আমার বয়স সাড়ে ত্রিশ বছর পরে আমি মুসলিম ধর্মের "
     "অনুসারী আপনি বিবাহিত তো",
     "আপনার বয়স কত মনে আছে? আমার বয়স সাড়ে ত্রিশ বছর। আমি মুসলিম ধর্মের "
     "অনুসারী। আপনি বিবাহিত তো?"),
]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--flash", action="store_true",
                    help="enable flash attention (smaller KV, faster)")
    ap.add_argument("--kv-quant", action="store_true",
                    help="store KV cache as q8_0 (half the memory)")
    args = ap.parse_args()

    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
    from asr_pipeline.fusion import load_prompts

    fp = Template(load_prompts("config/prompts/bn_survey.yaml")
                  ["fusion_prompt"])

    path = hf_hub_download(
        repo_id="unsloth/gemma-3-12b-it-GGUF",
        filename="gemma-3-12b-it-UD-Q4_K_XL.gguf", local_files_only=True)
    kw = dict(model_path=path, n_ctx=args.n_ctx, n_gpu_layers=-1,
              flash_attn=args.flash, verbose=True)
    if args.kv_quant:
        from llama_cpp import GGML_TYPE_Q8_0
        kw["type_k"] = GGML_TYPE_Q8_0
        kw["type_v"] = GGML_TYPE_Q8_0
    print(f">>> n_ctx={args.n_ctx} flash={args.flash} "
          f"kv_quant={args.kv_quant}", file=sys.stderr)
    # verbose=True → llama.cpp prints "prompt eval time" (PREFILL) and
    # "eval time" (DECODE) per call, with tokens and tok/s.
    llm = Llama(**kw)

    for i, (ph, fl) in enumerate(PAIRS):
        prompt = fp.safe_substitute(phonetic=ph, fluent=fl)
        print(f"\n========== SEGMENT {i} ==========", file=sys.stderr)
        t0 = time.perf_counter()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=384)
        dt = time.perf_counter() - t0
        usage = out.get("usage", {})
        print(f"[seg {i}] wall={dt:.1f}s  prompt_tokens="
              f"{usage.get('prompt_tokens')}  completion_tokens="
              f"{usage.get('completion_tokens')}", file=sys.stderr)
        print("  ->", (out['choices'][0]['message']['content'] or '')
              .replace('\n', ' ')[:120], file=sys.stderr)
    print("\nRead the llama.cpp 'prompt eval time' (PREFILL) vs "
          "'eval time' (DECODE) lines above for each segment.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
