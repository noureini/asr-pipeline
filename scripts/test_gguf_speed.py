"""Time a fusion call through the REAL GGUFProcessor path (flash+kv).

Mirrors exactly what FusionTranslator._run_fusion does, so if this is
fast (~4s/seg) the pipeline code is fine and a slow run was stale code;
if this is slow (~30s/seg) the bug is in GGUFProcessor itself.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PH = ("আমার ফ্যামিলিতে আমার ছয় জন মানুষ এই ছয় জন মানুষে আমার এক কৃষি কি "
      "কাজের উপরে নির্ভর করে কুনু বছর যদি মাইর যায় তখনি কষ্ট বেড়ে যায়")
FL = ("আমার ফ্যামিলিতে ছয় জন মানুষ। এই ছয় জন মানুষ আমার কৃষি কাজের উপরে "
      "নির্ভর করে। কোনো বছর যদি ফসল নষ্ট হয়, তখনই কষ্ট বেড়ে যায়।")


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO)
    from asr_pipeline.config import load_config
    from asr_pipeline.fusion import load_prompts
    from asr_pipeline.postprocessor import GGUFProcessor

    fc = load_config(Path("fusion_production.yaml")).postprocessing.fusion
    tpl = Template(load_prompts(fc.prompts_file)["fusion_prompt"])

    # EXACTLY how _run_fusion builds it.
    proc = GGUFProcessor(
        gguf_repo=fc.gguf_repo, gguf_file=fc.gguf_file,
        n_ctx=fc.n_ctx, n_gpu_layers=fc.gguf_n_gpu_layers,
        prompt_lookup_tokens=fc.prompt_lookup_tokens,
        flash_attn=fc.flash_attn, kv_quant=fc.kv_quant)
    if not proc.load():
        print("load failed", file=sys.stderr)
        return 1

    prompt = tpl.safe_substitute(phonetic=PH, fluent=FL)
    for i in range(3):
        t0 = time.perf_counter()
        out = proc.generate(prompt, temperature=fc.temperature,
                            top_p=fc.top_p, max_tokens=fc.fusion_max_tokens)
        dt = time.perf_counter() - t0
        print(f"[call {i}] {dt:.1f}s  ->  {out[:90]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
