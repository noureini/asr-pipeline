"""4B vs 12B fusion quality A/B on KNOWN segments, at f16 KV (max fidelity).

Runs the production fusion prompt through each model on the segments where
we know the right answer (Ward, six people, আড়াইশো=250, age). Both models
get identical input and f16 KV + eager (no kv-quant, no flash) so this is a
pure model-quality comparison — you judge the Bengali + translation.

Usage:
    uv run python scripts/quality_ab.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# (label, witness text from real titu output, what the right answer is)
KNOWN = [
    ("WARD vs WORLD",
     "টা ক্ষুদ্র পৌরসভার ঠিক আছে এই এলাকাতে বেশি ভাগ গরিবের পৌরসভার পরে ও "
     "নামার ওয়ারাল্ড নাম কত ভাই এটা এটা এক নম্বর ওয়ার্ড এটাই গরিবের চলাফেরা "
     "কেউ মাছ মারে কেউ দিন মজুরি খামও যায় আমি কৃষি কাজ করি ধানের ফসলের",
     "should say 'ward number one' (NOT world)"),
    ("SIX PEOPLE + crop",
     "আমার ফ্যামিলিতে আমাকে নিয়ে ছয় মানুষ এই ছয় জন মানুষের আমার এক কৃষি "
     "কি কাজের উপরে নির্ভর করে কোন বছর যদি মাইর যায় তখনই কষ্ট বেড়ে যায়",
     "6 people; 'crop destroyed' (মাইর যায়)"),
    ("আড়াইশো = 250",
     "আমার এখন বর্তমানে ডেইলি আমার দুইশ থেকে আড়াইশো টাকার সাউল লাগে মাছ চাষ "
     "লাগাইলে সবজি লাগাইলে আমার হাজার টাকার উপরে রুজি হয়",
     "'200 to 250 taka' (আড়াইশো=250, NOT 2.5)"),
    ("AGE সাড়ে ত্রিশ",
     "বয়স কত মনে আছে কি আমার বয়স সাড়ে ত্রিশ বছর পরে আমি মুসলিম ধর্মের "
     "অনুসারী আপনি বিবাহিত তো",
     "age 'thirty-and-a-half' (সাড়ে ত্রিশ=30.5)"),
]

MODELS = [
    ("12B", "unsloth/gemma-3-12b-it-GGUF", "gemma-3-12b-it-UD-Q4_K_XL.gguf"),
    ("4B", "unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-UD-Q4_K_XL.gguf"),
]


def main() -> int:
    from asr_pipeline.postprocessor import GGUFProcessor
    from asr_pipeline.fusion import load_prompts, _parse_bn_en
    tpl = Template(load_prompts("config/prompts/bn_survey.yaml")
                   ["fusion_prompt"])

    results = {}  # label -> {model: (bn,en)}
    for mname, repo, gfile in MODELS:
        print(f"\n### loading {mname} ({gfile}) — f16 KV, eager ###",
              file=sys.stderr)
        proc = GGUFProcessor(gguf_repo=repo, gguf_file=gfile, n_ctx=2048,
                             n_gpu_layers=-1, flash_attn=False,
                             kv_quant=False)   # max fidelity
        if not proc.load():
            print(f"  {mname} failed to load", file=sys.stderr)
            continue
        for label, witness, _ in KNOWN:
            prompt = tpl.safe_substitute(phonetic=witness, fluent=witness)
            ans = proc.generate(prompt, temperature=0.0, max_tokens=384)
            bn, en = _parse_bn_en(ans)
            results.setdefault(label, {})[mname] = (bn, en)
        proc.unload()

    print("\n" + "=" * 78)
    print("QUALITY A/B — f16 KV (max fidelity). YOU judge.")
    print("=" * 78)
    for label, witness, expected in KNOWN:
        print(f"\n■ {label}   (expected: {expected})")
        for mname, _, _ in MODELS:
            bn, en = results.get(label, {}).get(mname, ("—", "—"))
            print(f"  [{mname}] bn: {bn[:100]}")
            print(f"  [{mname}] en: {en[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
