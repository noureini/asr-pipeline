"""Quick test: can Gemma-3 translate the fused Bengali faithfully —
keeping numbers IndicTrans2 botched (6→696, 30.5→35)?

Translates the exact [bn] lines from the production run that IndicTrans2
mistranslated. If Gemma keeps "6 people" and "thirty and a half", we
drop IndicTrans2 and let Gemma do BOTH fusion and translation (one
model, fully local).

Usage:
    uv run python scripts/test_gemma_translate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Exact fused [bn] from the production run (the ones IndicTrans2 failed).
SEGMENTS = [
    # IndicTrans2 said "696 / 699 people" — should be 6
    "আমার ফ্যামিলিতে আমার ছয় জন মানুষ। এই ছয় জন মানুষে আমার এক কৃষি কি "
    "কাজের উপরে নির্ভর করে কুনু বছর যদি মাইর যায়, তখনি কষ্ট বেড়ে যায়।",
    # IndicTrans2 said "thirty-five" + "became a Muslim" — should be 30.5
    "বয়স কত মনে আছে কি? আমার বয়স সাড়ে ত্রিশ বছর। পরে আমি মুসলিম ধর্মের "
    "অনুসারী। আপনি বিবাহিত তো?",
    # IndicTrans2 mangled পড়া → "Pod"
    "স্ত্রী বেঁচে আছেন? আছেন। আচ্ছা। আপনি কি বাংলা পড়তে পারেন? না, পড়তে "
    "লিখতে পারেন না।",
]

_PROMPT = """\
Translate this Bengali survey utterance into natural, fluent English. The \
speaker is a rural respondent in coastal Bangladesh using COLLOQUIAL / \
regional Bengali — translate the MEANING (not word-by-word), interpreting \
dialectal forms by sense (e.g. মাইর যায় = the crop is destroyed; কুনু = \
some/any; খাজ = work). Render every number EXACTLY as written, and read \
Bengali half-numbers correctly: সাড়ে X = X-and-a-half, দেড় = 1.5, আড়াই = \
2.5. Keep proper nouns / place / bank names in standard romanization and \
code-switched English as-is. If a word's meaning is genuinely \
unrecoverable, write [unclear] — never transliterate a Bengali word into \
Latin letters and never guess a number. Output only the English.

Bengali: {bn}
English:"""


def main() -> int:
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-repo", default="unsloth/gemma-3-12b-it-GGUF")
    ap.add_argument("--gguf-file",
                    default="gemma-3-12b-it-UD-Q4_K_XL.gguf")
    ap.add_argument("--prompt-lookup", type=int, default=0,
                    help="prompt-lookup speculative decoding tokens "
                         "(0=off, 10=on). Compare timing on vs off.")
    args = ap.parse_args()
    from asr_pipeline.postprocessor import GGUFProcessor
    proc = GGUFProcessor(
        gguf_repo=args.gguf_repo, gguf_file=args.gguf_file,
        n_ctx=4096, n_gpu_layers=-1,
        prompt_lookup_tokens=args.prompt_lookup)
    if not proc.load():
        print("Gemma GGUF failed to load", file=sys.stderr)
        return 1
    total = 0.0
    for bn in SEGMENTS:
        t0 = time.perf_counter()
        en = proc.generate(_PROMPT.format(bn=bn), temperature=0.0,
                           max_tokens=256).strip()
        dt = time.perf_counter() - t0
        total += dt
        print("BN:", bn)
        print(f"EN ({dt:.1f}s):", en)
        print()
    print(f"=== total generation: {total:.1f}s "
          f"(prompt_lookup={args.prompt_lookup}) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
