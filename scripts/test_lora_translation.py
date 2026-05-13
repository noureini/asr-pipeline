"""
Test if the LoRA-trained model can also translate to English.

Two strategies:
  A. Two-pass: LoRA corrects FST → then ask base model capability to translate
  B. Single prompt: ask LoRA to correct AND translate in one call

The LoRA was trained ONLY on (noisy → clean Bengali). Translation wasn't
in the training objective, so this tests whether:
  1. Aya's base multilingual translation survived LoRA fine-tuning
  2. The model can be prompted to translate post-correction
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

CORRECTION_PROMPT = (
    "You are a Bengali transcription corrector. The input is a noisy Bengali "
    "transcription where word boundaries are missing and some characters are "
    "phonetically confused (দ↔ধ, ব↔ভ, ত↔থ, ক↔খ, প↔ফ, স↔শ↔ষ). Output ONLY "
    "the corrected, properly-segmented Bengali sentence."
)

TRANSLATE_PROMPT = (
    "You are a Bengali-to-English translator. Translate the Bengali sentence "
    "to natural English. Output ONLY the English translation, no explanation."
)

COMBINED_PROMPT = (
    "You are a Bengali transcription corrector and translator. The input is "
    "a noisy Bengali transcription. First correct it, then translate. Output "
    "in this exact format on two lines:\n"
    "BN: <corrected Bengali>\n"
    "EN: <English translation>"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--n-gpu-layers", type=int, default=15)
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    from llama_cpp import Llama
    print(f"Loading {args.gguf}...")
    m = Llama(
        model_path=str(args.gguf),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        verbose=False,
    )
    print("  ✓ loaded\n")

    # Test samples (with English meaning for ground truth comparison)
    tests = [
        {
            "noisy": "তবেতারসমুদ্শিজবাবদেবেআওএআমেলি",
            "truth_bn": "তবে তার সমুচিত জবাব দেবে আওয়ামী লীগ",
            "truth_en": "But the Awami League will give a fitting response.",
        },
        {
            "noisy": "আরিট্টুকুলেআলুনা",
            "truth_bn": "আর একটু খুলে বলো না",
            "truth_en": "Tell me a bit more openly.",
        },
        {
            "noisy": "সম্পুর্নপাল্কেদাকাএইউনরক্তবিশিতপাখিতিবেলোসিরাপতেরোরমতোনোগদিএদুইপাএসুজাপাহেএহাতেবলেবিশ্বাসঅকরাহেএছিল",
            "truth_bn": "সম্পূর্ণ পালকে ঢাকা এই উষ্ণ রক্তবিশিষ্ট পাখিটি ভেলোসিরাপটরের মতো নখ দিয়ে দুই পায়ে সোজা হয়ে হাঁটে বলে বিশ্বাস করা হয়েছিল",
            "truth_en": "It was believed that this fully-feathered warm-blooded bird walked upright on two legs with claws like a Velociraptor.",
        },
    ]

    for i, t in enumerate(tests, 1):
        print("=" * 100)
        print(f"TEST {i}")
        print("=" * 100)
        print(f"NOISY (FST):       {t['noisy']}")
        print(f"TRUTH (Bengali):   {t['truth_bn']}")
        print(f"TRUTH (English):   {t['truth_en']}")

        # ─── Approach A: two-pass ─────────────────────────────────
        print("\n--- APPROACH A: Two-pass (correct → then translate) ---")
        # Pass 1: correction
        r1 = m.create_chat_completion(
            messages=[
                {"role": "system", "content": CORRECTION_PROMPT},
                {"role": "user", "content": t["noisy"]},
            ],
            temperature=0.0, max_tokens=args.max_tokens,
        )
        corrected = r1["choices"][0]["message"]["content"].strip()
        print(f"  Step 1 (LoRA correct):  {corrected}")

        # Pass 2: translation of corrected text
        r2 = m.create_chat_completion(
            messages=[
                {"role": "system", "content": TRANSLATE_PROMPT},
                {"role": "user", "content": corrected},
            ],
            temperature=0.0, max_tokens=args.max_tokens,
        )
        translated = r2["choices"][0]["message"]["content"].strip()
        print(f"  Step 2 (translate):     {translated}")

        # ─── Approach B: combined prompt ─────────────────────────
        print("\n--- APPROACH B: Combined (correct + translate in one call) ---")
        r3 = m.create_chat_completion(
            messages=[
                {"role": "system", "content": COMBINED_PROMPT},
                {"role": "user", "content": t["noisy"]},
            ],
            temperature=0.0, max_tokens=args.max_tokens,
        )
        combined = r3["choices"][0]["message"]["content"].strip()
        print(f"  Output:")
        for line in combined.splitlines():
            print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
