"""Minimal IndicTrans2 Bengali→English smoke test — isolates the
translation step so we can see the FULL traceback and fix it precisely
(instead of re-running the 3-min fusion). Uses the canonical
IndicTransToolkit inference recipe, incl. as_target_tokenizer() for
decoding (which the main script was missing).

Usage:
    uv run python scripts/test_indictrans2.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def hf_token():
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, val = line.partition("=")
                if key.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN",
                                   "HUGGING_FACE_HUB_TOKEN"):
                    return val.strip().strip('"').strip("'") or None
    return None


TESTS = [
    "বন্যা কেমন হতে পারে এ বিষয়ে আন্দাজ করেন সেটা বলছিলেন",      # flood
    "পূবালী ব্যাংক আপনাকে দেয় না, এই কৃষি ব্যাংকে আমি পঁচিশ",    # bank + 25
    "আমার বয়স সাড়ে ত্রিশ বছর",                                  # 30.5 number
]


def main() -> int:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor

    repo = "ai4bharat/indictrans2-indic-en-1B"
    tok = hf_token()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {repo} on {dev} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        repo, trust_remote_code=True, token=tok)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repo, trust_remote_code=True, token=tok).to(dev).eval()
    ip = IndicProcessor(inference=True)

    print("Preprocessing ...")
    batch = ip.preprocess_batch(
        TESTS, src_lang="ben_Beng", tgt_lang="eng_Latn")
    print("  preprocessed:", batch)

    print("Tokenizing ...")
    inputs = tokenizer(batch, truncation=True, padding="longest",
                       return_tensors="pt", return_attention_mask=True)
    print("  keys:", list(inputs.keys()),
          "| input_ids shape:", getattr(inputs.get("input_ids"),
                                        "shape", None),
          "| attn shape:", getattr(inputs.get("attention_mask"),
                                   "shape", None))
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    import transformers
    print("transformers version:", transformers.__version__)
    print("Generating (use_cache=False — avoids the old past_key_values "
          "format the model's custom code expects) ...")
    with torch.no_grad():
        gen = model.generate(**inputs, use_cache=False, min_length=0,
                             max_length=256, num_beams=5,
                             num_return_sequences=1)
    print("  generated shape:", gen.shape)

    print("Decoding (as_target_tokenizer) ...")
    with tokenizer.as_target_tokenizer():
        dec = tokenizer.batch_decode(
            gen.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)
    trans = ip.postprocess_batch(dec, lang="eng_Latn")

    print("\n=== RESULTS ===")
    for bn, en in zip(TESTS, trans):
        print(f"BN: {bn}")
        print(f"EN: {en}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n=== FULL TRACEBACK ===", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
