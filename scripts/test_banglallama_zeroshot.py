"""
Zero-shot test of BanglaLlama 3.1 8B (or any candidate base model)
on FST → clean Bengali correction. NO fine-tuning, just the base model
+ system prompt.

This reveals how much Bengali knowledge the base already has for this task.
If zero-shot CER << raw FST CER, the base model is genuinely useful and
fine-tuning will amplify it. If zero-shot CER ~ raw FST CER, the model
doesn't know Bengali well enough for this task.

Usage (run in Colab with GPU):
  python scripts/test_banglallama_zeroshot.py \
    --baseline-json results/baseline_merged.json
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

CANDIDATE_MODELS = [
    "BanglaLLM/BanglaLLama-3.1-8b-bangla-alpaca-orca-instruct-v0.0.1",
    # Optional: add more for comparison
    # "BanglaLLM/BanglaLLama-3-8b-bangla-alpaca-orca-instruct-v0.0.1",
    # "CohereLabs/aya-expanse-8b",  # for sanity
]

SYSTEM_PROMPT = (
    "You are a Bengali transcription corrector. The input is a noisy Bengali "
    "transcription where word boundaries are missing and some characters are "
    "phonetically confused (দ↔ধ, ব↔ভ, ত↔থ, ক↔খ, প↔ফ, স↔শ↔ষ). Output ONLY "
    "the corrected, properly-segmented Bengali sentence."
)


def edit_distance(a, b):
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1,
                           prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, ref):
    if not ref: return 0.0 if not pred else 1.0
    return edit_distance(pred, ref) / len(ref)


def run_zero_shot(model_id, samples, max_tokens=512):
    """Load model with Unsloth, run zero-shot on samples."""
    print(f"\n{'='*100}")
    print(f"MODEL: {model_id}")
    print(f"{'='*100}\n")

    from unsloth import FastLanguageModel
    print("Loading model (4-bit quantized)...")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"  ✓ loaded in {time.time()-t0:.0f}s")

    rows = []
    sum_cer = 0.0
    sum_input_cer = 0.0

    print(f"\n{'#':>3}  {'input CER':>9}  {'output CER':>10}  {'Δ':>7}")
    print("─" * 60)
    for i, s in enumerate(samples, 1):
        ref = s["ref"]
        noisy = s.get("continuous", "") or s.get("fst", "")
        if not noisy:
            continue

        # Build prompt and run
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": noisy},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Take first non-empty Bengali line
        output = gen
        for line in gen.splitlines():
            line = line.strip()
            if line and any(0x0980 <= ord(c) <= 0x09FF for c in line):
                output = line
                break

        c_in = cer(noisy, ref)
        c_out = cer(output, ref)
        sum_cer += c_out
        sum_input_cer += c_in
        delta = (c_out - c_in) * 100
        marker = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        print(f"{i:>3}  {c_in:>8.1%}  {c_out:>9.1%}  {delta:>+6.1f} {marker}")

        rows.append({
            "ref": ref, "noisy": noisy, "output": output, "raw": gen,
            "input_cer": c_in, "output_cer": c_out,
        })

    n = len(rows)
    avg_in = sum_input_cer / n
    avg_out = sum_cer / n
    print("─" * 60)
    print(f"AVG  {avg_in:>8.1%}  {avg_out:>9.1%}  {(avg_out-avg_in)*100:>+6.1f}")

    return {
        "model": model_id,
        "input_cer_avg": avg_in,
        "output_cer_avg": avg_out,
        "improvement_pct": (avg_in - avg_out) * 100,
        "samples": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-json", type=Path,
                   default=Path("results/baseline_merged.json"))
    p.add_argument("--output", type=Path,
                   default=Path("results/zero_shot_comparison.json"))
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    # Load samples
    with open(args.baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)
    samples = baseline["samples"]
    print(f"Loaded {len(samples)} test samples\n")

    # Test each candidate model
    results = {}
    for model_id in CANDIDATE_MODELS:
        try:
            results[model_id] = run_zero_shot(model_id, samples, args.max_tokens)
        except Exception as e:
            print(f"\n⚠ FAILED {model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final comparison
    print(f"\n{'='*100}")
    print("FINAL ZERO-SHOT COMPARISON")
    print("="*100)
    print(f"\n  {'Model':<55}  {'CER':>8}  {'Improvement vs raw FST':>24}")
    print(f"  {'-'*55}  {'-'*8}  {'-'*24}")
    for model_id, r in results.items():
        short = model_id.split('/')[-1][:55]
        print(f"  {short:<55}  {r['output_cer_avg']:>7.1%}  "
              f"{r['improvement_pct']:>+22.1f}pp")

    # For reference: known baselines
    print(f"\n  Known baselines for comparison:")
    print(f"    Raw FST (no correction):           ~34.5%")
    print(f"    Lattice top-1:                     ~19.9%")
    print(f"    Lattice + Gemma rescore:           ~18.3%")
    print(f"    Aya 8B + LoRA fine-tuned:          ~13.8%")
    print(f"    ChatGPT zero-shot:                 ~3-5% (estimated)")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "n": len(samples),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved → {args.output}")


if __name__ == "__main__":
    main()
