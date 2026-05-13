"""
Test 3 different LoRA input strategies on the same FLEURS samples:

  Strategy A: ZIPA → FST → LoRA              (current production — feeds noisy FST)
  Strategy B: ZIPA → FST → LATTICE → LoRA    (NEW — feeds lattice-decoded output)
  Strategy C: ZIPA → FST → GEMMA → LoRA      (NEW — feeds prior best, double-pass)

Hypothesis: cleaner input → less hallucination risk, better final quality.

Uses results/baseline_merged.json which has:
  ref          ground truth
  continuous   FST output (~35% CER) — input for Strategy A
  top1         lattice output (~20% CER) — input for Strategy B
  gemma        Gemma rescored (~18% CER) — input for Strategy C

Usage:
  uv run python scripts/compare_lora_chain_strategies.py \
    --gguf models/aya-expanse-8b.Q4_K_M.gguf
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

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


def wer(pred, ref):
    pred_w = pred.split(); ref_w = ref.split()
    if not ref_w: return 0.0 if not pred_w else 1.0
    return edit_distance(pred_w, ref_w) / len(ref_w)


def lora_correct(llm, noisy_input, max_tokens=512, temperature=0.0):
    """Run LoRA correction on a noisy input, return clean Bengali."""
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": noisy_input},
        ],
        temperature=temperature, max_tokens=max_tokens,
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    # Take first non-empty Bengali line
    for line in raw.splitlines():
        line = line.strip()
        if line and any(0x0980 <= ord(c) <= 0x09FF for c in line):
            return line
    return raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--baseline-json", type=Path,
                   default=Path("results/baseline_merged.json"))
    p.add_argument("--n-gpu-layers", type=int, default=15)
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--output", type=Path,
                   default=Path("results/comparison_lora_chains.json"))
    args = p.parse_args()

    # Load baseline
    print(f"Loading {args.baseline_json}...")
    with open(args.baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)
    samples = baseline["samples"]
    print(f"  {len(samples)} samples\n")

    # Load LoRA
    from llama_cpp import Llama
    print(f"Loading LoRA model from {args.gguf}...")
    t0 = time.time()
    llm = Llama(
        model_path=str(args.gguf),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        verbose=False,
    )
    print(f"  ✓ loaded in {time.time()-t0:.1f}s\n")

    # Run all 3 strategies
    print("=" * 130)
    print(f"{'#':>2}  "
          f"{'A: FST→LoRA':>12}  "
          f"{'B: Lattice→LoRA':>16}  "
          f"{'C: Gemma→LoRA':>14}  "
          f"{'baseline FST':>12}  "
          f"{'baseline Latt':>13}  "
          f"{'baseline Gemma':>14}")
    print("=" * 130)

    sums = {
        "fst_input_cer": 0.0,    "fst_input_wer": 0.0,
        "latt_input_cer": 0.0,   "latt_input_wer": 0.0,
        "gemma_input_cer": 0.0,  "gemma_input_wer": 0.0,
        "lora_from_fst_cer": 0.0,    "lora_from_fst_wer": 0.0,
        "lora_from_latt_cer": 0.0,   "lora_from_latt_wer": 0.0,
        "lora_from_gemma_cer": 0.0,  "lora_from_gemma_wer": 0.0,
        "total_seconds": 0.0,
    }
    rows = []

    for i, s in enumerate(samples, 1):
        ref = s["ref"]
        fst = s.get("continuous", "")
        latt = s.get("top1", "")
        gemma = s.get("gemma", "")

        if not fst or not latt or not gemma:
            print(f"  [{i}] missing baseline field, skipping")
            continue

        # Strategy A: LoRA on raw FST
        t = time.time()
        lora_a = lora_correct(llm, fst, args.max_tokens)
        ta = time.time() - t

        # Strategy B: LoRA on lattice top-1
        t = time.time()
        lora_b = lora_correct(llm, latt, args.max_tokens)
        tb = time.time() - t

        # Strategy C: LoRA on Gemma rescored
        t = time.time()
        lora_c = lora_correct(llm, gemma, args.max_tokens)
        tc = time.time() - t

        # Score everything
        fst_cer = cer(fst, ref);     fst_wer = wer(fst, ref)
        latt_cer = cer(latt, ref);   latt_wer = wer(latt, ref)
        gemma_cer = cer(gemma, ref); gemma_wer = wer(gemma, ref)
        lora_a_cer = cer(lora_a, ref);   lora_a_wer = wer(lora_a, ref)
        lora_b_cer = cer(lora_b, ref);   lora_b_wer = wer(lora_b, ref)
        lora_c_cer = cer(lora_c, ref);   lora_c_wer = wer(lora_c, ref)

        sums["fst_input_cer"] += fst_cer; sums["fst_input_wer"] += fst_wer
        sums["latt_input_cer"] += latt_cer; sums["latt_input_wer"] += latt_wer
        sums["gemma_input_cer"] += gemma_cer; sums["gemma_input_wer"] += gemma_wer
        sums["lora_from_fst_cer"] += lora_a_cer; sums["lora_from_fst_wer"] += lora_a_wer
        sums["lora_from_latt_cer"] += lora_b_cer; sums["lora_from_latt_wer"] += lora_b_wer
        sums["lora_from_gemma_cer"] += lora_c_cer; sums["lora_from_gemma_wer"] += lora_c_wer
        sums["total_seconds"] += (ta + tb + tc)

        print(f"{i:>2}  "
              f"{lora_a_cer:>11.1%}  "
              f"{lora_b_cer:>15.1%}  "
              f"{lora_c_cer:>13.1%}  "
              f"{fst_cer:>11.1%}  "
              f"{latt_cer:>12.1%}  "
              f"{gemma_cer:>13.1%}")

        rows.append({
            "ref": ref, "fst": fst, "latt": latt, "gemma": gemma,
            "lora_from_fst": lora_a, "lora_from_latt": lora_b, "lora_from_gemma": lora_c,
            "fst_cer": fst_cer, "latt_cer": latt_cer, "gemma_cer": gemma_cer,
            "lora_a_cer": lora_a_cer, "lora_b_cer": lora_b_cer, "lora_c_cer": lora_c_cer,
            "fst_wer": fst_wer, "latt_wer": latt_wer, "gemma_wer": gemma_wer,
            "lora_a_wer": lora_a_wer, "lora_b_wer": lora_b_wer, "lora_c_wer": lora_c_wer,
        })

    n = len(rows)
    print("─" * 130)
    print(f"AVG {sums['lora_from_fst_cer']/n:>11.1%}  "
          f"{sums['lora_from_latt_cer']/n:>15.1%}  "
          f"{sums['lora_from_gemma_cer']/n:>13.1%}  "
          f"{sums['fst_input_cer']/n:>11.1%}  "
          f"{sums['latt_input_cer']/n:>12.1%}  "
          f"{sums['gemma_input_cer']/n:>13.1%}")

    # Comprehensive summary
    print()
    print("=" * 90)
    print(f"COMPARISON OF LoRA INPUT STRATEGIES ({n} FLEURS samples)")
    print("=" * 90)
    print(f"\n  {'Pipeline':<55}  {'CER':>8}  {'WER':>8}")
    print(f"  {'-' * 55}  {'-' * 8}  {'-' * 8}")
    print(f"  {'(input only) Raw FST':<55}  "
          f"{sums['fst_input_cer']/n:>7.1%}  {sums['fst_input_wer']/n:>7.1%}")
    print(f"  {'(input only) Lattice top-1':<55}  "
          f"{sums['latt_input_cer']/n:>7.1%}  {sums['latt_input_wer']/n:>7.1%}")
    print(f"  {'(input only) Lattice + Gemma rescore':<55}  "
          f"{sums['gemma_input_cer']/n:>7.1%}  {sums['gemma_input_wer']/n:>7.1%}")
    print(f"  {'-' * 55}  {'-' * 8}  {'-' * 8}")
    print(f"  {'Strategy A: ZIPA→FST→LoRA':<55}  "
          f"{sums['lora_from_fst_cer']/n:>7.1%}  {sums['lora_from_fst_wer']/n:>7.1%}")
    print(f"  {'Strategy B: ZIPA→FST→LATTICE→LoRA':<55}  "
          f"{sums['lora_from_latt_cer']/n:>7.1%}  {sums['lora_from_latt_wer']/n:>7.1%}")
    print(f"  {'Strategy C: ZIPA→FST→LATTICE→GEMMA→LoRA':<55}  "
          f"{sums['lora_from_gemma_cer']/n:>7.1%}  {sums['lora_from_gemma_wer']/n:>7.1%}")

    # Pick winner
    cers = {
        "Strategy A (FST→LoRA)": sums["lora_from_fst_cer"]/n,
        "Strategy B (LATTICE→LoRA)": sums["lora_from_latt_cer"]/n,
        "Strategy C (GEMMA→LoRA)": sums["lora_from_gemma_cer"]/n,
    }
    winner = min(cers, key=cers.get)
    print(f"\n  ★ Winner: {winner}  (CER {cers[winner]:.1%})")

    # Per-sample which strategy won
    a_wins = sum(1 for r in rows
                 if r["lora_a_cer"] <= r["lora_b_cer"] and r["lora_a_cer"] <= r["lora_c_cer"])
    b_wins = sum(1 for r in rows
                 if r["lora_b_cer"] < r["lora_a_cer"] and r["lora_b_cer"] <= r["lora_c_cer"])
    c_wins = sum(1 for r in rows
                 if r["lora_c_cer"] < r["lora_a_cer"] and r["lora_c_cer"] < r["lora_b_cer"])
    print(f"\n  Per-sample winner counts:")
    print(f"    Strategy A best:   {a_wins}/{n}")
    print(f"    Strategy B best:   {b_wins}/{n}")
    print(f"    Strategy C best:   {c_wins}/{n}")

    print(f"\n  Inference: {sums['total_seconds']/n/3:.1f}s per sample per strategy")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "n": n,
        "totals": {k: v/n for k, v in sums.items() if k != "total_seconds"},
        "winner": winner,
        "winner_cer": cers[winner],
        "samples": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved → {args.output}")


if __name__ == "__main__":
    main()
