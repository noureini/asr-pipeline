"""
Apples-to-apples comparison: LoRA Aya vs your previous lattice/Gemma pipeline.

Uses the SAME FLEURS test samples and SAME ZIPA+FST output (`continuous` field
in results/llm_rescored.json) so the only difference is the post-processing
step. Compares against:

  - Raw FST  (continuous, no post-processing)         baseline floor
  - Lattice top-1   (mathematical lattice decoder)    your old best
  - Gemma rescored  (LLM rescoring on N-best)         your previous winner
  - Oracle          (ceiling — best of N-best)        upper bound
  - LoRA Aya        (new pipeline)                    what we're testing

Usage:
  uv run python scripts/compare_lora_vs_baseline.py \
    --gguf /path/to/aya-bengali.Q4_K_M.gguf \
    --baseline-json results/llm_rescored.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
    pred_words = pred.split()
    ref_words = ref.split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return edit_distance(pred_words, ref_words) / len(ref_words)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", type=Path, required=True,
                   help="Path to LoRA-trained Aya GGUF")
    p.add_argument("--baseline-json", type=Path,
                   default=Path("results/llm_rescored.json"),
                   help="Path to old pipeline results (must have 'continuous',"
                        " 'top1', 'gemma', 'ref' per sample)")
    p.add_argument("--n-gpu-layers", type=int, default=15)
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output", type=Path,
                   default=Path("results/comparison_lora_vs_baseline.json"))
    args = p.parse_args()

    # ─── Load baseline JSON ─────────────────────────────────────
    if not args.baseline_json.exists():
        raise FileNotFoundError(f"Baseline JSON not found: {args.baseline_json}")
    print(f"Loading baseline results from {args.baseline_json}...")
    with open(args.baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)
    samples = baseline["samples"]
    print(f"  {len(samples)} samples in baseline\n")

    # Field name detection (different baseline JSONs use different keys)
    first = samples[0]
    fst_key = "continuous" if "continuous" in first else "fst"
    top1_key = "top1" if "top1" in first else "lat_top1"
    gemma_key = "gemma" if "gemma" in first else "rescored"
    print(f"  Detected fields: noisy='{fst_key}', top1='{top1_key}', "
          f"gemma='{gemma_key}'")

    # ─── Load LoRA GGUF ─────────────────────────────────────────
    print(f"\nLoading LoRA model from {args.gguf}...")
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(
        model_path=str(args.gguf),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        verbose=False,
    )
    print(f"  ✓ loaded in {time.time()-t0:.1f}s\n")

    # ─── Process each sample ────────────────────────────────────
    print("=" * 110)
    print(f"{'#':>2}  {'FST':>7}  {'lattice':>8}  {'Gemma':>7}  "
          f"{'LoRA':>7}  {'Δ vs Gemma':>11}  {'time':>6}")
    print("=" * 110)

    sums = {"fst": 0.0, "top1": 0.0, "gemma": 0.0, "lora": 0.0,
            "fst_w": 0.0, "top1_w": 0.0, "gemma_w": 0.0, "lora_w": 0.0,
            "lora_seconds": 0.0}
    rows = []

    for i, s in enumerate(samples, 1):
        ref = s["ref"]
        fst = s.get(fst_key, "")
        top1 = s.get(top1_key, "")
        gemma = s.get(gemma_key, "")

        if not fst:
            continue

        # ─── Run LoRA on the FST noisy input ────────────────────
        t_lora = time.time()
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": fst},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        lora_raw = resp["choices"][0]["message"]["content"].strip()
        # Take first non-empty Bengali line
        lora_pred = lora_raw
        for line in lora_raw.splitlines():
            line = line.strip()
            if line and any(0x0980 <= ord(c) <= 0x09FF for c in line):
                lora_pred = line
                break
        t_lora = time.time() - t_lora

        # ─── Score everything ───────────────────────────────────
        fst_cer = cer(fst, ref);     fst_wer = wer(fst, ref)
        top1_cer = cer(top1, ref);   top1_wer = wer(top1, ref)
        gemma_cer = cer(gemma, ref); gemma_wer = wer(gemma, ref)
        lora_cer = cer(lora_pred, ref); lora_wer = wer(lora_pred, ref)

        sums["fst"] += fst_cer;     sums["fst_w"] += fst_wer
        sums["top1"] += top1_cer;   sums["top1_w"] += top1_wer
        sums["gemma"] += gemma_cer; sums["gemma_w"] += gemma_wer
        sums["lora"] += lora_cer;   sums["lora_w"] += lora_wer
        sums["lora_seconds"] += t_lora

        delta_gemma = (lora_cer - gemma_cer) * 100
        marker = "↓" if delta_gemma < 0 else ("↑" if delta_gemma > 0 else "=")
        print(f"{i:>2}  {fst_cer:>6.1%}  {top1_cer:>7.1%}  "
              f"{gemma_cer:>6.1%}  {lora_cer:>6.1%}  "
              f"{delta_gemma:>+10.1f} {marker}  {t_lora:>5.1f}s")

        rows.append({
            "ref": ref, "fst": fst, "top1": top1,
            "gemma": gemma, "lora": lora_pred, "lora_raw": lora_raw,
            "fst_cer": fst_cer, "top1_cer": top1_cer,
            "gemma_cer": gemma_cer, "lora_cer": lora_cer,
            "fst_wer": fst_wer, "top1_wer": top1_wer,
            "gemma_wer": gemma_wer, "lora_wer": lora_wer,
            "delta_vs_gemma": delta_gemma,
            "lora_seconds": t_lora,
        })

    n = len(rows)
    print("─" * 110)
    print(f"AVG {sums['fst']/n:>6.1%}  {sums['top1']/n:>7.1%}  "
          f"{sums['gemma']/n:>6.1%}  {sums['lora']/n:>6.1%}  "
          f"{(sums['lora']-sums['gemma'])/n*100:>+10.1f}  "
          f"{sums['lora_seconds']/n:>5.1f}s")

    # ─── Comparison summary ─────────────────────────────────────
    print()
    print("=" * 80)
    print(f"COMPARISON ON {n} FLEURS TEST SAMPLES (apples-to-apples)")
    print("=" * 80)
    print(f"\n  {'Pipeline':<40}  {'CER':>8}  {'WER':>8}")
    print(f"  {'-' * 40}  {'-' * 8}  {'-' * 8}")
    print(f"  {'Raw FST (no post-processing)':<40}  "
          f"{sums['fst']/n:>7.1%}  {sums['fst_w']/n:>7.1%}")
    print(f"  {'Lattice decoder (top-1)':<40}  "
          f"{sums['top1']/n:>7.1%}  {sums['top1_w']/n:>7.1%}")
    print(f"  {'Lattice + Gemma rescore (your best)':<40}  "
          f"{sums['gemma']/n:>7.1%}  {sums['gemma_w']/n:>7.1%}")
    print(f"  {'NEW: Trained LoRA Aya 8B':<40}  "
          f"{sums['lora']/n:>7.1%}  {sums['lora_w']/n:>7.1%}")

    # Improvements
    print(f"\n  Improvements (LoRA vs your previous):")
    print(f"    vs raw FST:        "
          f"CER {(sums['fst']-sums['lora'])/n*100:+.1f}pp  "
          f"WER {(sums['fst_w']-sums['lora_w'])/n*100:+.1f}pp")
    print(f"    vs lattice top-1:  "
          f"CER {(sums['top1']-sums['lora'])/n*100:+.1f}pp  "
          f"WER {(sums['top1_w']-sums['lora_w'])/n*100:+.1f}pp")
    print(f"    vs Gemma rescore:  "
          f"CER {(sums['gemma']-sums['lora'])/n*100:+.1f}pp  "
          f"WER {(sums['gemma_w']-sums['lora_w'])/n*100:+.1f}pp")

    # Per-sample win/lose vs Gemma
    helped = sum(1 for r in rows if r["delta_vs_gemma"] < -1.0)
    neutral = sum(1 for r in rows if abs(r["delta_vs_gemma"]) <= 1.0)
    hurt = sum(1 for r in rows if r["delta_vs_gemma"] > 1.0)
    print(f"\n  Per-sample (vs Gemma rescore):")
    print(f"    LoRA better:    {helped:>3}/{n}")
    print(f"    LoRA same:      {neutral:>3}/{n}")
    print(f"    LoRA worse:     {hurt:>3}/{n}")
    print(f"\n  Inference: {sums['lora_seconds']/n:.1f}s per sample (avg)")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "n": n,
        "model": str(args.gguf),
        "totals": {
            "fst_cer": sums["fst"]/n, "fst_wer": sums["fst_w"]/n,
            "top1_cer": sums["top1"]/n, "top1_wer": sums["top1_w"]/n,
            "gemma_cer": sums["gemma"]/n, "gemma_wer": sums["gemma_w"]/n,
            "lora_cer": sums["lora"]/n, "lora_wer": sums["lora_w"]/n,
        },
        "n_lora_better_than_gemma": helped,
        "n_lora_neutral": neutral,
        "n_lora_worse_than_gemma": hurt,
        "samples": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved → {args.output}")


if __name__ == "__main__":
    main()
