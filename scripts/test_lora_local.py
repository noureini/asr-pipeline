"""
Test the trained Bengali LoRA corrector on local hardware.

Loads a GGUF model (downloaded from Colab Studio export) and runs it
on a set of held-out FLEURS/banspeech/SKNahin samples to validate
the model is working correctly.

Usage:
  uv pip install llama-cpp-python
  uv run python scripts/test_lora_local.py \
    --model /path/to/aya-bengali.Q4_K_M.gguf \
    --eval-jsonl /path/to/lora_dataset_eval.jsonl \
    --n-samples 10
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a Bengali transcription corrector. The input is a noisy Bengali "
    "transcription where word boundaries are missing and some characters are "
    "phonetically confused (দ↔ধ, ব↔ভ, ত↔থ, ক↔খ, প↔ফ, স↔শ↔ষ). Output ONLY "
    "the corrected, properly-segmented Bengali sentence."
)


def edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein for char-level CER."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,        # deletion
                cur[j-1] + 1,       # insertion
                prev[j-1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    """Character Error Rate, normalized by reference length."""
    if not ref:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, ref) / len(ref)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True,
                   help="Path to the GGUF file downloaded from Studio export")
    p.add_argument("--eval-jsonl", type=Path,
                   default=Path("results/lora_dataset_eval.jsonl"),
                   help="Path to held-out eval set JSONL")
    p.add_argument("--n-samples", type=int, default=10,
                   help="Number of random eval samples to test")
    p.add_argument("--n-gpu-layers", type=int, default=15,
                   help="GPU layers to offload (0=CPU only, -1=all)")
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--by-source", action="store_true",
                   help="Sample evenly across (src, subsource) groups")
    args = p.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.eval_jsonl.exists():
        raise FileNotFoundError(f"Eval file not found: {args.eval_jsonl}")

    # Load eval samples
    print(f"Loading eval set from {args.eval_jsonl}...")
    samples = []
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  {len(samples)} samples available")

    # Pick samples
    rng = random.Random(args.seed)
    if args.by_source:
        from collections import defaultdict
        groups = defaultdict(list)
        for s in samples:
            key = (s.get("src", "?"), s.get("subsource", ""))
            groups[key].append(s)
        per_group = max(1, args.n_samples // len(groups))
        test_samples = []
        for key, lst in groups.items():
            test_samples.extend(rng.sample(lst, min(per_group, len(lst))))
    else:
        test_samples = rng.sample(samples, min(args.n_samples, len(samples)))

    print(f"  Testing on {len(test_samples)} samples\n")

    # Load model
    print(f"Loading model from {args.model}...")
    print(f"  n_gpu_layers={args.n_gpu_layers}, n_threads={args.n_threads}")
    print("  (~30-60s on first load)\n")

    from llama_cpp import Llama
    t0 = time.time()
    m = Llama(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        verbose=False,
    )
    print(f"  ✓ loaded in {time.time()-t0:.1f}s\n")

    # Run inference + score
    print("=" * 80)
    cers = []
    rows = []
    for i, s in enumerate(test_samples, 1):
        noisy = s["noisy"]
        truth = s["bengali"]
        src = s.get("src", "?")
        sub = s.get("subsource", "")
        label = f"{src}/{sub}" if sub else src

        t_start = time.time()
        resp = m.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": noisy},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        output = resp["choices"][0]["message"]["content"].strip()
        elapsed = time.time() - t_start

        # Take first non-empty Bengali line if multi-line
        for line in output.splitlines():
            line = line.strip()
            if line and any(0x0980 <= ord(c) <= 0x09FF for c in line):
                output = line
                break

        sample_cer = cer(output, truth)
        cers.append(sample_cer)
        rows.append({
            "src": label, "noisy": noisy, "truth": truth,
            "output": output, "cer": sample_cer, "seconds": elapsed,
        })

        # Per-sample report
        print(f"\n[{i}/{len(test_samples)}] [{label}]  CER={sample_cer:.1%}  "
              f"({elapsed:.1f}s)")
        print(f"  NOISY:  {noisy[:120]}")
        print(f"  TRUTH:  {truth[:120]}")
        print(f"  OUTPUT: {output[:120]}")

    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY ({len(cers)} samples)")
    print("=" * 80)
    avg_cer = sum(cers) / len(cers)
    print(f"  Average CER:   {avg_cer:.1%}")
    print(f"  Median CER:    {sorted(cers)[len(cers)//2]:.1%}")
    print(f"  Best CER:      {min(cers):.1%}")
    print(f"  Worst CER:     {max(cers):.1%}")
    print(f"  Avg time/sample: {sum(r['seconds'] for r in rows)/len(rows):.1f}s")

    # Per-source breakdown
    from collections import defaultdict
    per_src = defaultdict(list)
    for r in rows:
        per_src[r["src"]].append(r["cer"])
    print(f"\n  Per-source CER:")
    for src, cer_list in sorted(per_src.items()):
        avg = sum(cer_list) / len(cer_list)
        print(f"    {src:<30} {avg:.1%}  (n={len(cer_list)})")

    # Save full results
    out_path = Path("results") / "lora_local_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": str(args.model),
        "n_samples": len(rows),
        "avg_cer": avg_cer,
        "samples": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
