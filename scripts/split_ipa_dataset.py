"""
Split the IPA dataset into train/val with stratified sampling.

Per-source eval split ensures eval_loss reflects performance across
all distributions (not just the dominant one).

Usage:
  uv run python scripts/split_ipa_dataset.py \
    --input ./lora_data_ipa/lora_dataset_full.jsonl \
    --eval-per-source 100
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Combined JSONL with all sources")
    p.add_argument("--train-out", type=Path, default=None,
                   help="Defaults to <input>_train.jsonl")
    p.add_argument("--val-out", type=Path, default=None,
                   help="Defaults to <input>_val.jsonl")
    p.add_argument("--eval-per-source", type=int, default=100,
                   help="Max samples per (src, subsource) for eval")
    p.add_argument("--max-eval-frac", type=float, default=0.2,
                   help="Cap eval at N/group * frac (don't take more than 20% of any group)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Default output paths
    if args.train_out is None:
        args.train_out = args.input.parent / f"{args.input.stem}_train.jsonl"
    if args.val_out is None:
        args.val_out = args.input.parent / f"{args.input.stem}_val.jsonl"

    # ─── Group lines by (src, subsource) ────────────────────────
    print(f"Reading {args.input}...")
    groups = defaultdict(list)
    total = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                key = (d.get("src", "?"), d.get("subsource", ""))
                groups[key].append(line)
                total += 1
            except Exception:
                continue

    print(f"\n{total} samples across {len(groups)} (src, subsource) groups:")
    for k in sorted(groups.keys(), key=lambda x: -len(groups[x])):
        print(f"  {str(k):<45} {len(groups[k]):>6}")

    # ─── Stratified split ───────────────────────────────────────
    rng = random.Random(args.seed)
    train_lines, val_lines = [], []
    for key, lines in groups.items():
        rng.shuffle(lines)
        n_eval = min(args.eval_per_source,
                     int(len(lines) * args.max_eval_frac))
        n_eval = max(1, n_eval)  # at least 1 if group exists
        val_lines.extend(lines[:n_eval])
        train_lines.extend(lines[n_eval:])

    # Shuffle final files (mix sources in training batches)
    rng.shuffle(train_lines)
    rng.shuffle(val_lines)

    # ─── Write ──────────────────────────────────────────────────
    with open(args.train_out, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(args.val_out, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print(f"\n  train: {len(train_lines):>6}  → {args.train_out}")
    print(f"  val:   {len(val_lines):>6}  → {args.val_out}")

    # ─── Verify val composition ────────────────────────────────
    print(f"\nVal set composition (per source):")
    val_groups = defaultdict(int)
    for line in val_lines:
        d = json.loads(line)
        key = (d.get("src", "?"), d.get("subsource", ""))
        val_groups[key] += 1
    for k in sorted(val_groups.keys(), key=lambda x: -val_groups[x]):
        bar = "█" * (val_groups[k] // 5)
        print(f"  {str(k):<45} {val_groups[k]:>4}  {bar}")


if __name__ == "__main__":
    main()
