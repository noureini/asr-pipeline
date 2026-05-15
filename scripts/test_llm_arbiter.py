"""
LLM-arbiter prototype on top of the v3 physics lattice.

The hypothesis: v3 bucket+DTW gives recall@100 = 91.5% (gold word IS in
top-100 nine times out of ten), but its rank-1 accuracy is only 58.5%
(it ranks the wrong word first half the time). An LLM picking from
top-K candidates — a constrained selection task, not open generation —
should boost rank-1 accuracy substantially.

This script measures the lift:
    DTW rank-1 (baseline)   →  LLM-arbiter rank-1 (the test)

If LLM rank-1 ≥ 80% on the same n=200 noisy queries v3 was tested on,
the hybrid architecture is validated and we move to real-audio testing.

Usage:
    # First make sure ollama has the model pulled
    ollama pull qwen2.5:7b

    # Run the arbiter (reuses the existing BN-only physics index)
    uv run python scripts/test_llm_arbiter.py --n 200

    # Smaller K (faster prompts, may hurt accuracy)
    uv run python scripts/test_llm_arbiter.py --n 50 --k 20

    # Different model
    uv run python scripts/test_llm_arbiter.py --n 50 --model llama3.1:8b

    # Test on your own (gold_word, zipa_ipa) pairs
    uv run python scripts/test_llm_arbiter.py --n 100 \
        --jsonl results/my_zipa_pairs.jsonl

Output:
    Console: DTW vs LLM rank-1 accuracy, sample disagreements
    results/llm_arbiter_results.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

# Reuse the existing physics-lattice infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from test_phys_lattice_recall import (  # noqa: E402
    CACHE_INDEX_BN, CACHE_INDEX_FULL,
    FeatureSpace, load_index,
    gen_noisy_test, load_jsonl_test,
)


SYSTEM_PROMPT = (
    "You are an expert Bengali linguist and IPA reader. "
    "You receive a noisy IPA phoneme transcription from an audio recognizer "
    "(phones may be substituted, deleted, or inserted) and a numbered list of "
    "candidate Bengali words ranked by phonetic distance. "
    "Pick the SINGLE most plausible Bengali word, considering both phonetic "
    "match and which word makes sense as a real Bengali word."
)


def build_prompt(noisy_ipa: str, candidates: list[tuple[float, str]]) -> str:
    """Format the user prompt for LLM selection. Returns plain-text prompt."""
    lines = [
        f"IPA query (noisy): /{noisy_ipa}/",
        "",
        "Candidate Bengali words (ranked by phonetic distance, lower = closer):",
    ]
    for i, (dist, word) in enumerate(candidates, start=1):
        lines.append(f"  {i}. {word}  (DTW {dist:.2f})")
    lines.append("")
    lines.append("Output ONLY the number of the most likely word. "
                 "No explanation, no extra text.")
    return "\n".join(lines)


def parse_llm_response(text: str, n_candidates: int) -> int | None:
    """Extract the candidate number from the LLM's reply.
    Returns 0-indexed position, or None if unparseable."""
    if not text:
        return None
    # Look for the first integer in the response
    m = re.search(r"\b(\d+)\b", text.strip())
    if not m:
        return None
    n = int(m.group(1))
    if 1 <= n <= n_candidates:
        return n - 1  # convert to 0-indexed
    return None


def call_ollama(model: str, system: str, user: str,
                temperature: float = 0.0, num_predict: int = 8) -> str:
    """Single chat call. Low temp + short num_predict for crisp picks."""
    try:
        import ollama
    except ImportError:
        print("ERROR: ollama-python not installed. Run: uv pip install ollama")
        sys.exit(1)
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": temperature, "num_predict": num_predict},
        )
        return resp["message"]["content"]
    except Exception as e:
        print(f"  ollama error: {e}")
        return ""


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n", type=int, default=200, help="Number of test items")
    p.add_argument("--k", type=int, default=30,
                   help="Number of candidates to show the LLM (default 30). "
                        "Higher = better recall ceiling but slower / more "
                        "tokens. Top-100 from v3 has ~91% recall, but the "
                        "LLM gets confused by long lists.")
    p.add_argument("--model", default="qwen2.5:7b",
                   help="Ollama model tag (default qwen2.5:7b)")
    p.add_argument("--bengali-only", action="store_true", default=True,
                   help="Use BN-only physics index (default True)")
    p.add_argument("--full-index", action="store_true",
                   help="Use the full mixed-language index instead")
    p.add_argument("--jsonl", type=Path, default=None,
                   help="Test on user-supplied (word, noisy_ipa) JSONL "
                        "instead of synthetic noise")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--out", type=Path,
                   default=Path("results/llm_arbiter_results.json"))
    args = p.parse_args()

    # ─── Load the physics index ───────────────────────────────────────
    cache_path = CACHE_INDEX_FULL if args.full_index else CACHE_INDEX_BN
    if not cache_path.exists():
        print(f"ERROR: index not built at {cache_path}")
        print("Run: just phys-build-bn")
        sys.exit(1)
    fs = FeatureSpace()
    idx = load_index(cache_path)
    if len(idx) == 0:
        print("ERROR: empty index")
        sys.exit(1)

    # ─── Build the test set ───────────────────────────────────────────
    rng = random.Random(args.seed)
    if args.jsonl:
        test_items = load_jsonl_test(args.jsonl)
        if args.n > 0:
            test_items = test_items[:args.n]
        print(f"Loaded {len(test_items)} items from {args.jsonl}")
    else:
        test_items = gen_noisy_test(idx, fs, args.n, args.min_len, rng)
        print(f"Synthetic noise on {len(test_items)} BN dict words "
              f"(seed={args.seed})")

    # ─── Run the arbiter loop ─────────────────────────────────────────
    print(f"\nLLM model: {args.model}  |  candidates per query: {args.k}\n")

    results = []
    n_dtw_correct = 0          # baseline: DTW rank-1 == gold
    n_llm_correct = 0          # arbiter:  LLM-picked candidate == gold
    n_recall_at_k = 0          # gold present in top-k candidate list
    n_llm_unparseable = 0
    n_llm_errors = 0           # LLM picked a number but not the gold

    t0 = time.time()
    for i, item in enumerate(test_items):
        gold = item["word"]
        noisy_ipa = item["noisy_ipa"]

        # 1. Get top-K candidates from v3 (bucket+DTW)
        candidates = idx.search(noisy_ipa, fs, k=args.k)
        if not candidates:
            results.append({
                "word": gold, "noisy_ipa": noisy_ipa,
                "dtw_top": None, "llm_pick": None, "gold_in_topk": False,
            })
            continue

        cand_words = [w for _, w in candidates]
        dtw_top = cand_words[0]
        gold_in_topk = gold in cand_words
        if gold_in_topk:
            n_recall_at_k += 1
        if dtw_top == gold:
            n_dtw_correct += 1

        # 2. Build prompt + call LLM
        prompt = build_prompt(noisy_ipa, candidates)
        reply = call_ollama(args.model, SYSTEM_PROMPT, prompt)
        pick_idx = parse_llm_response(reply, len(candidates))

        if pick_idx is None:
            n_llm_unparseable += 1
            llm_pick = None
        else:
            llm_pick = cand_words[pick_idx]
            if llm_pick == gold:
                n_llm_correct += 1
            else:
                n_llm_errors += 1

        results.append({
            "word": gold,
            "noisy_ipa": noisy_ipa,
            "dtw_top": dtw_top,
            "llm_pick": llm_pick,
            "llm_reply_raw": reply.strip()[:50],
            "gold_in_topk": gold_in_topk,
            "gold_rank_in_topk": cand_words.index(gold) if gold_in_topk else -1,
            "topk": cand_words[:10],
        })

        # Progress every 10 items
        if (i + 1) % 10 == 0 or i == len(test_items) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            done = i + 1
            print(f"  [{done:>3}/{len(test_items)}] {rate:.2f}/s  "
                  f"DTW@1={n_dtw_correct/done:.1%}  "
                  f"LLM@1={n_llm_correct/done:.1%}  "
                  f"recall@k={n_recall_at_k/done:.1%}")

    # ─── Report ───────────────────────────────────────────────────────
    n = len(test_items)
    dtw_acc = n_dtw_correct / n
    llm_acc = n_llm_correct / n
    recall = n_recall_at_k / n
    lift = llm_acc - dtw_acc

    print(f"\n{'=' * 64}")
    print(f"LLM-Arbiter results — n={n}, k={args.k}, model={args.model}")
    print(f"{'=' * 64}")
    print(f"  DTW rank-1 (baseline):    {dtw_acc:.1%}  ({n_dtw_correct}/{n})")
    print(f"  LLM-arbiter rank-1:       {llm_acc:.1%}  ({n_llm_correct}/{n})")
    print(f"  LLM lift over baseline:   {lift:+.1%}")
    print(f"  Recall@k (ceiling):       {recall:.1%}  "
          f"(words actually in top-{args.k})")
    print(f"  LLM unparseable replies:  {n_llm_unparseable}")
    print(f"  LLM picked wrong number:  {n_llm_errors}")
    print()
    if recall > 0:
        utilization = llm_acc / recall
        print(f"  LLM utilization of ceiling: {utilization:.1%}")
        print(f"    (of words that WERE in top-k, LLM picked correctly "
              f"{utilization:.0%})")

    print(f"\n{'─' * 64}")
    if lift >= 0.20:
        print(f"VERDICT: lift={lift:+.1%}. LLM arbiter is a strong win.")
        print(f"         Build full pipeline + test on real ZIPA audio.")
    elif lift >= 0.05:
        print(f"VERDICT: lift={lift:+.1%}. LLM arbiter helps but modestly.")
        print(f"         Try larger model / higher --k / better prompt.")
    else:
        print(f"VERDICT: lift={lift:+.1%}. LLM arbiter doesn't help.")
        print(f"         Either the LLM can't read IPA or candidates aren't "
              f"discriminable to it. Inspect disagreements.")

    # Sample disagreements where LLM beat DTW
    wins = [r for r in results
            if r["dtw_top"] != r["word"] and r["llm_pick"] == r["word"]]
    losses = [r for r in results
              if r["dtw_top"] == r["word"] and r["llm_pick"] != r["word"]
              and r["llm_pick"] is not None]
    if wins:
        print(f"\nSample LLM wins (DTW wrong, LLM correct), showing 5:")
        for r in wins[:5]:
            print(f"  {r['word']:<15} ipa={r['noisy_ipa']:<15} "
                  f"DTW->'{r['dtw_top']}'  LLM->'{r['llm_pick']}'")
    if losses:
        print(f"\nSample LLM losses (DTW correct, LLM wrong), showing 3:")
        for r in losses[:3]:
            print(f"  {r['word']:<15} ipa={r['noisy_ipa']:<15} "
                  f"DTW->'{r['dtw_top']}'  LLM->'{r['llm_pick']}'")

    # ─── Save ─────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "n": n, "k": args.k, "model": args.model,
                "bengali_only": args.bengali_only,
                "jsonl": str(args.jsonl) if args.jsonl else None,
            },
            "metrics": {
                "dtw_rank1_accuracy": dtw_acc,
                "llm_rank1_accuracy": llm_acc,
                "lift": lift,
                "recall_at_k": recall,
                "llm_utilization": llm_acc / recall if recall else 0.0,
                "n_unparseable": n_llm_unparseable,
                "n_llm_wrong_pick": n_llm_errors,
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nFull results -> {args.out}")


if __name__ == "__main__":
    main()
