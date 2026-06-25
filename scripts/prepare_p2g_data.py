"""Merge real (ZIPA-IPA, Bengali) pairs + synthetic code-switch
top-up into the P2G training set.

Inputs:
  Primary  — lora_data_ipa/lora_dataset_full_train.jsonl
             (~11,500 real pairs from FLEURS + banspeech + bengali_ai;
             schema: {src, subsource, id, ipa, bengali, english})
  Primary val — lora_data_ipa/lora_dataset_full_val.jsonl

  Secondary — synthetic code-switch pairs generated on the fly from
             data/gt_lm/gt_corpus.txt via gen_synthetic_p2g logic.
             These cover coastal dialect + English code-switch tokens
             absent from FLEURS/banspeech.

Outputs:
  data/p2g/train.jsonl    primary + synthetic mixed
  data/p2g/val.jsonl      primary val (clean held-out, no synthetic)
  data/p2g/vocab.json     {ipa_vocab: [...], text_vocab: [...]}
                          + special tokens (<pad>, <bos>, <eos>, <unk>)
  data/p2g/stats.json     dataset statistics for sanity
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LORA_DIR = Path("/mnt/c/Users/Noureini Sayouti/OneDrive/Documents"
                "/asr_pipeline/lora_data_ipa")
LORA_TRAIN = LORA_DIR / "lora_dataset_full_train.jsonl"
LORA_VAL = LORA_DIR / "lora_dataset_full_val.jsonl"
GT_CORPUS = REPO / "data" / "gt_lm" / "gt_corpus.txt"
OUT_DIR = REPO / "data" / "p2g"
SYNTH_PATH = OUT_DIR / "synth_codeswitch.jsonl"
SEED = 0
SYNTH_N_AUGMENT = 3  # noisy samples per source sentence


def normalize_row(r: dict) -> dict | None:
    """Map any input row to {ipa, text} schema."""
    ipa = r.get("ipa", "").strip()
    text = (r.get("text") or r.get("bengali") or "").strip()
    if not ipa or not text:
        return None
    return {"ipa": ipa, "text": text}


def load_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open(encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return rows


def build_vocab(rows: list[dict]) -> tuple[list[str], list[str]]:
    """IPA vocab: space-split tokens. Text vocab: character-level."""
    ipa_tokens: Counter[str] = Counter()
    text_chars: Counter[str] = Counter()
    for r in rows:
        for tok in r["ipa"].split():
            ipa_tokens[tok] += 1
        for ch in r["text"]:
            text_chars[ch] += 1
    ipa_vocab = sorted(ipa_tokens)
    text_vocab = sorted(text_chars)
    return ipa_vocab, text_vocab


def main() -> None:
    if not LORA_TRAIN.exists():
        sys.exit(f"missing {LORA_TRAIN}\n"
                 f"(expected on OneDrive — verify path)")
    if not LORA_VAL.exists():
        sys.exit(f"missing {LORA_VAL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load primary (real ZIPA pairs)
    print("loading primary (real ZIPA pairs) ...", flush=True)
    raw_train = load_jsonl(LORA_TRAIN)
    raw_val = load_jsonl(LORA_VAL)
    primary_train = [n for r in raw_train if (n := normalize_row(r))]
    primary_val = [n for r in raw_val if (n := normalize_row(r))]
    print(f"  train: {len(primary_train)} pairs  "
          f"(from {len(raw_train)} raw)")
    print(f"  val:   {len(primary_val)} pairs")

    # 2. Generate synthetic code-switch top-up from gt_corpus.txt
    print("\ngenerating synthetic code-switch top-up ...", flush=True)
    if not SYNTH_PATH.exists():
        if not GT_CORPUS.exists():
            print(f"  WARN: missing {GT_CORPUS}; skipping synthetic")
            synth_rows: list[dict] = []
        else:
            cmd = [
                sys.executable, str(REPO / "scripts"
                                    / "gen_synthetic_p2g.py"),
                "--text-files", str(GT_CORPUS),
                "--out", str(SYNTH_PATH),
                "--n-augment", str(SYNTH_N_AUGMENT),
                "--p-swap", "0.15",  # heavier code-switch swap
            ]
            print(f"  running: {' '.join(cmd[1:])}")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  synthetic gen failed:\n{r.stderr[-500:]}")
                synth_rows = []
            else:
                print(f"  synthetic gen ok ({r.stdout.splitlines()[-1]})")
                synth_rows = load_jsonl(SYNTH_PATH)
    else:
        print(f"  reusing existing {SYNTH_PATH}")
        synth_rows = load_jsonl(SYNTH_PATH)
    synth_train = [n for r in synth_rows if (n := normalize_row(r))]
    print(f"  synthetic: {len(synth_train)} pairs")

    # 3. Mix
    rng = random.Random(SEED)
    train = primary_train + synth_train
    rng.shuffle(train)
    val = primary_val
    print(f"\nfinal train: {len(train)} "
          f"(primary {len(primary_train)} + synth {len(synth_train)})")
    print(f"final val:   {len(val)}")

    # 4. Vocab + stats
    print("\nbuilding vocab ...")
    ipa_vocab, text_vocab = build_vocab(train + val)
    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    print(f"  IPA tokens   : {len(ipa_vocab)}")
    print(f"  text chars   : {len(text_vocab)}")

    ipa_lens = [len(r["ipa"].split()) for r in train]
    text_lens = [len(r["text"]) for r in train]
    ipa_lens.sort()
    text_lens.sort()
    p50 = lambda xs: xs[len(xs) // 2]
    p95 = lambda xs: xs[int(0.95 * len(xs))]
    print(f"  ipa len  p50={p50(ipa_lens):3d}  "
          f"p95={p95(ipa_lens):3d}  max={ipa_lens[-1]}")
    print(f"  text len p50={p50(text_lens):3d}  "
          f"p95={p95(text_lens):3d}  max={text_lens[-1]}")

    # 5. Write outputs
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"
    vocab_path = OUT_DIR / "vocab.json"
    stats_path = OUT_DIR / "stats.json"

    with train_path.open("w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as fh:
        for r in val:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    vocab_path.write_text(json.dumps({
        "specials": specials,
        "ipa_vocab": ipa_vocab,
        "text_vocab": text_vocab,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps({
        "n_train": len(train),
        "n_val": len(val),
        "n_primary_train": len(primary_train),
        "n_synth_train": len(synth_train),
        "ipa_vocab_size": len(ipa_vocab),
        "text_vocab_size": len(text_vocab),
        "ipa_len_p50": p50(ipa_lens),
        "ipa_len_p95": p95(ipa_lens),
        "text_len_p50": p50(text_lens),
        "text_len_p95": p95(text_lens),
    }, indent=2), encoding="utf-8")

    print(f"\nwrote:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {vocab_path}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    main()
