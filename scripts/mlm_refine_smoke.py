"""Smoke test: BanglaBERT MLM refinement of raw titu CTC output.

Strategy (no titu confidence available):
  1. Tokenize titu text with BanglaBERT's WordPiece tokenizer
  2. For each iteration:
     - Compute BanglaBERT's own confidence in each token via
       leave-one-out masking (mask each position singly, read
       the model's P(original_token | rest of sentence))
     - Mask the K positions where the model's confidence in the
       original token is LOWEST (i.e., positions BERT thinks
       should be something else)
     - Fill all masks at once with greedy argmax
  3. After N iterations, print a 3-column comparison.

We print 20 random samples from stem 15's titu_lm output. The eyeball
test: does refined output look more coherent? Does coastal dialect
survive (forms like 'নাই', 'করতেছি', 'তাইনে')? Or does it standardize
to West-Bengal / Wikipedia register ('নেই', 'করছি', 'তিনি')?
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
TITU_FILE = (REPO / "outputs/real_eval_titu_excellent"
             / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635"
             / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.titu_lm.jsonl")
N_SAMPLES = 20
N_ITER = 3
MASK_FRAC = 0.30   # mask 30% of positions per iteration
SEED = 0
MODEL_NAME = "sagorsarker/bangla-bert-base"  # classic BERT-MLM
# alt: "csebuetnlp/banglabert" (ELECTRA — different API)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mlm(name: str):
    print(f"loading {name} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForMaskedLM.from_pretrained(name).to(device).eval()
    return tok, mdl


@torch.no_grad()
def loo_confidence(text: str, tok, mdl) -> tuple[list[str], list[float]]:
    """Leave-one-out confidence. For each token position, mask it and
    read P(original | context). Returns (tokens, confidences)."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    ids = enc["input_ids"][0].tolist()
    mask_id = tok.mask_token_id
    special = set(tok.all_special_ids)
    confs = [1.0] * len(ids)
    # Batch up the leave-one-out forward passes
    batch_inputs = []
    positions = []
    for i, t in enumerate(ids):
        if t in special:
            continue
        seq = list(ids)
        seq[i] = mask_id
        batch_inputs.append(seq)
        positions.append(i)
    if not batch_inputs:
        return [tok.convert_ids_to_tokens(t) for t in ids], confs
    inp = torch.tensor(batch_inputs, device=device)
    logits = mdl(inp).logits  # (B, T, V)
    probs = torch.softmax(logits, dim=-1)
    for b, i in enumerate(positions):
        p = probs[b, i, ids[i]].item()
        confs[i] = p
    toks = tok.convert_ids_to_tokens(ids)
    return toks, confs


@torch.no_grad()
def fill_masked(ids: list[int], tok, mdl) -> list[int]:
    inp = torch.tensor([ids], device=device)
    logits = mdl(inp).logits[0]
    preds = logits.argmax(dim=-1)
    out = list(ids)
    mask_id = tok.mask_token_id
    for i, t in enumerate(ids):
        if t == mask_id:
            out[i] = int(preds[i].item())
    return out


def refine(text: str, tok, mdl, n_iter: int, mask_frac: float) -> str:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    ids = enc["input_ids"][0].tolist()
    mask_id = tok.mask_token_id
    special = set(tok.all_special_ids)
    for _ in range(n_iter):
        # confidence pass
        cur_text = tok.decode(ids, skip_special_tokens=True)
        _, confs = loo_confidence(cur_text, tok, mdl)
        # Re-tokenize because loo_confidence re-encoded; align by length
        enc = tok(cur_text, return_tensors="pt", truncation=True,
                  max_length=256)
        ids = enc["input_ids"][0].tolist()
        if len(confs) != len(ids):
            # fall back: trim
            confs = confs[: len(ids)]
        # Pick K positions with LOWEST confidence to mask
        candidates = [(i, c) for i, c in enumerate(confs)
                      if ids[i] not in special]
        candidates.sort(key=lambda x: x[1])
        k = max(1, int(len(candidates) * mask_frac))
        for i, _ in candidates[:k]:
            ids[i] = mask_id
        # Fill in one shot
        ids = fill_masked(ids, tok, mdl)
    return tok.decode(ids, skip_special_tokens=True)


def main() -> None:
    if not TITU_FILE.exists():
        raise SystemExit(f"missing titu output: {TITU_FILE}")
    rows = [json.loads(l) for l in
            TITU_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(SEED)
    samples = random.sample(rows, min(N_SAMPLES, len(rows)))

    tok, mdl = load_mlm(MODEL_NAME)

    print("\n" + "=" * 100)
    for i, r in enumerate(samples, 1):
        orig = r["text"].strip()
        if not orig:
            continue
        try:
            refined = refine(orig, tok, mdl,
                              n_iter=N_ITER, mask_frac=MASK_FRAC)
        except Exception as e:
            refined = f"[ERR: {type(e).__name__}: {e}]"
        print(f"\n[{i:2d}] idx={r['idx']}  t={r.get('start',0):.1f}s")
        print(f"  TITU    : {orig[:160]}")
        print(f"  REFINED : {refined[:160]}")
    print("\n" + "=" * 100)
    print(f"\nmodel={MODEL_NAME}  iter={N_ITER}  mask_frac={MASK_FRAC}")
    print("Eyeball checklist:")
    print("  (1) Is refined more coherent than titu?")
    print("  (2) Did coastal dialect survive (নাই, করতেছি, তাইনে)?")
    print("  (3) Did code-switch (vaccine/NGO/project) survive?")


if __name__ == "__main__":
    main()
