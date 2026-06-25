"""Generate synthetic (IPA, text) pairs for the P2G transformer.

Pipeline per source sentence:
  1. Tokenize on whitespace
  2. Per word: detect script
       Bengali  → epitran('ben-Beng')
       Latin    → epitran('eng-Latn')  [+ acronym override dict]
  3. Char-tokenize each word's IPA into single Unicode codepoints,
     space-separated  (matches ZIPA output format)
  4. Concatenate words with ▁ word-boundary marker
  5. Apply DANP noise (del / sub / ins) at configurable rates
  6. Optionally inject code-switch swaps (Bengali word → English equiv)

Output: JSONL {"ipa": "...", "text": "...", "noise": "..."}
  - `ipa`   the noisy IPA stream (matches what ZIPA will produce on audio)
  - `text`  the clean target Bengali+English text
  - `noise` summary of operations applied (for debugging)

Usage:
  uv run python scripts/gen_synthetic_p2g.py \\
      --text-files data/gt_lm/gt_corpus.txt \\
      --out data/p2g_synth/pairs.jsonl \\
      --n-augment 5

  # produces n_augment noisy-IPA samples per source sentence
  # → with gt_corpus 74K tokens and n_augment=5, ~30K pairs
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import epitran

REPO = Path(__file__).resolve().parents[1]

# ---- script detection -----------------------------------------------

_BENG = re.compile(r"[ঀ-৿]")
_LATN = re.compile(r"[A-Za-z]")


def script_of(word: str) -> str:
    """Return 'bn', 'en', or 'other'."""
    bn = len(_BENG.findall(word))
    en = len(_LATN.findall(word))
    if bn > en:
        return "bn"
    if en > 0:
        return "en"
    return "other"


# ---- code-switch swap dictionary -------------------------------------

def load_swap_dict() -> dict[str, str]:
    """Load Bengali→Latin translit table to use as code-switch
    candidates. Forward direction (Bengali word → English equivalent)."""
    p = REPO / "data" / "translit.json"
    if not p.exists():
        return {}
    fwd = json.loads(p.read_text(encoding="utf-8"))
    return {bn: lat for bn, lat in fwd.items()
            if isinstance(lat, str) and lat}


# ---- acronyms: epitran(eng-Latn) fails on letter-by-letter words ----
# We override the most common survey-domain acronyms. The IPA below
# matches the way these are spoken aloud (letter-by-letter for caps).
_ACRONYM_IPA = {
    "NGO":  "ɛn d͡ʒi oʊ",
    "NID":  "ɛn aɪ di",
    "MP":   "ɛm pi",
    "OK":   "oʊ keɪ",
    "ID":   "aɪ di",
    "USA":  "ju ɛs eɪ",
    "GPS":  "d͡ʒi pi ɛs",
    "WHO":  "dʌbəlju eɪt͡ʃ oʊ",
    "BRAC": "bɹæk",
    "UN":   "ju ɛn",
}


def english_ipa(word: str, en_epi: epitran.Epitran) -> str:
    w = word.strip()
    if w.upper() in _ACRONYM_IPA:
        return _ACRONYM_IPA[w.upper()]
    if w.isupper() and len(w) <= 5 and len(w) >= 2:
        # Unknown short acronym → spell out letter by letter via epitran
        return " ".join(en_epi.transliterate(c) for c in w)
    try:
        return en_epi.transliterate(w.lower())
    except Exception:
        return ""


# ---- char tokenization (match ZIPA's per-codepoint format) ----------

# Punctuation/digits that should NEVER appear in the IPA stream.
# ZIPA emits clean phonemes; matching its distribution matters.
_IPA_DROP = set(",.?!|:;-—()[]{}\"'`/\\<>" + "0123456789"
                + "০১২৩৪৫৬৭৮৯")


def char_tokenize(ipa_str: str) -> list[str]:
    """ZIPA output is space-separated single codepoints. We match that.
    Drops punctuation/digits to keep the IPA stream phoneme-pure."""
    return [c for c in ipa_str if not c.isspace() and c not in _IPA_DROP]


# ---- text → IPA token sequence --------------------------------------

WORD_BOUND = "▁"


def text_to_ipa_tokens(sentence: str,
                       bn_epi: epitran.Epitran,
                       en_epi: epitran.Epitran,
                       swap_dict: dict[str, str],
                       p_swap: float,
                       rng: random.Random) -> tuple[list[str], str]:
    """Return (ipa_tokens, target_text).

    target_text is the *post-swap* text the model should learn to
    produce. ipa_tokens is the clean (un-noised) IPA token sequence.
    """
    out_words: list[str] = []
    out_tokens: list[str] = [WORD_BOUND]
    for w in sentence.split():
        # optional code-switch swap (Bengali word -> English equivalent)
        if (script_of(w) == "bn"
                and w in swap_dict
                and rng.random() < p_swap):
            w = swap_dict[w]
        s = script_of(w)
        if s == "bn":
            ipa_str = bn_epi.transliterate(w)
        elif s == "en":
            ipa_str = english_ipa(w, en_epi)
        else:
            # punctuation / digits — skip from IPA but keep in text
            out_words.append(w)
            continue
        if not ipa_str:
            continue
        for c in char_tokenize(ipa_str):
            out_tokens.append(c)
        out_tokens.append(WORD_BOUND)
        out_words.append(w)
    return out_tokens, " ".join(out_words)


# ---- DANP noise -----------------------------------------------------

def apply_danp_noise(tokens: list[str], vocab: list[str],
                     p_del: float, p_sub: float, p_ins: float,
                     rng: random.Random) -> tuple[list[str], dict]:
    out: list[str] = []
    counts = {"del": 0, "sub": 0, "ins": 0, "keep": 0}
    for t in tokens:
        # Never touch the word boundary marker
        if t == WORD_BOUND:
            out.append(t)
            continue
        r = rng.random()
        if r < p_del:
            counts["del"] += 1
            continue
        r2 = rng.random()
        if r2 < p_ins:
            out.append(rng.choice(vocab))
            counts["ins"] += 1
        if rng.random() < p_sub:
            out.append(rng.choice(vocab))
            counts["sub"] += 1
        else:
            out.append(t)
            counts["keep"] += 1
    return out, counts


# ---- main -----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-files", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=REPO / "data" / "p2g_synth" / "pairs.jsonl")
    ap.add_argument("--n-augment", type=int, default=5,
                    help="noisy IPA samples per source sentence")
    ap.add_argument("--p-del",  type=float, default=0.07)
    ap.add_argument("--p-sub",  type=float, default=0.07)
    ap.add_argument("--p-ins",  type=float, default=0.05)
    ap.add_argument("--p-swap", type=float, default=0.05,
                    help="prob of Bengali→English code-switch swap")
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--max-words", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("loading epitran engines (bn-Beng, eng-Latn) ...")
    bn_epi = epitran.Epitran("ben-Beng")
    en_epi = epitran.Epitran("eng-Latn")
    swap_dict = load_swap_dict()
    print(f"  code-switch swap dict: {len(swap_dict)} entries")

    # First pass: read all sentences + collect IPA vocab
    sentences: list[str] = []
    for f in args.text_files:
        for ln in f.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            # Split into sentences (Bengali sentence terminator । or .?!)
            for sent in re.split(r"[।.?!]+\s*", ln):
                sent = sent.strip()
                wc = len(sent.split())
                if args.min_words <= wc <= args.max_words:
                    sentences.append(sent)
    print(f"  source sentences after filter: {len(sentences)}")

    # Vocab discovery pass (clean IPA only)
    vocab_counter: Counter[str] = Counter()
    clean_pairs: list[tuple[list[str], str]] = []
    for s in sentences:
        toks, text = text_to_ipa_tokens(
            s, bn_epi, en_epi, swap_dict,
            p_swap=0.0, rng=rng)
        if len(toks) < 4:
            continue
        clean_pairs.append((toks, text))
        for t in toks:
            if t != WORD_BOUND:
                vocab_counter[t] += 1
    vocab = sorted(vocab_counter)
    print(f"  IPA vocab size: {len(vocab)}  "
          f"top: {vocab_counter.most_common(10)}")
    print(f"  clean pairs collected: {len(clean_pairs)}")

    # Second pass: produce N noisy samples per clean pair
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    op_totals = Counter()
    with args.out.open("w", encoding="utf-8") as fh:
        for clean_toks, text in clean_pairs:
            for _ in range(args.n_augment):
                # Re-do text generation with code-switch swap on
                noisy_text_toks, swapped_text = text_to_ipa_tokens(
                    text, bn_epi, en_epi, swap_dict,
                    p_swap=args.p_swap, rng=rng)
                # Apply DANP noise on top of the (post-swap) clean IPA
                noisy, ops = apply_danp_noise(
                    noisy_text_toks, vocab,
                    args.p_del, args.p_sub, args.p_ins, rng)
                op_totals.update(ops)
                fh.write(json.dumps({
                    "ipa": " ".join(noisy),
                    "text": swapped_text,
                    "noise": ops,
                }, ensure_ascii=False) + "\n")
                n_written += 1
            # also include one clean (no-noise) pair so the model sees
            # the un-corrupted mapping
            fh.write(json.dumps({
                "ipa": " ".join(clean_toks),
                "text": text,
                "noise": {"clean": True},
            }, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"\nwrote {n_written} pairs -> {args.out}")
    print(f"  total ops across noisy samples: "
          f"del={op_totals['del']}, sub={op_totals['sub']}, "
          f"ins={op_totals['ins']}, keep={op_totals['keep']}")


if __name__ == "__main__":
    main()
