"""
Recall@K test for physics-informed lattice (PhysASR Stage 2 sanity check).

The decisive question:
    Given a (possibly noisy) IPA phoneme sequence for a Bengali word,
    does articulatory-feature dictionary search return the correct word
    in the top-K candidates?

If recall@32 is < 90%, the framework has a fundamental ceiling problem
and no amount of LLM reranking will fix it. If recall@32 is > 95%, the
physics lattice is a viable candidate generator and the LLM-as-arbiter
hybrid is worth building.

This script does NOT need ZIPA, FastText, or an LLM. It tests *only*
Stage 2 (phonetic dictionary search) — the foundation of the whole
framework. Everything else is downstream of this number.

Pipeline:
    1. Build (or load cached) Bengali IPA dictionary index
       — uses bangla-ipa package (~50K-200K entries)
       — each entry: (word, ipa, panphon feature vector)
       — indexed by first-phoneme bucket for fast lookup
    2. Generate a test set of (word, ipa, noisy_ipa) triples
       — clean mode: just look up the word's own IPA (sanity ceiling)
       — synthetic noise: perturb IPA with realistic ZIPA-style errors
                          (substitution among feature-close phones,
                           occasional insertion/deletion)
       — jsonl mode: read your own (gold, ZIPA-output) pairs
    3. For each test item: score all dictionary candidates by DTW
       distance in panphon's 24-dim articulatory feature space
    4. Report recall@1, @5, @10, @32, @100

Usage:
    # 0. Make sure the P2G TSV lexicons exist (one-time, reuses existing
    #    pipeline). This builds Epitran + WikiPron + CMUDict TSVs in
    #    ~/.asr-pipeline/p2g/.
    uv run python scripts/build_p2g_dictionaries.py
    # (or, for the bigger bangla-academy dict:
    #  uv run python scripts/build_proper_bengali_dictionary.py)

    # 1. Build the panphon-feature index over those TSVs (~1 min)
    uv run python scripts/test_phys_lattice_recall.py --build

    # 2. Sanity check: clean recall (should be ~100%)
    uv run python scripts/test_phys_lattice_recall.py --mode clean --n 500

    # 3. Real test: synthetic ZIPA-like noise (the decisive number)
    uv run python scripts/test_phys_lattice_recall.py --mode noisy --n 500

    # 4. Test on your own (gold_word, zipa_ipa) pairs
    uv run python scripts/test_phys_lattice_recall.py \
        --mode jsonl --test-jsonl ./my_zipa_outputs.jsonl --n 500

    # Alternative source — rebuild from the bangla-dictionary pip packages
    uv run python scripts/test_phys_lattice_recall.py --build --source bangla-pkg

JSONL test format (one obj per line):
    {"word": "কফির", "gold_ipa": "kɔpir", "noisy_ipa": "kobir"}

Outputs:
    results/phys_lattice_recall.json
    Console: recall@K table + per-error-type breakdown
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


CACHE_DIR = Path.home() / ".asr-pipeline" / "phys_lattice"
CACHE_INDEX_FULL = CACHE_DIR / "ipa_dict_index.pkl"
CACHE_INDEX_BN = CACHE_DIR / "ipa_dict_index_bn.pkl"
P2G_TSV_DIR = Path.home() / ".asr-pipeline" / "p2g"


def is_bengali_word(w: str) -> bool:
    """True iff the word contains at least one Bengali codepoint."""
    return any('ঀ' <= c <= '৿' for c in w)


# ─── Panphon wrapper ─────────────────────────────────────────────────────

class FeatureSpace:
    """Cache panphon feature lookups. Phone string -> 24-dim vector."""

    def __init__(self):
        try:
            import panphon
        except ImportError:
            print("ERROR: panphon not installed. Run: uv pip install panphon")
            sys.exit(1)
        self.ft = panphon.FeatureTable()
        self._cache: dict[str, np.ndarray] = {}

    def vec(self, phone: str) -> np.ndarray | None:
        """Return 24-dim feature vector for a single IPA segment, or None."""
        if phone in self._cache:
            return self._cache[phone]
        feats = self.ft.word_to_vector_list(phone, numeric=True)
        if not feats:
            self._cache[phone] = None
            return None
        v = np.array(feats[0], dtype=np.float32)
        self._cache[phone] = v
        return v

    def segment(self, ipa: str) -> list[np.ndarray]:
        """IPA → list of 24-dim feature vectors. Robust to ZIPA-style
        space-separated token output and to standalone modifier diacritics.

        Strategy:
          1. Strip whitespace (ZIPA emits ['d', '̪', 'ʰ', 'ɔ'] which the
             tokenizer often joins with spaces — we want d̪ʰɔ).
          2. panphon.ipa_segs() groups base + diacritics into one segment
             (e.g. 'd̪ʰ' as a single 3-codepoint segment) — this is the
             "Option B" combine-modifier-into-previous-token behavior.
          3. If a segment doesn't featurize (e.g. an orphan diacritic
             that ipa_segs split off), fall back to "Option A": skip it
             and let DTW absorb the gap with a small insertion cost.
        """
        # 1. Normalize ZIPA-style tokenization (strip whitespace + join)
        clean = "".join(ipa.split())
        # 2. Let panphon segment with diacritic grouping
        segs = self.ft.ipa_segs(clean)
        out = []
        for s in segs:
            v = self.vec(s)
            if v is not None:
                out.append(v)
            # else: Option A — skip; DTW pays a small insertion cost
        return out

    def diagnose_segment(self, ipa: str) -> dict:
        """Return per-segment featurization status — for --diagnose mode."""
        clean = "".join(ipa.split())
        segs = self.ft.ipa_segs(clean)
        per_seg = []
        for s in segs:
            v = self.vec(s)
            per_seg.append({
                "segment": s,
                "n_codepoints": len(s),
                "featurized": v is not None,
            })
        return {
            "input": ipa,
            "normalized": clean,
            "n_segments": len(segs),
            "n_featurized": sum(1 for p in per_seg if p["featurized"]),
            "segments": per_seg,
        }

    def first_phone(self, ipa: str) -> str:
        clean = "".join(ipa.split())
        segs = self.ft.ipa_segs(clean)
        return segs[0] if segs else ""


# ─── Distance functions ─────────────────────────────────────────────────

def hamming_feat(a: np.ndarray, b: np.ndarray) -> float:
    """Per-feature mismatch count (panphon features are -1/0/+1)."""
    return float(np.sum(a != b))


def dtw_distance(seq_a: list[np.ndarray], seq_b: list[np.ndarray]) -> float:
    """Standard DTW with hamming feature distance as local cost.
    Returns total path cost (lower = closer)."""
    if not seq_a or not seq_b:
        return 1e6
    n, m = len(seq_a), len(seq_b)
    INF = 1e9
    # Rolling 2-row DP for memory efficiency
    prev = np.full(m + 1, INF, dtype=np.float64)
    curr = np.full(m + 1, INF, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[0] = INF
        for j in range(1, m + 1):
            cost = hamming_feat(seq_a[i - 1], seq_b[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m])


# ─── Dictionary index ───────────────────────────────────────────────────

class IPAIndex:
    """Holds (word, ipa, feature_seq) entries with first-phone bucketing."""

    def __init__(self):
        self.entries: list[tuple[str, str, list[np.ndarray]]] = []
        self.buckets: dict[str, list[int]] = defaultdict(list)
        self.fs: FeatureSpace | None = None  # set after build/load

    def add(self, word: str, ipa: str, feats: list[np.ndarray], first: str):
        idx = len(self.entries)
        self.entries.append((word, ipa, feats))
        self.buckets[first].append(idx)
        # Also bucket under aspirated/unaspirated equivalents for fuzzy lookup
        for variant in self._first_phone_variants(first):
            if variant != first:
                self.buckets[variant].append(idx)

    @staticmethod
    def _first_phone_variants(p: str) -> list[str]:
        """For first-phone lookup, treat aspirated≈unaspirated as same bucket."""
        if not p:
            return [p]
        # Strip aspiration (ʰ) for fuzzy lookup
        base = p.replace("ʰ", "")
        if base != p:
            return [p, base]
        # Add aspirated variant
        return [p, p + "ʰ"]

    def candidates(self, ipa: str, fs: FeatureSpace,
                   max_extra_buckets: int = 2) -> list[int]:
        """Return candidate dict-entry indices. Strategy: first-phone bucket
        plus a few feature-similar first-phone buckets."""
        first = fs.first_phone(ipa)
        cands = set(self.buckets.get(first, []))
        for v in self._first_phone_variants(first):
            cands.update(self.buckets.get(v, []))
        # If the bucket is small (rare phone), expand to all entries
        if len(cands) < 100:
            target_v = fs.vec(first)
            if target_v is not None:
                # Pull all bucket keys whose feature vec is within distance 3
                for k_phone, idxs in self.buckets.items():
                    kv = fs.vec(k_phone)
                    if kv is not None and hamming_feat(target_v, kv) <= 3.0:
                        cands.update(idxs)
        return list(cands)

    def search(self, ipa: str, fs: FeatureSpace, k: int = 32) -> list[tuple[float, str]]:
        """Top-K (distance, word) for this IPA query."""
        query = fs.segment(ipa)
        if not query:
            return []
        cand_idxs = self.candidates(ipa, fs)
        scored = []
        for idx in cand_idxs:
            word, _wipa, feats = self.entries[idx]
            d = dtw_distance(query, feats)
            scored.append((d, word))
        scored.sort(key=lambda x: x[0])
        return scored[:k]

    def __len__(self):
        return len(self.entries)


# ─── Dictionary builders ────────────────────────────────────────────────

def build_from_tsv(fs: FeatureSpace, tsv_dir: Path,
                   include_globs: tuple[str, ...] = ("*.tsv",),
                   exclude_files: tuple[str, ...] = ("word_frequencies.tsv",),
                   bengali_only: bool = False,
                   max_words: int = -1) -> IPAIndex:
    """Build IPA index from existing TSV lexicons in ~/.asr-pipeline/p2g/.

    Format: each .tsv file has lines of `word\\tipa` (the standard format
    produced by scripts/build_p2g_dictionaries.py and
    scripts/build_proper_bengali_dictionary.py).

    Combines all TSVs in the directory (Epitran + WikiPron + bangla-ipa +
    everyday lexicon, etc.) and de-duplicates by (word, ipa).
    """
    import csv as _csv
    if not tsv_dir.exists():
        print(f"ERROR: TSV dir not found: {tsv_dir}")
        print("Build it first with one of:")
        print("  uv run python scripts/build_p2g_dictionaries.py")
        print("  uv run python scripts/build_proper_bengali_dictionary.py")
        sys.exit(1)

    tsv_files = []
    for g in include_globs:
        tsv_files.extend(tsv_dir.glob(g))
    tsv_files = [p for p in tsv_files if p.name not in exclude_files]
    if not tsv_files:
        print(f"ERROR: no .tsv files in {tsv_dir}")
        sys.exit(1)

    print(f"Loading {len(tsv_files)} TSV lexicon(s) from {tsv_dir}"
          + (" [Bengali only]" if bengali_only else "") + ":")
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    n_skipped_lang = 0
    for tsv_path in sorted(tsv_files):
        n_file = 0
        with open(tsv_path, encoding="utf-8") as f:
            reader = _csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                word, ipa = row[0].strip(), row[1].strip()
                if not word or not ipa:
                    continue
                if bengali_only and not is_bengali_word(word):
                    n_skipped_lang += 1
                    continue
                key = (word, ipa)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
                n_file += 1
        print(f"  {tsv_path.name:<40} +{n_file} entries")
    if bengali_only and n_skipped_lang:
        print(f"  ({n_skipped_lang} non-Bengali entries filtered out)")

    print(f"  Total unique (word, ipa) pairs: {len(pairs)}")
    if max_words > 0:
        pairs = pairs[:max_words]

    idx = IPAIndex()
    n_failed = 0
    t0 = time.time()
    for i, (w, ipa) in enumerate(pairs):
        try:
            feats = fs.segment(ipa)
            if not feats:
                n_failed += 1
                continue
            first = fs.first_phone(ipa)
            idx.add(w, ipa, feats, first)
        except Exception:
            n_failed += 1
            continue
        if (i + 1) % 10000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{i + 1}/{len(pairs)}] indexed={len(idx)} "
                  f"failed={n_failed} ({rate:.0f} pairs/s)")

    print(f"\nBuilt index: {len(idx)} entries "
          f"({n_failed} failed panphon segmentation)")
    print(f"Bucket count: {len(idx.buckets)} unique first-phones")
    return idx


def build_from_bangla_ipa(fs: FeatureSpace, max_words: int = -1) -> IPAIndex:
    """Build IPA index from the bangla-ipa pip package + bangla-dictionary."""
    print("Loading bangla-dictionary + bangla-ipa packages...")
    try:
        from bangla_dictionary.dictionary import BanglaDictionary
        from bangla_ipa.ipa import BanglaIPATranslator
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install with: uv pip install bangla-dictionary bangla-ipa")
        sys.exit(1)

    bd = BanglaDictionary()
    ipa_tx = BanglaIPATranslator()

    # Get the word list (API varies across bangla-dictionary versions)
    print("Extracting word list...")
    words: list[str] = []
    for method_name in ("get_all_words", "all_words", "list_words", "words"):
        if hasattr(bd, method_name):
            try:
                attr = getattr(bd, method_name)
                result = attr() if callable(attr) else attr
                if result:
                    words = list(result)
                    print(f"  Used BanglaDictionary.{method_name}() -> {len(words)} words")
                    break
            except Exception:
                continue
    if not words:
        for attr_name in ("data", "_data", "dictionary", "_dictionary", "entries"):
            if hasattr(bd, attr_name):
                attr = getattr(bd, attr_name)
                if isinstance(attr, dict) and len(attr) > 100:
                    words = list(attr.keys())
                    print(f"  Used BanglaDictionary.{attr_name} -> {len(words)} words")
                    break
    if not words:
        print("Could not enumerate words. Inspect the API with:")
        print("  uv run python -c \"from bangla_dictionary.dictionary import "
              "BanglaDictionary; print(dir(BanglaDictionary()))\"")
        sys.exit(1)

    print(f"  {len(words)} words in raw dictionary")
    if max_words > 0:
        words = words[:max_words]

    idx = IPAIndex()
    n_failed = 0
    t0 = time.time()
    for i, w in enumerate(words):
        if not w or not isinstance(w, str):
            continue
        try:
            ipa = ipa_tx.translate(w)
            if not ipa:
                n_failed += 1
                continue
            feats = fs.segment(ipa)
            if not feats:
                n_failed += 1
                continue
            first = fs.first_phone(ipa)
            idx.add(w, ipa, feats, first)
        except Exception:
            n_failed += 1
            continue
        if (i + 1) % 5000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{i + 1}/{len(words)}] indexed={len(idx)} "
                  f"failed={n_failed} ({rate:.0f} words/s)")

    print(f"\nBuilt index: {len(idx)} entries "
          f"({n_failed} failed IPA conversion)")
    print(f"Bucket count: {len(idx.buckets)} unique first-phones")
    return idx


def save_index(idx: IPAIndex, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert feature lists to ndarray-of-ndarray for compact pickle
    serializable_entries = [(w, ipa, np.stack(feats) if feats else np.zeros((0, 24)))
                            for w, ipa, feats in idx.entries]
    with open(path, "wb") as f:
        pickle.dump({
            "entries": serializable_entries,
            "buckets": dict(idx.buckets),
        }, f)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"Saved index -> {path} ({size_mb:.1f} MB)")


def load_index(path: Path) -> IPAIndex:
    with open(path, "rb") as f:
        d = pickle.load(f)
    idx = IPAIndex()
    for w, ipa, feat_arr in d["entries"]:
        feats = [feat_arr[i] for i in range(feat_arr.shape[0])]
        idx.entries.append((w, ipa, feats))
    idx.buckets = defaultdict(list, d["buckets"])
    print(f"Loaded index: {len(idx)} entries from {path}")
    return idx


# ─── Synthetic noise injection (ZIPA-style errors) ──────────────────────

# Phone-substitution probabilities calibrated to typical CTC phoneme
# recognizer error patterns: aspiration confusion, voicing flips,
# place-of-articulation neighbors.
SUBSTITUTION_GROUPS = [
    ["p", "b"], ["pʰ", "p"], ["bʰ", "b"],
    ["t", "d"], ["tʰ", "t"], ["dʰ", "d"],
    ["ʈ", "ɖ"], ["ʈʰ", "ʈ"], ["ɖʰ", "ɖ"],
    ["k", "g"], ["kʰ", "k"], ["gʰ", "g"],
    ["tʃ", "dʒ"], ["tʃʰ", "tʃ"], ["dʒʰ", "dʒ"],
    ["s", "ʃ"], ["s", "z"],
    ["i", "e"], ["e", "ɛ"], ["o", "ɔ"], ["u", "o"],
    ["a", "ɔ"], ["a", "ɑ"],
    ["n", "ɳ"], ["m", "n"], ["r", "ɾ"], ["l", "r"],
]


def noise_inject(ipa: str, fs: FeatureSpace,
                 sub_rate: float = 0.20,
                 ins_rate: float = 0.03,
                 del_rate: float = 0.05,
                 rng: random.Random | None = None) -> str:
    """Inject ZIPA-style errors into clean IPA. Returns noisy IPA string."""
    if rng is None:
        rng = random.Random()

    # Build substitution lookup
    sub_map: dict[str, list[str]] = defaultdict(list)
    for group in SUBSTITUTION_GROUPS:
        for a in group:
            for b in group:
                if a != b:
                    sub_map[a].append(b)

    segs = fs.ft.ipa_segs(ipa)
    out_segs = []
    for s in segs:
        if rng.random() < del_rate:
            continue  # delete
        chosen = s
        if rng.random() < sub_rate and s in sub_map:
            chosen = rng.choice(sub_map[s])
        out_segs.append(chosen)
        if rng.random() < ins_rate and s in sub_map:
            out_segs.append(rng.choice(sub_map[s]))
    return "".join(out_segs)


# ─── Recall@K evaluation ────────────────────────────────────────────────

def evaluate_recall(idx: IPAIndex, fs: FeatureSpace,
                    test_items: list[dict],
                    k_values: list[int] = (1, 5, 10, 32, 100)) -> dict:
    """test_items: list of {word, gold_ipa, noisy_ipa}. Returns metrics dict."""
    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    rank_sum = 0
    n_ranked = 0
    n_total = len(test_items)
    n_in_dict = 0
    per_word_results = []

    t0 = time.time()
    print(f"\nEvaluating {n_total} items (search top-{max_k} per item)...")
    for i, item in enumerate(test_items):
        word = item["word"]
        query_ipa = item["noisy_ipa"]

        # Sanity: is the gold word even IN our dictionary?
        word_in_dict = any(w == word for w, _, _ in idx.entries)
        if word_in_dict:
            n_in_dict += 1

        results = idx.search(query_ipa, fs, k=max_k)
        result_words = [w for _, w in results]

        # Rank of gold word (-1 if not found in top-K)
        rank = -1
        for r, w in enumerate(result_words):
            if w == word:
                rank = r
                break
        if rank >= 0:
            rank_sum += rank
            n_ranked += 1

        for k in k_values:
            if rank >= 0 and rank < k:
                hits[k] += 1

        per_word_results.append({
            "word": word,
            "gold_ipa": item.get("gold_ipa", ""),
            "noisy_ipa": query_ipa,
            "rank": rank,
            "in_dict": word_in_dict,
            "top5": result_words[:5],
        })

        if (i + 1) % 50 == 0 or i == n_total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i + 1:>4}/{n_total}] {rate:.1f}/s   "
                  f"recall@1={hits[1] / (i + 1):.2%}  "
                  f"recall@32={hits[32] / (i + 1):.2%}")

    recall = {f"recall@{k}": hits[k] / n_total for k in k_values}
    metrics = {
        "n_test": n_total,
        "n_in_dict": n_in_dict,
        "dict_coverage": n_in_dict / n_total if n_total else 0,
        "mean_rank_when_found": rank_sum / n_ranked if n_ranked else -1,
        **recall,
    }
    return metrics, per_word_results


# ─── Test set generators ────────────────────────────────────────────────

def gen_clean_test(idx: IPAIndex, n: int, min_len: int = 0,
                   rng: random.Random | None = None) -> list[dict]:
    """Sanity ceiling: gold word IPA fed back as query. Should hit ~100%."""
    if rng is None:
        rng = random.Random(42)
    eligible = [(w, ipa) for w, ipa, _ in idx.entries if len(w) >= min_len]
    sample = rng.sample(eligible, min(n, len(eligible)))
    return [{"word": w, "gold_ipa": ipa, "noisy_ipa": ipa} for w, ipa in sample]


def gen_noisy_test(idx: IPAIndex, fs: FeatureSpace, n: int,
                   min_len: int = 0,
                   rng: random.Random | None = None) -> list[dict]:
    """Realistic: synthesize ZIPA-style noise on top of clean IPA."""
    if rng is None:
        rng = random.Random(42)
    eligible = [(w, ipa) for w, ipa, _ in idx.entries if len(w) >= min_len]
    sample = rng.sample(eligible, min(n, len(eligible)))
    return [{"word": w, "gold_ipa": ipa,
             "noisy_ipa": noise_inject(ipa, fs, rng=rng)}
            for w, ipa in sample]


def load_jsonl_test(path: Path) -> list[dict]:
    """Accept either {word, noisy_ipa: "d̪ʰɔn"} or
    {word, noisy_ipa_tokens: ["d", "̪", "ʰ", "ɔ", "n"]} (ZIPA-style)."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "word" not in d:
                continue
            if "noisy_ipa" not in d and "noisy_ipa_tokens" in d:
                # ZIPA-style: list of tokens → join (FeatureSpace.segment
                # also strips whitespace, so " ".join is safe)
                d["noisy_ipa"] = "".join(d["noisy_ipa_tokens"])
            if "noisy_ipa" not in d:
                continue
            items.append(d)
    return items


def run_diagnose(fs: FeatureSpace):
    """Print panphon segmentation for representative inputs so the user
    can VERIFY that diacritic grouping works before trusting recall@K."""
    samples = [
        # (label, raw IPA — possibly with ZIPA-style spaces)
        ("dict-style aspirated dental",     "d̪ʰɔnnobad"),
        ("ZIPA-style space-tokenized",      "d ̪ ʰ ɔ n n o b a d"),
        ("aspirated retroflex + nasal",     "ʈʰɔɳɖa"),
        ("affricate + aspiration",          "tʃʰatra"),
        ("voiced aspirated",                "bʰalo"),
        ("English code-switch (vaccine)",   "vækʃin"),
        ("standalone modifier orphan",      "ʰ"),
    ]
    print(f"\n{'─' * 70}")
    print(f"DIAGNOSE: panphon segmentation + featurization")
    print(f"{'─' * 70}")
    for label, ipa in samples:
        d = fs.diagnose_segment(ipa)
        status = "OK" if d["n_featurized"] == d["n_segments"] else "PARTIAL"
        print(f"\n[{status}] {label}")
        print(f"  input:      {ipa!r}")
        if d["normalized"] != ipa:
            print(f"  normalized: {d['normalized']!r}")
        print(f"  segments ({d['n_featurized']}/{d['n_segments']} featurized):")
        for p in d["segments"]:
            mark = "✓" if p["featurized"] else "✗"
            print(f"    {mark} {p['segment']!r} ({p['n_codepoints']} codepoints)")

    print(f"\n{'─' * 70}")
    print("Interpretation:")
    print("  ✓ If 'dict-style' and 'ZIPA-style' produce IDENTICAL segments,")
    print("    Option B (panphon's diacritic grouping) works for both inputs.")
    print("  ✗ If a base+diacritic splits into separate segments and the")
    print("    diacritic doesn't featurize, the script falls back to Option A")
    print("    (skip) — DTW absorbs the gap. Recall may still be high but")
    print("    the framework is doing more lifting via DTW alignment.")
    print(f"{'─' * 70}\n")


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--build", action="store_true",
                   help="Build the IPA dictionary index (one-time, ~1 min)")
    p.add_argument("--diagnose", action="store_true",
                   help="Print panphon segmentation for sample inputs and "
                        "exit. Run this BEFORE trusting recall numbers — it "
                        "verifies that diacritics group correctly.")
    p.add_argument("--source", choices=["tsv", "bangla-pkg"], default="tsv",
                   help="Where to read the lexicon from. 'tsv' (default) reads "
                        "all *.tsv files in --tsv-dir (the format produced by "
                        "scripts/build_p2g_dictionaries.py). 'bangla-pkg' "
                        "rebuilds from the bangla-dictionary + bangla-ipa "
                        "pip packages.")
    p.add_argument("--tsv-dir", type=Path, default=P2G_TSV_DIR,
                   help=f"Directory of word-tab-ipa TSVs (default: {P2G_TSV_DIR})")
    p.add_argument("--bengali-only", action="store_true",
                   help="Filter dictionary to entries containing Bengali "
                        "characters (drops CMUDict English). Uses a separate "
                        "cache file so toggling doesn't clobber the full index.")
    p.add_argument("--max-words", type=int, default=-1,
                   help="Cap dict size while building (debug only)")

    p.add_argument("--mode", choices=["clean", "noisy", "jsonl"],
                   default="noisy",
                   help="Test set source. clean=ceiling, noisy=synthetic ZIPA "
                        "noise, jsonl=your own (word, noisy_ipa) pairs")
    p.add_argument("--test-jsonl", type=Path, default=None,
                   help="Path to JSONL with {word, gold_ipa?, noisy_ipa} (mode=jsonl)")
    p.add_argument("--n", type=int, default=500, help="Number of test items")
    p.add_argument("--min-len", type=int, default=2,
                   help="Restrict to words with at least N characters")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=Path("results/phys_lattice_recall.json"))

    args = p.parse_args()

    fs = FeatureSpace()

    # ─── Diagnose-only fast path ──────────────────────────────────────
    if args.diagnose:
        run_diagnose(fs)
        return

    # ─── Load or build the dictionary index ───────────────────────────
    cache_path = CACHE_INDEX_BN if args.bengali_only else CACHE_INDEX_FULL
    if args.build or not cache_path.exists():
        if not args.build and not cache_path.exists():
            print(f"No cached index at {cache_path} — building now.")
        if args.source == "tsv":
            idx = build_from_tsv(fs, args.tsv_dir,
                                 bengali_only=args.bengali_only,
                                 max_words=args.max_words)
        else:
            idx = build_from_bangla_ipa(fs, max_words=args.max_words)
        save_index(idx, cache_path)
    else:
        idx = load_index(cache_path)

    if len(idx) == 0:
        print("ERROR: empty index. Aborting.")
        sys.exit(1)

    # ─── Build the test set ───────────────────────────────────────────
    rng = random.Random(args.seed)
    if args.mode == "clean":
        test_items = gen_clean_test(idx, args.n, args.min_len, rng)
        print(f"\nMode: CLEAN (sanity ceiling, gold IPA = query IPA)")
    elif args.mode == "noisy":
        test_items = gen_noisy_test(idx, fs, args.n, args.min_len, rng)
        print(f"\nMode: NOISY (synthetic ZIPA-style errors)")
    else:
        if not args.test_jsonl:
            print("ERROR: --mode jsonl requires --test-jsonl PATH")
            sys.exit(1)
        test_items = load_jsonl_test(args.test_jsonl)
        if args.n > 0:
            test_items = test_items[:args.n]
        print(f"\nMode: JSONL ({args.test_jsonl}, {len(test_items)} items)")

    # ─── Run evaluation ───────────────────────────────────────────────
    metrics, per_item = evaluate_recall(idx, fs, test_items)

    # ─── Report ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"PhysASR recall@K — {args.mode.upper()} mode, n={metrics['n_test']}")
    print(f"{'=' * 60}")
    print(f"Dictionary coverage: {metrics['n_in_dict']}/{metrics['n_test']} "
          f"({metrics['dict_coverage']:.1%}) gold words present")
    print(f"Mean rank when found: {metrics['mean_rank_when_found']:.2f}")
    print()
    for k in [1, 5, 10, 32, 100]:
        v = metrics[f"recall@{k}"]
        bar = "█" * int(v * 40)
        verdict = "✓" if v >= 0.95 else ("~" if v >= 0.80 else "✗")
        print(f"  recall@{k:<4} {v:.2%}  {bar}  {verdict}")

    # Verdict
    print(f"\n{'─' * 60}")
    r32 = metrics["recall@32"]
    if r32 >= 0.95:
        print(f"VERDICT: recall@32={r32:.1%} ≥ 95%. Physics lattice IS a viable")
        print(f"         candidate generator. Build the LLM-arbiter hybrid.")
    elif r32 >= 0.85:
        print(f"VERDICT: recall@32={r32:.1%} (85-95%). Promising but borderline.")
        print(f"         Investigate failure cases before committing.")
    else:
        print(f"VERDICT: recall@32={r32:.1%} < 85%. Framework has a ceiling.")
        print(f"         No amount of reranking will fix this. Reconsider.")

    # Sample failures for debugging
    failures = [r for r in per_item if r["rank"] < 0 or r["rank"] >= 32]
    if failures:
        print(f"\nSample failures (recall@32 misses, showing 5):")
        for r in failures[:5]:
            print(f"  word={r['word']:<20} ipa_query={r['noisy_ipa']:<20}")
            print(f"    in_dict={r['in_dict']}  rank={r['rank']}  "
                  f"top5={r['top5']}")

    # Save full results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "per_item": per_item,
                   "config": vars(args)}, f, ensure_ascii=False,
                  indent=2, default=str)
    print(f"\nFull results -> {args.out}")


if __name__ == "__main__":
    main()
