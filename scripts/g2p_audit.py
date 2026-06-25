"""G2P inherent-vowel audit — artifact-size estimate BEFORE any re-probe.

Standalone, CPU-only, zero GPU time. Quantifies how badly epitran's
inherent-vowel over-production contaminates the Stage 0 probe labels,
against the phonology-literature expectation that word-final inherent-
vowel deletion in Bangla is near-categorical outside conjunct-final
contexts (overall deletion across all sites ~60-70%).

Outputs (outputs/g2p_audit.md):
  1. Empirically detected inherent-vowel symbol
  2. Word-final over-production rate: among word tokens whose
     orthography ends in a bare consonant (non-conjunct), how often
     epitran emits a final inherent vowel. Expectation: deletion is
     near-categorical here, so emission rate >> ~10% = systematic bug.
  3. Token-level phone distribution before/after the word-final patch
     (projected — no probes touched)
  4. Medial candidate-site count (sized for the linguist, NOT applied)
  5. Linguist-review block: 30 most frequent affected word types with
     before/after IPA

Also writes both G2P caches so probe runs diff cleanly:
  outputs/g2p_cache.unpatched.json
  outputs/g2p_cache.final.json

Usage:
    uv run python scripts/g2p_audit.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bn_g2p_patch import (  # noqa: E402
    cluster_final,
    detect_inherent,
    ends_bare_consonant,
    medial_candidate_sites,
    patch_segs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("g2p_audit")

DATA_DIR = Path("data/w2v_train_longform")
JSONLS = [
    DATA_DIR / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl",
    DATA_DIR / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.longform.jsonl",
]

_BENGALI = re.compile(r"[ঀ-৿]")
_TOKEN_CLEAN = re.compile(r"[^A-Za-zঀ-৿০-৯]")


def main() -> int:
    import epitran
    import panphon

    ft = panphon.FeatureTable()
    epi = epitran.Epitran("ben-Beng")

    inherent = detect_inherent(epi)
    logger.info("Detected inherent-vowel symbol: %r", inherent)

    # ── Collect Bengali word tokens from both interviews ─────────────
    word_tokens: Counter = Counter()
    for path in JSONLS:
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["gt_text"]
                for raw in text.split():
                    tok = _TOKEN_CLEAN.sub("", raw)
                    if tok and _BENGALI.search(tok) \
                            and not re.search(r"[০-৯0-9]", tok):
                        word_tokens[tok] += 1
    n_types = len(word_tokens)
    n_tokens = sum(word_tokens.values())
    logger.info("Collected %d word types / %d tokens", n_types, n_tokens)

    # ── G2P every type once; build unpatched + final-patched caches ──
    unpatched: dict[str, list[str]] = {}
    patched: dict[str, list[str]] = {}
    affected: list[tuple[str, int, str, str]] = []   # word, count, pre, post

    # Audit accumulators (token-weighted)
    final_sites = 0          # tokens ending in bare consonant (non-conjunct)
    final_emitted = 0        # ... where epitran emitted the inherent vowel
    cluster_sites = 0        # conjunct-final tokens (vowel correctly retained)
    cluster_emitted = 0
    medial_sites_tok = 0     # medial VCəCV candidate sites (token-weighted)
    pre_dist: Counter = Counter()
    post_dist: Counter = Counter()
    g2p_fail = 0

    for word, count in word_tokens.items():
        try:
            segs = ft.ipa_segs(epi.transliterate(word))
        except Exception:
            segs = []
        if not segs:
            g2p_fail += count
            continue
        unpatched[word] = segs
        new, info = patch_segs(word, segs, inherent, medial=False)
        patched[word] = new

        bare = ends_bare_consonant(word)
        clus = cluster_final(word)
        if bare and not clus:
            final_sites += count
            if segs[-1] == inherent:
                final_emitted += count
        elif bare and clus:
            cluster_sites += count
            if segs[-1] == inherent:
                cluster_emitted += count

        medial_sites_tok += medial_candidate_sites(segs, inherent) * count
        for s in segs:
            pre_dist[s] += count
        for s in new:
            post_dist[s] += count
        if info["final"]:
            affected.append((word, count, " ".join(segs), " ".join(new)))

    affected.sort(key=lambda t: -t[1])
    over_rate = final_emitted / max(1, final_sites)
    n_pre = sum(pre_dist.values())
    n_post = sum(post_dist.values())
    inh_pre = pre_dist[inherent] / max(1, n_pre)
    inh_post = post_dist[inherent] / max(1, n_post)

    # ── Verdict heuristic ────────────────────────────────────────────
    # Word-final deletion outside conjuncts is near-categorical in
    # Bangla: emission >> 10% in that context = systematic artifact.
    if over_rate > 0.50:
        verdict = ("CATASTROPHIC over-production — vowel-tier AUCs from "
                   "the unpatched run are uninterpretable; patch + re-run "
                   "required, and medial review by a linguist is likely "
                   "worth the budget.")
    elif over_rate > 0.10:
        verdict = ("SYSTEMATIC over-production — vowel tier contaminated; "
                   "patch + re-run worthwhile.")
    else:
        verdict = ("Mild — epitran mostly deletes correctly here; the "
                   "vowel tier's low AUCs are NOT explained by the "
                   "word-final artifact (look to acoustics/alignment).")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "g2p_cache.unpatched.json").write_text(
        json.dumps(unpatched, ensure_ascii=False), encoding="utf-8")
    (out_dir / "g2p_cache.final.json").write_text(
        json.dumps(patched, ensure_ascii=False), encoding="utf-8")

    lines = ["# G2P inherent-vowel audit (pre-probe decision gate)\n"]
    lines.append(f"- Inherent-vowel symbol (empirical): **{inherent!r}**")
    lines.append(f"- Word types: {n_types} | tokens: {n_tokens} | "
                 f"G2P failures: {g2p_fail} tokens")
    lines.append(
        f"- **Word-final over-production rate: {over_rate:.1%}** "
        f"({final_emitted}/{final_sites} bare-consonant-final non-conjunct "
        f"tokens got an emitted inherent vowel; phonology expects "
        f"near-categorical deletion here)")
    lines.append(
        f"- Conjunct-final tokens (vowel correctly retained): "
        f"{cluster_sites} — epitran emitted vowel on {cluster_emitted} "
        f"(retention here is CORRECT)")
    lines.append(
        f"- Inherent-vowel share of all phone tokens: "
        f"**{inh_pre:.1%} → {inh_post:.1%}** after word-final patch")
    lines.append(
        f"- Medial VCəCV candidate sites (token-weighted, NOT applied — "
        f"for linguist review): **{medial_sites_tok}** "
        f"(~{medial_sites_tok / max(1, n_tokens):.2f} per word token)")
    lines.append(f"\n**Verdict: {verdict}**\n")

    lines.append("## Top phone tokens before → after (word-final patch only)")
    lines.append("| phone | before | after |")
    lines.append("|---|---|---|")
    for p, c in pre_dist.most_common(15):
        lines.append(f"| {p} | {c} | {post_dist.get(p, 0)} |")

    lines.append("\n## Linguist-review block — 30 most frequent patched "
                 "word types")
    lines.append("| word | tokens | epitran IPA | after final-deletion |")
    lines.append("|---|---|---|---|")
    for word, count, pre, post in affected[:30]:
        lines.append(f"| {word} | {count} | {pre} | {post} |")

    lines.append("\nCaches written: outputs/g2p_cache.unpatched.json, "
                 "outputs/g2p_cache.final.json — probe runs against each "
                 "diff cleanly.")

    report = out_dir / "g2p_audit.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))
    logger.info("Full audit: %s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
