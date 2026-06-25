"""
Parse the human-transcribed GT .docx files into a corpus suitable
for training a domain-bootstrapped KenLM, plus a unigram lexicon
for pyctcdecode.

Honest design choices (driven by stress-test risk #1: GT may be
edited toward standard Bengali):
  * Strip speaker markers (প্র:/উ:) and parenthetical metadata
    (সময়, নিজেদের কথোপকথন, ভেবে সময় নিয়ে, ...).
  * DO NOT normalize/standardize Bengali. Preserve whatever form
    the human transcriber wrote (dialect or otherwise).
  * Preserve code-switched English (NGO, loan, ...) as-is.
  * Split on Bengali full-stop ( । ) and ASCII . / ? — one sentence
    per line for lmplz.

Outputs (in --out-dir):
  gt_corpus.txt           one sentence per line, tokens space-sep
  gt_unigrams.txt         unique word forms (one per line)
  gt_unigrams_counts.tsv  word<TAB>count (for lexicon analysis)
  gt_stats.json           per-file + total word counts

Usage:
  uv run python scripts/parse_gt_for_lm.py \
      --folder "for testing (Noureini)" --out-dir /tmp/gt_lm
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


# Bengali speaker markers + parenthetical metadata to strip.
_RE_SPEAKER_TAG = re.compile(r"^(প্র:|উ:|Q:|A:)\s*", re.M)
_RE_PARENTHETICAL = re.compile(r"\([^)]{0,200}\)")
_RE_BRACKET = re.compile(r"\[[^\]]{0,200}\]")
_RE_WS = re.compile(r"\s+")
# split on Bengali daanda, period, or question mark
_RE_SENT_SPLIT = re.compile(r"[।.?!]+\s*")
# token = any run of non-space chars (preserves Bengali +
# code-switched English as separate tokens)
_RE_TOK = re.compile(r"\S+")


def docx_to_paragraphs(path: Path) -> list[str]:
    import docx  # type: ignore
    d = docx.Document(str(path))
    return [p.text for p in d.paragraphs if p.text.strip()]


def clean_paragraph(t: str) -> str:
    # strip speaker tags (line-anchored)
    t = _RE_SPEAKER_TAG.sub("", t)
    # strip parentheticals/brackets — these are transcriber notes,
    # not speech
    t = _RE_PARENTHETICAL.sub(" ", t)
    t = _RE_BRACKET.sub(" ", t)
    t = _RE_WS.sub(" ", t).strip()
    return t


def split_sentences(t: str) -> list[str]:
    parts = _RE_SENT_SPLIT.split(t)
    return [p.strip() for p in parts if p.strip()]


def tokenize(s: str) -> list[str]:
    return _RE_TOK.findall(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path,
                    default=Path("for testing (Noureini)"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1]
                            / "data" / "gt_lm")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    docs = sorted(p for p in args.folder.iterdir()
                  if p.suffix.lower() == ".docx"
                  and "ranscription" in p.name.lower())
    if not docs:
        raise SystemExit(f"no transcription .docx in {args.folder}")

    sents: list[str] = []
    counts: collections.Counter = collections.Counter()
    stats = {}
    for d in docs:
        n_p = n_s = n_t = 0
        for p in docx_to_paragraphs(d):
            clean = clean_paragraph(p)
            if not clean:
                continue
            n_p += 1
            for s in split_sentences(clean):
                toks = tokenize(s)
                if not toks:
                    continue
                sents.append(" ".join(toks))
                counts.update(toks)
                n_s += 1
                n_t += len(toks)
        stats[d.name] = {"paragraphs": n_p, "sentences": n_s,
                          "tokens": n_t}
        print(f"  {d.name}: {n_p} para, {n_s} sent, {n_t} tok")

    corpus = args.out_dir / "gt_corpus.txt"
    corpus.write_text("\n".join(sents) + "\n", encoding="utf-8")
    print(f"\n-> {corpus} : {len(sents)} sentences, "
          f"{sum(c for _, c in counts.items())} tokens, "
          f"{len(counts)} unique words")

    uni = args.out_dir / "gt_unigrams.txt"
    uni.write_text("\n".join(sorted(counts)) + "\n", encoding="utf-8")
    print(f"-> {uni} : {len(counts)} unique unigrams")

    cnt = args.out_dir / "gt_unigrams_counts.tsv"
    with cnt.open("w", encoding="utf-8") as fh:
        for w, c in sorted(counts.items(), key=lambda x: -x[1]):
            fh.write(f"{w}\t{c}\n")
    print(f"-> {cnt} : ranked by frequency")
    print(f"  top 20: {[w for w, _ in counts.most_common(20)]}")

    (args.out_dir / "gt_stats.json").write_text(
        json.dumps({"files": stats,
                    "total_sentences": len(sents),
                    "total_tokens": sum(c for _, c in counts.items()),
                    "unique_words": len(counts)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
