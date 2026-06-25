"""Elegant titu × Whisper fusion via phonetic cross-validation — no GT.

Insight: titu (CTC, phonetically faithful, garbled) and regional-Whisper
(seq2seq, fluent, but silently smooths content words like flood→"bon")
are INDEPENDENT ASR systems. Where they agree, confidence is high. Where
they disagree on a content word, Whisper may have confabulated — that's
the span to flag. Disagreement between independent systems is the
confidence signal; no ground truth required.

Two transcripts of the same audio spell/segment differently, so we align
them in a neutral PHONETIC space (epitran IPA → romanized key), classify
each region (agree / disagree / whisper-only / titu-only), and emit:

  1. A confidence-annotated Whisper transcript — fluent text kept, but
     disagreement spans marked  word⟨titu: alt⟩  and Whisper-only spans
     marked  {wonly: ...}  (over-production candidates).
  2. Per-utterance + corpus agreement stats (a GT-free quality proxy).
  3. (optional) An LLM fusion prompt per utterance that hands BOTH
     transcripts + the flagged disagreements to a frontier model to
     adjudicate — preferring titu's phonetic evidence for proper nouns
     / domain terms, Whisper's fluency for structure.

Usage:
    uv run python scripts/asr_fuse.py --n 14
    uv run python scripts/asr_fuse.py --n 14 --emit-llm-prompts
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("fuse")

DATA_DIR = Path("data/w2v_train_longform")
JSONLS = [
    ("07_Shatkhira",
     DATA_DIR / "07_Shatkhira_CA-MMI_URBAN_Male_07_0202_1716.longform.jsonl"),
    ("15_Sunamganj",
     DATA_DIR / "15_Sunamganj_CA-MMI_URBAN_Male_15_0602_1635.longform.jsonl"),
]

_IPA2ROMAN = {
    "ʈ": "t", "ɖ": "d", "ɽ": "r", "ʃ": "s", "ʂ": "s", "ɕ": "s",
    "ŋ": "n", "ɲ": "n", "ɳ": "n", "ɡ": "g", "ɣ": "g", "ʒ": "j",
    "ɔ": "o", "æ": "a", "ɛ": "e", "ə": "a", "ɪ": "i", "ʊ": "u",
    "j": "y", "ʋ": "w", "β": "b", "ɸ": "f", "ʰ": "h", "ʱ": "h",
    "x": "k", "q": "k", **{c: c for c in "abcdefghiklmnoprstuvwyz"},
}
_BENGALI = re.compile(r"[ঀ-৿]")
_LATIN = re.compile(r"[A-Za-z]")


def _ipa_roman(ipa: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFD", ipa):
        if unicodedata.category(ch) == "Mn":
            continue
        out.append(_IPA2ROMAN.get(ch, ""))
    return re.sub(r"[^a-z]", "", "".join(out))


def make_keyer():
    """Return word -> phonetic key. Bengali via epitran IPA; Latin
    code-switch kept lowercased (so 'NGO'/'loan' compare across both)."""
    import epitran
    import panphon
    epi = epitran.Epitran("ben-Beng")
    ft = panphon.FeatureTable()
    cache: dict[str, str] = {}

    def key(word: str) -> str:
        w = word.strip()
        if w in cache:
            return cache[w]
        if _BENGALI.search(w):
            try:
                k = _ipa_roman("".join(ft.ipa_segs(epi.transliterate(w))))
            except Exception:
                k = re.sub(r"\W", "", w.lower())
        else:
            k = re.sub(r"[^a-z0-9]", "", w.lower())
        cache[w] = k or "·"
        return cache[w]
    return key


def _lev(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i]
        for j, bj in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ai != bj)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def fuse(titu: str, whisper: str, key, var_thresh=0.34):
    """Phonetically align titu vs whisper words. Returns
    (annotated_whisper, stats)."""
    tw = [w for w in re.split(r"\s+", titu.strip()) if w]
    ww = [w for w in re.split(r"\s+", whisper.strip()) if w]
    tk = [key(w) for w in tw]
    wk = [key(w) for w in ww]
    sm = SequenceMatcher(a=tk, b=wk, autojunk=False)

    parts, agree, disagree, wonly, tonly = [], 0, 0, 0, 0
    flags = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        wseg = ww[j1:j2]
        tseg = tw[i1:i2]
        if op == "equal":
            agree += (j2 - j1)
            parts.extend(wseg)
        elif op == "replace":
            # phonetic distance between the two spans
            d = _lev("".join(tk[i1:i2]), "".join(wk[j1:j2]))
            if d < var_thresh:        # same word, spelling/dialect variant
                agree += (j2 - j1)
                parts.extend(wseg)
            else:                     # genuine disagreement -> flag
                disagree += (j2 - j1)
                alt = " ".join(tseg)
                parts.append(f"{' '.join(wseg)}⟨titu: {alt}⟩")
                flags.append((" ".join(wseg), alt, round(d, 2)))
        elif op == "insert":          # whisper has words titu doesn't
            wonly += (j2 - j1)
            parts.append("{wonly: " + " ".join(wseg) + "}")
        elif op == "delete":          # titu has words whisper dropped
            tonly += (i2 - i1)
            parts.append("{titu-only: " + " ".join(tseg) + "}")

    n_w = max(1, len(ww))
    stats = dict(
        n_whisper=len(ww), n_titu=len(tw),
        agree=agree, disagree=disagree, wonly=wonly, tonly=tonly,
        agree_rate=round(agree / n_w, 3),
        flags=flags)
    return " ".join(parts), stats


# ── ASR runners (reuse the working simple-compare logic) ──────────────
def select(n, seed=0):
    import random
    rng = random.Random(seed)
    picks, per = [], max(1, n // len(JSONLS))
    for group, path in JSONLS:
        recs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if 3.0 <= (r.get("end", 0) - r.get("start", 0)) <= 15.0:
                    recs.append(r)
        for r in rng.sample(recs, min(per, len(recs))):
            r["_group"] = group
            picks.append(r)
    return picks


def run_titu(recs, device):
    import torch
    import nemo.collections.asr as nemo_asr
    logger.info("Loading titu ...")
    m = nemo_asr.models.ASRModel.from_pretrained(
        "hishab/titu_stt_bn_fastconformer")
    if device == "cuda" and torch.cuda.is_available():
        m = m.to(torch.device("cuda"))
    m.eval()
    out = []
    for r in recs:
        o = m.transcribe([r["wav"]], batch_size=1, verbose=False)[0]
        out.append(o if isinstance(o, str) else getattr(o, "text", str(o)))
    del m
    torch.cuda.empty_cache()
    return out


def run_whisper(hf_id, recs, device):
    import torch
    from transformers import pipeline
    logger.info("Loading Whisper %s ...", hf_id)
    dev = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
    asr = pipeline("automatic-speech-recognition", model=hf_id,
                   device=dev, chunk_length_s=30)
    out = []
    for r in recs:
        try:
            out.append((asr(r["wav"]).get("text") or "").strip())
        except Exception as e:
            out.append(f"[ERROR: {type(e).__name__}]")
    del asr
    torch.cuda.empty_cache()
    return out


_LLM_PROMPT = """\
Two ASR systems transcribed the SAME Bengali audio clip from a climate-
disaster survey in coastal Bangladesh (enumerator interviews a respondent
about cyclones, floods, NGOs, loans, agriculture).

- titu is phonetically faithful but garbled and may run words together.
- whisper is fluent and well-structured but sometimes SMOOTHS a content
  word into a plausible-but-wrong one (e.g. it once turned "বন্যা"=flood
  into "বোন না").

Produce one clean Bengali transcript: keep whisper's fluency and
sentence/turn structure, but at any point where the two disagree on a
content word — especially proper nouns, place/bank names, numbers, and
domain terms (flood, cyclone, NGO names) — prefer titu's phonetic reading
if it is the more plausible word in context. If neither is recoverable,
mark the span [অস্পষ্ট]. Do not invent content. Output only the merged
Bengali transcript.

titu:    {titu}
whisper: {whisper}

Merged:"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--whisper", default=
                    "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--emit-llm-prompts", action="store_true",
                    help="also write per-utt fusion prompts for a "
                         "frontier LLM adjudication pass")
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/asr_fuse.md"))
    args = ap.parse_args()

    recs = select(args.n)
    logger.info("Selected %d utterances", len(recs))
    titu = run_titu(recs, args.device)
    whisper = run_whisper(args.whisper, recs, args.device)
    key = make_keyer()

    lines = ["# titu × Whisper phonetic cross-validation (no GT)\n"]
    lines.append("Agreement between two independent ASR systems = "
                 "confidence. ⟨titu: …⟩ marks content-word disagreements "
                 "(Whisper-smoothing candidates); {wonly: …} marks "
                 "Whisper-only spans (over-production candidates).\n")
    tot_agree = tot_w = tot_flag = tot_wonly = 0
    prompts = []
    for i, (r, t, w) in enumerate(zip(recs, titu, whisper), 1):
        annotated, st = fuse(t, w, key)
        tot_agree += st["agree"]
        tot_w += st["n_whisper"]
        tot_flag += st["disagree"]
        tot_wonly += st["wonly"]
        lines.append(f"## {i}. [{r['_group']}]  agree={st['agree_rate']:.0%}"
                     f"  flags={len(st['flags'])}  wonly={st['wonly']}")
        lines.append(f"- titu   : {t}")
        lines.append(f"- whisper: {w}")
        lines.append(f"- **fused**: {annotated}")
        if st["flags"]:
            fl = "; ".join(f"'{a}'≠titu'{b}'(d{d})"
                           for a, b, d in st["flags"])
            lines.append(f"- flags : {fl}")
        lines.append("")
        prompts.append(_LLM_PROMPT.format(titu=t, whisper=w))

    corpus_agree = tot_agree / max(1, tot_w)
    hdr = (f"- **Corpus agreement: {corpus_agree:.0%}** of Whisper tokens "
           f"corroborated by titu | disagreements flagged: {tot_flag} | "
           f"Whisper-only tokens: {tot_wonly}\n")
    lines.insert(2, hdr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    if args.emit_llm_prompts:
        pp = args.output.with_suffix(".llm_prompts.txt")
        pp.write_text(
            "\n\n========================================\n\n".join(prompts),
            encoding="utf-8")
        logger.info("LLM fusion prompts -> %s", pp)
    logger.info("Report -> %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
