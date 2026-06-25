"""Training-free GER (generative error correction) guard for fusion output.

The fusion LLM (Gemma-3-12B) free-generates a reconciled Bengali [bn] from the
witness pool — titu (LITERAL) + Whisper N-best (FLUENT). For a 12B the
literature says UNCONSTRAINED free-text correction is the wrong mode: it can
drift off the acoustic evidence (hallucinate) and, worst for a survey, fabricate
NUMBERS. This module is the deterministic, character-distance-based guard that
constrains that output WITHOUT retraining:

  1. closest-to-N-best snap — if [bn] is far from EVERY witness, the LLM went
     off the rails → snap to the consensus (medoid) witness instead.
  2. number-mismatch flag — a digit-number in [bn] that no witness supports
     (when a witness also uses digits) → flag for human QA.
  3. agreement-gating — witnesses agree but [bn] drifted, or witnesses disagree
     (a genuinely hard segment) → flag.

All distances are over BENGALI GRAPHEME CLUSTERS (base char + its combining
vowel signs / virama), not raw codepoints: char-level Levenshtein over Bengali
over-penalises combining marks and makes the thresholds meaningless.

Pure-python, no model, no extra deps — unit-tested in scripts/test_ger_guard.py.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Bengali digits ০-৯ → ASCII 0-9 so number VALUES compare regardless of script.
_BENGALI_DIGIT_MAP = {ord(b): ord(a)
                      for a, b in zip("0123456789", "০১২৩৪৫৬৭৮৯")}
# Punctuation / whitespace to drop before comparison: titu emits none, Gemma
# adds dandas/commas — comparing those would inflate the distance spuriously.
_PUNCT = re.compile(r"[\s।,.;:!?\-—–…\"'`()\[\]{}/০-৯0-9]+")
_ZW = ("‍", "‌")  # ZWJ / ZWNJ — attach to the preceding cluster


def grapheme_clusters(s: str) -> list[str]:
    """Segment into Bengali grapheme clusters: a base char plus any following
    combining marks (vowel signs Mc/Mn, virama, nukta) and zero-width joiners.
    Approximate (not full UAX-29) but stable and dependency-free."""
    out: list[str] = []
    for ch in s:
        combining = (unicodedata.combining(ch)
                     or unicodedata.category(ch) in ("Mn", "Mc")
                     or ch in _ZW)
        if out and combining:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def _normalize(s: str) -> str:
    """Lower-noise form for comparison: drop punctuation, whitespace AND digits
    (numbers are checked separately by `numbers()`, so leaving them in would
    double-count). Bengali digits are first mapped to ASCII for `numbers()`."""
    return _PUNCT.sub("", (s or ""))


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance over two token lists (here: grapheme clusters)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if not la:
        return lb
    if not lb:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def ndist(a: str, b: str) -> float:
    """Normalised grapheme edit distance in [0, 1]; 0 = identical content."""
    ga = grapheme_clusters(_normalize(a))
    gb = grapheme_clusters(_normalize(b))
    if not ga and not gb:
        return 0.0
    return edit_distance(ga, gb) / max(len(ga), len(gb), 1)


def numbers(s: str) -> set[str]:
    """Digit-number VALUES in the text (Bengali digits normalised to ASCII)."""
    return set(re.findall(r"\d+", (s or "").translate(_BENGALI_DIGIT_MAP)))


def _char_ok(ch: str) -> bool:
    """Allowed in a Bengali [bn]: the Bengali block (incl. Bengali digits),
    ASCII (Latin code-switches + digits + punctuation), zero-width joiners, and
    any Unicode punctuation/separator/control. ANY other script (Cyrillic, CJK,
    Devanagari letters, …) means a cross-lingual LLM hallucination."""
    if 0x0980 <= ord(ch) <= 0x09FF:           # Bengali block
        return True
    if ch.isascii():                          # Latin letters/digits/punct/space
        return True
    if ch in _ZW:
        return True
    return unicodedata.category(ch)[0] in ("P", "Z", "C")


def foreign_tokens(s: str) -> list[str]:
    """Whitespace tokens containing a character outside the allowed
    Bengali+Latin scripts — e.g. Cyrillic «хозяйства», CJK. These are LLM
    hallucinations, never transcribed speech."""
    return [tok for tok in (s or "").split()
            if any(not _char_ok(c) for c in tok)]


_ABSTAIN_BN = "[অস্পষ্ট]"
_ABSTAIN_EN = "[unclear]"


def _blank_fabricated(text: str, fabricated: set[str], marker: str) -> str:
    """Replace each digit-run whose VALUE is in `fabricated` with `marker`.
    Operates on the raw text so the original (Bengali or ASCII) digits are
    matched, but compares normalised values."""
    if not text or not fabricated:
        return text

    def repl(m: "re.Match") -> str:
        val = m.group(0).translate(_BENGALI_DIGIT_MAP)
        return marker if val in fabricated else m.group(0)

    return re.sub(r"[0-9০-৯]+", repl, text)


@dataclass
class GerVerdict:
    bn: str                                  # guarded [bn] (snapped/abstained)
    en: str = ""                             # guarded [en]
    snapped: bool = False                    # True → snapped, [en] blanked
    reasons: list[str] = field(default_factory=list)
    min_dist: float = 0.0                    # [bn] vs NEAREST witness
    witness_agree: float | None = None       # titu vs whisper-top (None if N/A)


def assess(
    bn: str,
    en: str,
    titu: str,
    whisper_nbest: list[str] | None,
    *,
    snap: bool = False,
    snap_tau: float = 0.72,
    agree_tau: float = 0.30,
    disagree_tau: float = 0.55,
    deviate_tau: float = 0.55,
) -> GerVerdict:
    """Constrain + audit one fused [bn] against its witness pool.

    The snap target is the witness NEAREST the fused [bn] (literal closest-to-
    N-best) — NOT an inter-witness medoid. Medoid-of-4 is biased: the 3 Whisper
    beams are correlated near-duplicates, so it almost always picks Whisper and
    discards titu exactly when titu is the only witness that heard a rare
    word/number/dialect form. Snapping to the nearest hypothesis keeps titu an
    equal candidate and leaves titu-vs-Whisper arbitration to the fusion LLM
    (the snap's only job is to rescue EGREGIOUS hallucination, not to vote).

    `snap` is off by default: run once in observe-mode to collect the min_dist
    distribution, calibrate `snap_tau` to where good and bad segments separate,
    THEN enable. A snap_tau picked by intuition clobbers good reconciliations.
    """
    witnesses = [w for w in ([titu] + list(whisper_nbest or []))
                 if (w or "").strip()]
    v = GerVerdict(bn=bn, en=en)
    if not (bn or "").strip() or not witnesses:
        return v

    pairs = [(ndist(bn, w), w) for w in witnesses]
    v.min_dist, nearest = min(pairs, key=lambda x: x[0])
    if (titu or "").strip() and whisper_nbest and (whisper_nbest[0] or "").strip():
        v.witness_agree = ndist(titu, whisper_nbest[0])

    # ── closest-to-N-best snap (only when enabled AND egregiously far) ──
    if v.min_dist > snap_tau:
        if snap:
            v.bn, v.en, v.snapped = nearest, "", True   # stale en blanked
            v.reasons.append(f"snapped→nearest({v.min_dist:.0%})")
            return v
        v.reasons.append(f"would-snap({v.min_dist:.0%})")  # observe-mode hint

    # ── fabricated-number ABSTAIN (deterministic, not threshold-gated) ──
    # A digit in [bn] that no witness supports — but ONLY when a witness also
    # uses digits (else it's a legit word→digit transcription, not fabrication,
    # e.g. titu "বাইশ" → Gemma "২০২২"). Fabricated → abstain ([অস্পষ্ট]), not
    # retain-and-flag: a wrong age must not flow downstream as a real number.
    wit_nums: set[str] = set()
    for w in witnesses:
        wit_nums |= numbers(w)
    fabricated = (numbers(bn) - wit_nums) if wit_nums else set()
    if fabricated:
        v.bn = _blank_fabricated(v.bn, fabricated, _ABSTAIN_BN)
        v.en = _blank_fabricated(v.en, fabricated, _ABSTAIN_EN)
        v.reasons.append("number-abstain:" + ",".join(sorted(fabricated)))

    # ── foreign-script ABSTAIN (deterministic) ──
    # A [bn] token outside Bengali+Latin (e.g. Cyrillic "хозяйства", CJK) is an
    # LLM cross-lingual hallucination — blank it (don't ship it) and flag for
    # review. Catches what temperature/prompt rules can't guarantee.
    foreign = foreign_tokens(v.bn)
    if foreign:
        for t in foreign:
            v.bn = v.bn.replace(t, _ABSTAIN_BN)
            v.en = v.en.replace(t, _ABSTAIN_EN)  # if it leaked into [en] too
        v.reasons.append("foreign-script:" + ",".join(foreign[:5]))

    # ── agreement-gating review signals (flag only; no rewrite) ──
    # NOTE the 0.30–0.55 band on witness_agree is deliberately "no opinion":
    # moderate witness disagreement is normal in fusion and not worth flagging.
    if v.witness_agree is not None:
        if v.witness_agree > disagree_tau:
            v.reasons.append(f"witness-disagree({v.witness_agree:.0%})")
        elif v.witness_agree < agree_tau and v.min_dist > deviate_tau:
            v.reasons.append(f"deviated-from-consensus({v.min_dist:.0%})")

    return v
