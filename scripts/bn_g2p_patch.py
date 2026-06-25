"""Bangla inherent-vowel (schwa) deletion patch for epitran ben-Beng output.

epitran's known weak point for Bangla is inherent-vowel deletion: it
emits the inherent vowel where native phonology deletes it. Word-final
deletion is near-categorical (EXCEPT after conjunct clusters, where the
vowel is retained: শান্ত /ʃan.to/), so the word-final rule is safe and
high-yield. Medial deletion is conditioned by morpheme boundaries and
lexical exceptions that are NOT recoverable from orthography or the
phone string — the pipeline has no morphological segmentation — so the
medial pass exists only behind an explicit flag, is conservative, and
its output is labeled as REQUIRING LINGUIST REVIEW, never applied
silently.

The inherent-vowel surface symbol is detected EMPIRICALLY from epitran
(transliterate bare consonants, take the majority trailing vowel) rather
than assumed, since epitran may emit ɔ, o, or a.
"""

from __future__ import annotations

import re
import unicodedata

HOSONTO = "্"
KHANDA_TA = "ৎ"          # ৎ — inherently vowel-less, never patched

# Bengali consonant letters (ক..হ plus ড়/ঢ়/য়)
_BN_CONSONANTS = set(
    [chr(c) for c in range(0x0995, 0x09BA)]
    + ["ড়", "ঢ়", "য়"]
)

_VOWELISH = re.compile(r"^[aeiouɔoæɛəɪʊɐ]")


def is_vowel_seg(seg: str) -> bool:
    base = unicodedata.normalize("NFD", seg)
    base = "".join(c for c in base if unicodedata.category(c) != "Mn")
    return bool(_VOWELISH.match(base))


def detect_inherent(epi) -> str:
    """Empirically determine epitran's inherent-vowel symbol by
    transliterating bare consonants and majority-voting the trailing
    vowel segment."""
    import panphon
    ft = panphon.FeatureTable()
    votes: dict[str, int] = {}
    for c in ["ক", "ব", "ত", "ম", "ল", "ন"]:
        try:
            segs = ft.ipa_segs(epi.transliterate(c))
        except Exception:
            continue
        if len(segs) >= 2 and is_vowel_seg(segs[-1]):
            votes[segs[-1]] = votes.get(segs[-1], 0) + 1
    if not votes:
        return "ɔ"          # literature default for Bangla অ
    return max(votes, key=votes.get)


def ends_bare_consonant(word: str) -> bool:
    """Orthographic word ends in a consonant letter carrying the
    inherent vowel (no matra, no hosonto, not khanda-ta)."""
    return bool(word) and word[-1] in _BN_CONSONANTS


def cluster_final(word: str) -> bool:
    """Final consonant is part of a conjunct (preceded by hosonto) —
    inherent vowel is RETAINED in this context."""
    return len(word) >= 3 and word[-2] == HOSONTO


def patch_segs(
    word: str,
    segs: list[str],
    inherent: str,
    medial: bool = False,
) -> tuple[list[str], dict]:
    """Apply schwa-deletion to one word's phone segments.

    Returns (patched_segs, info) where info records what fired:
      {"final": bool, "medial_sites": [indices deleted]}.

    Word-final rule (default): delete the final inherent vowel when the
    orthography ends in a bare consonant and that consonant is not
    conjunct-final. Guard len>=3 so single-consonant citation forms
    keep their vowel.

    Medial rule (--medial only, NEEDS LINGUIST REVIEW): delete inherent
    vowel in the classic V C ə C V context, one non-cascading
    left-to-right pass. Condition-blind to morphology — over-deletion
    risk is real; off by default.
    """
    info = {"final": False, "medial_sites": []}
    out = list(segs)

    if (len(out) >= 3
            and out[-1] == inherent
            and ends_bare_consonant(word)
            and not cluster_final(word)
            and not is_vowel_seg(out[-2])):
        out = out[:-1]
        info["final"] = True

    if medial:
        i = 2
        while i < len(out) - 2:
            if (out[i] == inherent
                    and is_vowel_seg(out[i - 2])
                    and not is_vowel_seg(out[i - 1])
                    and not is_vowel_seg(out[i + 1])
                    and is_vowel_seg(out[i + 2])):
                info["medial_sites"].append(i)
                out = out[:i] + out[i + 1:]
                i += 2          # non-cascading
            else:
                i += 1

    return out, info


def medial_candidate_sites(segs: list[str], inherent: str) -> int:
    """Count medial V C ə C V sites WITHOUT deleting — used by the
    audit to size the medial question for the consultant."""
    n = 0
    for i in range(2, len(segs) - 2):
        if (segs[i] == inherent
                and is_vowel_seg(segs[i - 2])
                and not is_vowel_seg(segs[i - 1])
                and not is_vowel_seg(segs[i + 1])
                and is_vowel_seg(segs[i + 2])):
            n += 1
    return n
