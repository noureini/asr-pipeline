"""Production ASR-combination bin resolver with a TOKEN-TYPE GATE.

Resolves one confusion-network bin (the candidates from N-best ROVER / CNC over
≥2 systems) into a single decision. The gate is the whole point and is an
EXPLICIT, UN-BYPASSABLE branch:

  • PROSE bins            → confidence-weighted vote (the standard CNC method;
                            safe here — WER is the right metric for filler).
  • NUMBER / ENTITY bins  → NEVER confidence-voted. If the systems agree, accept;
                            if they DISAGREE, FLAG for human review and log every
                            candidate. Auto-resolution is forbidden on these.

Why the asymmetry (the hard-won throughline of the whole pivot): every
combination method optimises an aggregate (WER); WER weights all words equally;
a wrong age and a wrong filler cost the same to WER and wildly different to a
survey record. ASR confidence is weakly correlated with correctness, so
confidence voting can confidently choose the WRONG REAL number. ROVER "cannot
fabricate from nothing" — but it CAN auto-select the wrong real token, which for
a survey field is nearly as bad. Hence: standard method for prose,
flag-don't-vote for the fields where a human must decide.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NUMBER = "number"
ENTITY = "entity"
PROSE = "prose"

# Bengali number detection (HIGH-RECALL by design: a missed number that leaks
# into the confidence vote is the catastrophic case, so we over-classify).
# Units/teens matched EXACTLY (substring would false-fire on এক→একটা). Tens,
# scales and fractions matched as SUBSTRINGS so irregular compounds are caught
# (সাঁইত্রিশ=37 contains ত্রিশ; আড়াইশো contains both আড়াই and শো).
_BN_NUM_EXACT = {
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়", "দশ",
    "এগারো", "এগার", "বারো", "বার", "তেরো", "তের", "চৌদ্দ", "পনেরো", "পনের",
    "ষোল", "ষোলো", "সতেরো", "সতের", "আঠারো", "আঠার", "উনিশ", "কুড়ি",
}
_BN_NUM_ROOTS = (
    "বিশ", "ত্রিশ", "চল্লিশ", "পঞ্চাশ", "ষাট", "সত্তর", "আশি", "নব্বই",   # tens
    "শো", "শত", "হাজার", "লাখ", "লক্ষ", "কোটি",                          # scales
    "সাড়ে", "দেড়", "আড়াই", "পৌনে",                                      # fractions
)
_PUNCT = re.compile(r"[\s।,.;:!?\-—–…\"'`()\[\]{}/]+")


def _norm(w: str) -> str:
    return _PUNCT.sub("", (w or ""))


def is_numberish(word: str) -> bool:
    """True if the token plausibly encodes a number (digit, exact word, or a
    distinctive tens/scale/fraction root as substring)."""
    if not word:
        return False
    if re.search(r"[0-9০-৯]", word):
        return True
    n = _norm(word)
    if n in _BN_NUM_EXACT:
        return True
    return any(root in n for root in _BN_NUM_ROOTS)


def token_type(candidates: list[str], entities: set[str] | None = None) -> str:
    """Classify a bin from ALL its candidate tokens. High-recall NUMBER first
    (over-flag is safe), then ENTITY via a survey gazetteer (NER for Bengali is
    unreliable — entities come from the instrument, not a heuristic), else
    PROSE."""
    if any(is_numberish(c) for c in candidates):
        return NUMBER
    if entities:
        ents = {_norm(e) for e in entities}
        if any(_norm(c) in ents for c in candidates):
            return ENTITY
    return PROSE


@dataclass
class BinDecision:
    token: str                       # chosen text (base token if flagged)
    token_type: str
    needs_review: bool
    reason: str
    candidates: list = field(default_factory=list)   # logged for the worklist


def _conf_vote(per_system, weights):
    """Standard confidence-weighted vote over each system's top candidate.
    weights: {system_name: w}. Only ever reached for PROSE bins."""
    best_tok, best_score = "", None
    for name, cands in per_system:
        if not cands:
            continue
        tok, conf = cands[0]
        score = (weights or {}).get(name, 1.0) * float(conf)
        if best_score is None or score > best_score:
            best_tok, best_score = tok, score
    return best_tok


def resolve_bin(per_system, base_system=None, weights=None, entities=None):
    """Resolve one CN bin. EXPLICIT BRANCHES — survey-critical CANNOT reach the
    confidence vote.

    per_system: list of (system_name, [(token, confidence), …n-best]) — the
                candidates each system contributed to this bin ([] = the system
                had no token here, i.e. a deletion).
    base_system: name of the prose base (e.g. "whisper"); its token is kept when
                 a survey-critical bin is flagged. Defaults to the first system.
    """
    tops = {name: (cands[0][0] if cands else "") for name, cands in per_system}
    names = [n for n, _ in per_system]
    base_name = base_system if base_system in tops else names[0]
    base = tops[base_name]
    all_toks = [t for _, cands in per_system for (t, _) in cands]
    ttype = token_type(all_toks, entities)

    if ttype in (NUMBER, ENTITY):
        toks = [tops[n] for n in names]            # incl. "" for a deletion
        nonempty = {_norm(t) for t in toks if t.strip()}
        # AGREE only if EVERY system produced the SAME token. A deletion by one
        # system (empty) vs a number by another is a DISAGREEMENT — "did they
        # say 13 or not?" is exactly a flag case, not silent agreement.
        if all(t.strip() for t in toks) and len(nonempty) == 1:
            return BinDecision(base or next(iter(nonempty)), ttype, False,
                               f"{ttype}-agree", all_toks)
        # disagreement OR one-sided deletion → FLAG; never auto-resolve/insert.
        return BinDecision(base, ttype, True, f"{ttype}-disagree-REVIEW",
                           all_toks)

    # PROSE → standard confidence-weighted CNC vote (safe; WER metric).
    return BinDecision(_conf_vote(per_system, weights), PROSE, False,
                       "prose-vote", all_toks)
