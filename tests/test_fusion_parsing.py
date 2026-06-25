"""Guards for the two-witness fusion prompt + output parsing.

These are cheap and catch regressions that already bit once:
  1. the YAML prompt and the in-code fallback drifting apart (two copies);
  2. the <think> reasoning block breaking [bn]/[en] extraction, or a
     skipped/unclosed block silently dropping a segment.

(The per-segment role-attribution stage — SegmentRoles / _apply_role_turns /
role_prompt — was removed; roles now come from the one-shot cluster→role
classifier on speaker-pure segments, so those tests are gone.)
"""
import re
from string import Template

import pytest

from asr_pipeline.fusion import (
    _parse_bn_en,
    _parse_fusion,
    _split_sentences,
    _split_span,
    _strip_think_instruction,
    load_prompts,
)

PROMPTS_FILE = "config/prompts/bn_survey.yaml"


def _norm(s: str) -> str:
    """Collapse whitespace so line-wrapping differences don't count as
    drift — we care that the two prompt copies say the same thing."""
    return re.sub(r"\s+", " ", s).strip()


# ── drift guard: YAML prompt == in-code fallback ──────────────────────
def test_fusion_prompt_yaml_matches_fallback():
    yaml_p = load_prompts(PROMPTS_FILE)["fusion_prompt"]
    code_p = load_prompts(None)["fusion_prompt"]   # in-code default
    args = dict(literal="LIT", fluent="FLU")
    assert _norm(Template(yaml_p).safe_substitute(**args)) == \
        _norm(Template(code_p).safe_substitute(**args))


@pytest.mark.parametrize("which", [PROMPTS_FILE, None])
def test_fusion_prompt_has_no_unfilled_placeholders(which):
    p = load_prompts(which)["fusion_prompt"]
    filled = Template(p).safe_substitute(literal="X", fluent="Y")
    assert not re.findall(r"\$[a-zA-Z_]+", filled)


# ── output parsing: <think> handling + role-tagged turns ──────────────
def test_parse_single_untagged_turn():
    """No @ROLE marker → one turn, role None (caller assigns a default).
    Backward-compatible with the pre-split output format."""
    out = ("<think> enumerator asking household size; overrode থানা->খানা "
           "</think>\n[bn] খানায় পাঁচ জন সদস্য।\n[en] Five members in the "
           "household.")
    think, turns = _parse_fusion(out)
    assert "খানা" in think                 # reasoning captured
    assert len(turns) == 1
    role, bn, en = turns[0]
    assert role is None                    # untagged
    assert bn == "খানায় পাঁচ জন সদস্য।"     # think NOT leaked into bn
    assert en == "Five members in the household."


def test_parse_two_role_turns():
    """_parse_fusion still supports @ROLE-tagged turns (defensive), even
    though the current prompt emits a single untagged turn."""
    out = ("<think> enumerator asks how they classify it, respondent says "
           "flood </think>\n"
           "@ENUMERATOR\n[bn] এই ঘটনাটাকে কী বলেন?\n[en] What do you call "
           "this event?\n"
           "@RESPONDENT\n[bn] আমরা বলি বন্যা।\n[en] We call it a flood.")
    _, turns = _parse_fusion(out)
    assert len(turns) == 2
    assert turns[0] == ("ENUMERATOR", "এই ঘটনাটাকে কী বলেন?",
                        "What do you call this event?")
    assert turns[1] == ("RESPONDENT", "আমরা বলি বন্যা।",
                        "We call it a flood.")


def test_parse_without_think_block_still_works():
    """Small models may ignore the <think> instruction entirely."""
    think, turns = _parse_fusion("@RESPONDENT\n[bn] হ্যাঁ।\n[en] Yes.")
    assert think == ""
    assert turns == [("RESPONDENT", "হ্যাঁ।", "Yes.")]


def test_think_mentioning_bracket_tokens_not_misparsed():
    """A [bn]/[en] mentioned inside the reasoning must not be grabbed."""
    out = ("<think> I will put the digit in [bn] and translate in [en] "
           "</think>\n@RESPONDENT\n[bn] তিন।\n[en] Three.")
    _, turns = _parse_fusion(out)
    assert turns == [("RESPONDENT", "তিন।", "Three.")]


def test_unclosed_think_returns_no_turns():
    """A token-capped, never-closed <think> yields NO turns — the trigger
    for the never-emit-empty fallback in process(). If this ever returns a
    bogus turn, the coverage logic breaks."""
    _, turns = _parse_fusion("<think> reasoning that got cut off mid-sen")
    assert turns == []


def test_split_span_proportional_and_exact_endpoints():
    # one turn → whole span
    assert _split_span(10.0, 20.0, ["abc"]) == [(10.0, 20.0)]
    # two turns → proportional to bn length, exact outer endpoints
    spans = _split_span(0.0, 10.0, ["aa", "aaaaaaaa"])  # 2 : 8
    assert spans[0][0] == 0.0 and spans[-1][1] == 10.0
    assert spans[0][1] == spans[1][0]          # contiguous
    assert abs(spans[0][1] - 2.0) < 1e-6       # 20% of 10s


def test_parse_bn_en_back_compat():
    bn, en = _parse_bn_en("[bn] ক\n[en] k")
    assert (bn, en) == ("ক", "k")


def test_split_sentences_bengali_and_latin():
    assert _split_sentences("খানায় পাঁচ জন। আপনি কেমন আছেন?") == \
        ["খানায় পাঁচ জন।", "আপনি কেমন আছেন?"]
    assert _split_sentences("") == []
    assert _split_sentences("one only") == ["one only"]


# ── think-OFF ablation arm: derived from the same prompt ──────────────
@pytest.mark.parametrize("which", [PROMPTS_FILE, None])
def test_fusion_think_off_has_no_think_tag(which):
    """The derived think-off prompt must contain no <think> tag and still
    fill cleanly — this is the matched ablation arm; if the derivation
    breaks, the comparison silently stops being think-off."""
    on = load_prompts(which)["fusion_prompt"]
    off = _strip_think_instruction(on)
    assert "<think>" not in off and "</think>" not in off
    assert "[bn]" in off and "[en]" in off       # output spec survives
    assert "$literal" in off and "$fluent" in off  # placeholders survive
    filled = Template(off).safe_substitute(literal="X", fluent="Y")
    assert not re.findall(r"\$[a-zA-Z_]+", filled)


def test_fusion_think_off_keeps_domain_content():
    """Stripping <think> must not collateral-damage the priors/rules."""
    off = _strip_think_instruction(load_prompts(PROMPTS_FILE)["fusion_prompt"])
    for must in ("খানা = household", "আড়াইশো=250", "Conflict rule",
                 "LITERAL", "FLUENT"):
        assert must in off
