"""Guards for the two-witness fusion prompt + output parsing.

These are cheap and catch regressions that already bit once:
  1. the YAML prompt and the in-code fallback drifting apart (two copies);
  2. the <think> reasoning block breaking [bn]/[en] extraction, or a
     skipped/unclosed block silently dropping a segment.

Speaker roles come from the fusion call itself: each turn is tagged
@ENUMERATOR / @RESPONDENT, and a question+answer segment is split.
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
    """A question+answer segment comes back as two @ROLE-tagged turns."""
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


# ── per-turn roles (no diarization) + GER on split turns ──────────────
from asr_pipeline.fusion import FusionTranslator  # noqa: E402
from asr_pipeline.models import AlignedSegment  # noqa: E402


def _seg(i, start, end, text):
    return AlignedSegment(segment_id=i, start_s=start, end_s=end,
                          speaker_id="SPEAKER_00", language="ben",
                          raw_text=text)


def _run_with_replies(replies, aligned, nbest):
    """Drive FusionTranslator.process with canned Gemma replies (no models)."""
    ft = FusionTranslator()            # default profile: languages/ben.yaml
    ft._run_whisper = lambda a, w: nbest

    def fake_fusion(a, titu, wn):
        for i, ans in enumerate(replies):
            ft._finish_segment(i, ans, {}, titu[i], wn[i])
        return [""] * len(a), [""] * len(a)
    ft._run_fusion = fake_fusion
    return ft.process(aligned, wav_path=None)


def test_roles_come_from_fusion_turns_and_split_question_answer():
    aligned = [_seg(0, 0.0, 10.0, "আপনার খানায় কতজন সদস্য আমার খানায় ছয়জন"),
               _seg(1, 10.0, 14.0, "হ্যাঁ")]
    nbest = [["আপনার খানায় কতজন সদস্য? আমার খানায় ছয়জন।"], ["হ্যাঁ।"]]
    replies = [
        "<think> enumerator asks, respondent answers </think>\n"
        "@ENUMERATOR\n[bn] আপনার খানায় কতজন সদস্য?\n"
        "[en] How many members in your household?\n"
        "@RESPONDENT\n[bn] আমার খানায় ছয়জন।\n[en] Six in my household.",
        "<think> short reply </think>\n[bn] হ্যাঁ।\n[en] Yes.",  # untagged
    ]
    out = _run_with_replies(replies, aligned, nbest)
    assert [s.speaker_id for s in out] == [
        "ENUMERATOR", "RESPONDENT", "RESPONDENT"]   # untagged → continues
    assert out[0].start_s == 0.0 and out[1].end_s == 10.0
    assert out[0].end_s == out[1].start_s          # contiguous split
    assert "split-turn" in out[0].review_reason


def test_ger_blanks_fabricated_number_per_turn_and_keeps_split():
    ft = FusionTranslator()            # default profile: languages/ben.yaml
    turns = [("ENUMERATOR", "বয়স কত?", "How old?"),
             ("RESPONDENT", "আমার বয়স ৬০", "I am 60")]
    titu, whyps = "বয়স কত আমার বয়স ৬", ["বয়স কত? আমার বয়স ৬।"]
    out = ft._apply_ger(0, turns, titu, whyps)
    assert [t[0] for t in out] == ["ENUMERATOR", "RESPONDENT"]
    assert out[0] == turns[0]
    assert "৬০" not in out[1][1] and "[অস্পষ্ট]" in out[1][1]
    assert "60" not in out[1][2] and "[unclear]" in out[1][2]
    assert any(r.startswith("number-abstain") for r in ft._ger_reasons[0])
