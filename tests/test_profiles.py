"""Language profiles: loading/validation, and the language-neutral pieces
they drive (reply tag, script guard, digits, CTC word timestamps)."""
import pytest
import yaml

from asr_pipeline import ger_guard
from asr_pipeline.config import AppConfig
from asr_pipeline.engines.ctc_engine import CTCEngine
from asr_pipeline.fusion import FusionTranslator, _parse_fusion
from asr_pipeline.profiles import (
    PROFILES_DIR,
    LanguageProfile,
    available_profiles,
    load_profile,
)

DEVANAGARI = ("DEVANAGARI", "LATIN")


# ── profile files ─────────────────────────────────────────────────────
def test_bengali_profile_is_pinned_and_complete():
    p = load_profile("ben")
    assert p.tag == "bn" and p.scripts == ["BENGALI", "LATIN"]
    assert p.witness_ctc.backend == "nemo"
    assert p.witness_ctc.model == "hishab/titu_stt_bn_fastconformer"
    assert p.witness_whisper.language == "bn"
    # reproducibility: full 40-char commit hashes, not branches
    for rev in (p.witness_ctc.revision, p.witness_whisper.revision):
        assert rev and len(rev) == 40
    assert p.prompt_path.exists()


def test_template_parses_and_is_not_listed():
    data = yaml.safe_load((PROFILES_DIR / "_template.yaml").read_text(
        encoding="utf-8"))
    LanguageProfile(**data)                       # schema-valid
    assert "_template" not in available_profiles()
    assert "ben" in available_profiles()


def test_missing_profile_error_lists_available():
    with pytest.raises(FileNotFoundError, match="Available: .*ben"):
        load_profile("zzz")


def test_fusion_routes_to_ctc_witness_for_any_language():
    cfg = AppConfig()
    cfg.postprocessing.translation_backend = "fusion"
    assert cfg.engine_for_language("ben") == "ctc"
    assert cfg.engine_for_language("nep") == "ctc"


# ── language-neutral parsing and guards ───────────────────────────────
def test_parse_fusion_uses_profile_tag():
    out = ("<think> respondent </think>\n@RESPONDENT\n"
           "[ne] मेरो परिवारमा छ जना।\n[en] Six in my family.")
    _, turns = _parse_fusion(out, tag="ne")
    assert turns == [("RESPONDENT", "मेरो परिवारमा छ जना।",
                      "Six in my family.")]
    _, wrong_tag = _parse_fusion(out, tag="bn")
    assert wrong_tag == [("RESPONDENT", "", "Six in my family.")]


def test_numbers_normalise_any_script_digits():
    assert ger_guard.numbers("उमेर ६० वर्ष") == {"60"}       # Devanagari
    assert ger_guard.numbers("বয়স ৬০") == {"60"}            # Bengali
    assert ger_guard.numbers("age 60") == {"60"}


def test_devanagari_profile_keeps_nepali_and_blanks_bengali():
    marker = "[अस्पष्ट]"
    ok = ger_guard.assess("मेरो घर नदी नजिक छ", "en", "मेरो घर नदी नजिक छ",
                          ["मेरो घर नदी नजिक छ"], scripts=DEVANAGARI,
                          unclear_marker=marker)
    assert ok.bn == "मेरो घर नदी नजिक छ"
    assert not any(r.startswith("foreign-script") for r in ok.reasons)
    bad = ger_guard.assess("मेरो আমার घर", "en", "मेरो घर", ["मेरो घर"],
                           scripts=DEVANAGARI, unclear_marker=marker)
    assert "আমার" not in bad.bn and marker in bad.bn


def test_fabricated_devanagari_number_abstains_with_profile_marker():
    marker = "[अस्पष्ट]"
    v = ger_guard.assess("उमेर ६५ वर्ष", "age 65", "उमेर ६० वर्ष",
                         ["उमेर ६० वर्ष"], scripts=DEVANAGARI,
                         unclear_marker=marker)
    assert "६५" not in v.bn and marker in v.bn
    assert "65" not in v.en and "[unclear]" in v.en


def test_review_flag_uses_profile_marker():
    review, reason = FusionTranslator._review_flag(
        "x", "y", "केही [अस्पष्ट]", "something [unclear]",
        unclear_marker="[अस्पष्ट]")
    assert review and "unclear" in reason


def test_bengali_prompt_must_exist_for_profile(tmp_path):
    p = load_profile("ben").model_copy(
        update={"code": "tst", "prompt_file": str(tmp_path / "none.yaml")})
    with pytest.raises(FileNotFoundError, match="Prompt file for 'tst'"):
        FusionTranslator(profile=p)


# ── CTC witness: HF frame offsets → absolute word timestamps ──────────
def test_hf_word_offsets_to_absolute_seconds():
    words = CTCEngine.hf_words(
        [{"word": "नमस्ते", "start_offset": 5, "end_offset": 30},
         {"word": "", "start_offset": 31, "end_offset": 32}],
        frame_s=0.02, offset_s=10.0)
    assert len(words) == 1
    assert words[0].word == "नमस्ते"
    assert words[0].start_s == pytest.approx(10.1)
    assert words[0].end_s == pytest.approx(10.6)


def test_nemo_filename_defaults_to_repo_name():
    e = CTCEngine(model_id="hishab/titu_stt_bn_fastconformer", language="ben")
    assert e.nemo_filename == "titu_stt_bn_fastconformer.nemo"
    e2 = CTCEngine(model_id="org/x", language="nep", nemo_file="y.nemo")
    assert e2.nemo_filename == "y.nemo"
