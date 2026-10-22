"""GER guard: grapheme distance, number extraction, closest-to-N-best snap,
number/script abstain and agreement flags. Pure Python — no model, no GPU."""
from asr_pipeline.ger_guard import assess, grapheme_clusters, ndist, numbers

TITU = "বাইশ বন্যায় তেরোটা ছাগল ছিল"
WHYPS = ["বাইশ বন্যায় তেরো ছাগল",        # 3 correlated Whisper readings,
         "বাইশ বন্যায় তেরোটা ছাগল",       # all slightly different from titu
         "বাইশ বন্যায় তেরো ছাগল ছিল"]
HALLUC_NEAR_TITU = "বাইশ বন্যায় তেরোটা ছাগল ছিল আমার অনেক"


def test_grapheme_clusters_attach_vowel_signs():
    assert len(grapheme_clusters("জা")) == 1
    assert len(grapheme_clusters("জায়গা")) < len("জায়গা")


def test_ndist():
    assert ndist("এটা এক নম্বর ওয়ার্ড", "এটা এক নম্বর ওয়ার্ড।") == 0.0
    assert ndist("পানি আসে", "পানি আসে") == 0.0
    assert ndist("পানি", "গরু") > 0.5


def test_numbers_normalise_bengali_digits():
    assert numbers("আমার ১৩টা ছাগল") == {"13"}
    assert numbers("ward 1 আর ২০২২") == {"1", "2022"}


def test_snap_targets_nearest_witness_not_medoid():
    v = assess(HALLUC_NEAR_TITU, "en", TITU, WHYPS, snap=True, snap_tau=0.10)
    assert v.snapped and v.bn == TITU
    assert v.en == "" and any("snapped" in r for r in v.reasons)


def test_snap_off_is_observe_mode():
    v = assess(HALLUC_NEAR_TITU, "en", TITU, WHYPS, snap_tau=0.10)
    assert not v.snapped and v.bn == HALLUC_NEAR_TITU
    assert any(r.startswith("would-snap") for r in v.reasons)


def test_good_reconciliation_is_not_snapped():
    recon = "বাইশ বন্যায় তেরোটা ছাগল ছিল।"
    v = assess(recon, "en", TITU, WHYPS, snap=True)
    assert not v.snapped and v.bn == recon


def test_fabricated_number_abstains_in_both_lines():
    v = assess("আমার ১৫টা ছাগল ছিল", "I had 15 goats",
               "আমার ১৩টা ছাগল ছিল", ["আমার ১৩ টা ছাগল"])
    assert "[অস্পষ্ট]" in v.bn and "১৫" not in v.bn
    assert "[unclear]" in v.en and "15" not in v.en


def test_word_to_digit_is_not_fabrication():
    v = assess("আমার বয়স ৩৫", "age 35", "আমার বয়স পঁয়ত্রিশ",
               ["বয়স পঁয়ত্রিশ"])
    assert "৩৫" in v.bn
    assert not any(r.startswith("number-abstain") for r in v.reasons)


def test_foreign_script_is_blanked():
    v = assess("আমাদের хозяйства আছে", "we have a farm",
               "আমাদের খামার আছে", ["আমাদের খামার আছে"])
    assert "хозяйства" not in v.bn and "[অস্পষ্ট]" in v.bn
    assert any(r.startswith("foreign-script") for r in v.reasons)


def test_divergent_witnesses_are_flagged():
    v = assess("পানি আসে নদী থেকে", "en", "পানি আসে নদী থেকে",
               ["গরু মাঠে ঘাস খায় বিকেলে"])
    assert any(r.startswith("witness-disagree") for r in v.reasons)


def test_empty_and_single_witness_edge_cases():
    assert assess("", "", TITU, WHYPS).bn == ""
    assert assess("ক", "", "", []).bn == "ক"
