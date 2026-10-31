"""Word-alignment helpers of scripts/witness_complementarity.py."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "witness_complementarity",
    Path(__file__).resolve().parents[1] / "scripts" /
    "witness_complementarity.py")
wc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wc)


def test_correct_positions_marks_only_matched_reference_words():
    ref = "a b c d".split()
    assert wc.correct_positions("a b c d".split(), ref) == {0, 1, 2, 3}
    assert wc.correct_positions("a x c d".split(), ref) == {0, 2, 3}
    assert wc.correct_positions("a c d".split(), ref) == {0, 2, 3}
    assert wc.correct_positions("a b y c d".split(), ref) == {0, 1, 2, 3}
    assert wc.correct_positions([], ref) == set()


def test_wer_counts_edits():
    assert wc.wer("a x c".split(), "a b c".split()) == 1
    assert wc.wer([], "a b".split()) == 2


@pytest.mark.parametrize("text,norm,expected", [
    ("मेरो घर, virtual 1800!", False, ["मेरो", "घर", "virtual", "1800"]),
    ("मेरो घर, virtual 1800!", True, ["मेरो", "घर"]),
    ("बयस ৬০ বছর", True, ["बयस", "বছর"]),
])
def test_tokens_normalisation(text, norm, expected):
    assert wc.toks(text, norm) == expected


def test_tokens_keep_vowel_signs_together():
    # regression: [^\w\s] split words at combining vowel signs
    assert wc.toks("गर्नुपरेको थिएन।") == ["गर्नुपरेको", "थिएन"]
    assert wc.toks("খানায় ছয়জন সদস্য।") == ["খানায়", "ছয়জন", "সদস্য"]
