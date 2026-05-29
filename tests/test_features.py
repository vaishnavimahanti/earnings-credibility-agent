"""Tests for the linguistic feature extractors."""

import pytest

from src.features import extractors
from src.features.lm_dictionary import hedge_phrase_count, non_answer_cue_count


def test_hedging_density_higher_for_vague_text():
    """A vague answer should score higher hedging density than a concrete one."""
    vague = (
        "We believe revenue could be roughly in line with expectations. "
        "It's possible margins might improve, though it's uncertain. "
        "We tend to think the outlook is generally positive."
    )
    concrete = (
        "Revenue was $4.2 billion, up 11% year over year. "
        "Operating margin expanded 240 basis points to 31.5%. "
        "We're guiding to $4.5 billion next quarter."
    )
    assert extractors.hedging_density(vague) > extractors.hedging_density(concrete)


def test_hedging_density_empty():
    assert extractors.hedging_density("") == 0.0


def test_specificity_score_higher_for_numbers():
    numeric = "Revenue was $4.2 billion, up 11%, with margins of 31.5%."
    abstract = "We had a really great quarter and feel very positive about the future."
    assert extractors.specificity_score(numeric) > extractors.specificity_score(abstract)


def test_specificity_score_bounded():
    """Score should stay within [0, 1] even for very specific text."""
    extreme = " ".join(["$10B"] * 100)
    assert 0 <= extractors.specificity_score(extreme) <= 1.0


def test_hedge_phrase_count_finds_multi_word_phrases():
    text = "We believe we'll execute, and we expect strong growth. We tend to be conservative."
    assert hedge_phrase_count(text) >= 3


def test_non_answer_cue_count_finds_deflections():
    text = "We don't disclose that, but we'll get back to you. Great question, by the way."
    assert non_answer_cue_count(text) >= 2


def test_tokenize_lowercase_and_alpha_only():
    toks = extractors.tokenize("Revenue grew 11.5% in Q3, driven by AWS.")
    assert all(t.islower() for t in toks)
    assert "revenue" in toks
    assert "11" not in toks  # number stripped
