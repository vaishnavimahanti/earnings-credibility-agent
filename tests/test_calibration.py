"""
Calibration regression tests for the dodge classifier rubric.

These encode the specific real-call cases that the FIRST version of the rubric
over-flagged (observed on the AAPL FQ4 2024 call). They are LLM-dependent, so
they are skipped automatically when no API key is configured.

Run with:
    pytest tests/test_calibration.py -v        # skipped without a key
    RUN_LLM_TESTS=1 pytest tests/test_calibration.py -v   # runs live
"""

import os
import pytest

from src.schemas import CallSection, DodgeCategory, QAPair, SpeakerRole, Turn

_RUN_LLM = os.getenv("RUN_LLM_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not _RUN_LLM, reason="LLM calibration tests run only when RUN_LLM_TESTS=1"
)


def _pair(question: str, answer: str) -> QAPair:
    q = Turn(turn_id="cal-q", speaker_name="Analyst", speaker_role=SpeakerRole.ANALYST,
             section=CallSection.QA, text=question, word_count=len(question.split()), position=0)
    a = Turn(turn_id="cal-a", speaker_name="Exec", speaker_role=SpeakerRole.EXECUTIVE,
             section=CallSection.QA, text=answer, word_count=len(answer.split()), position=1)
    return QAPair(pair_id="cal", question_turn=q, answer_turns=[a])


def _classify(question, answer):
    from src.agents import dodge as dodge_agent
    return dodge_agent.classify(_pair(question, answer))


def test_china_metrics_with_declined_subquestion_is_not_partial():
    q = ("What are you seeing from a demand perspective in China? And could the "
         "recent stimulus plan be a catalyst going forward?")
    a = ("China was relatively flat year over year, with sequential improvement. "
         "Our installed base is at an all-time high, we're the top two smartphones "
         "in urban China, and we saw a strong percentage of new-to-iPhone customers. "
         "On the stimulus, it's a clear focus of the team there, but I'm not an "
         "economist and don't want to ad lib on its impact.")
    label = _classify(q, a)
    assert label.category == DodgeCategory.DIRECT, (
        f"Expected DIRECT (primary demand question answered with metrics), "
        f"got {label.category.value}: {label.reasoning}"
    )


def test_no_forecast_on_undisclosed_metric_is_direct():
    q = "Can Apple Intelligence help the Services growth rate?"
    a = ("From an ecosystem point of view, more engagement drives more App Store "
         "activity, more transactions, and developer adoption of our APIs has been "
         "strong. We don't forecast a specific Services growth-rate impact, but the "
         "ecosystem flywheel is clearly reinforced.")
    label = _classify(q, a)
    assert label.category in {DodgeCategory.DIRECT, DodgeCategory.PARTIAL}
    assert label.category not in {DodgeCategory.REFRAMED, DodgeCategory.NON_ANSWER}


def test_explicit_punt_is_still_deferred():
    q = "How is Apple prepared to deal with potential tariffs post-election?"
    a = ("I wouldn't want to speculate about those sorts of things. And so, I'm "
         "going to punt on that one.")
    label = _classify(q, a)
    assert label.category in {DodgeCategory.DEFERRED, DodgeCategory.NON_ANSWER}


def test_enthusiasm_pivot_on_attribution_is_still_reframed():
    q = ("Guiding for mid- to low-single-digit growth doesn't sound like alarm "
         "bells. What are people missing analytically here?")
    a = ("I could not be more excited about Apple Intelligence. The features are "
         "incredible, customer emails have been wonderful, and the Mac lineup is "
         "the strongest ever. I tune out the noise.")
    label = _classify(q, a)
    assert label.category in {DodgeCategory.REFRAMED, DodgeCategory.NON_ANSWER}



def test_capex_framework_without_forward_range_is_not_severe_dodge():
    q = ("Could you talk about the CapEx outlook and whether investments in "
         "Private Cloud Compute could change the historical CapEx range of "
         "roughly $10 billion a year?")
    a = ("We use a hybrid model for data centers. Some capacity is owned and "
         "some is provided by third parties, so our CapEx numbers may not be "
         "fully comparable with others. You will see in our 10-K the amount of "
         "CapEx incurred during fiscal 2024, and in fiscal 2025 we will continue "
         "to make the investments necessary for the business.")
    label = _classify(q, a)
    assert label.category in {DodgeCategory.DIRECT, DodgeCategory.PARTIAL}, (
        f"Expected DIRECT/PARTIAL (framework plus historical reference is substantive), "
        f"got {label.category.value}: {label.reasoning}"
    )


def test_language_rollout_timeline_without_installed_base_percent_is_direct():
    q = ("How much of the global installed base of phones will have access to "
         "Apple Intelligence in their native language over the next year or two?")
    a = ("US English is available now. In December we will add localized English "
         "for the UK, Australia, Canada, New Zealand, and South Africa. We will "
         "add more languages in April and then more as we step through the year. "
         "We have not set the specifics yet for every language.")
    label = _classify(q, a)
    assert label.category == DodgeCategory.DIRECT, (
        f"Expected DIRECT (rollout timeline answers primary access question), "
        f"got {label.category.value}: {label.reasoning}"
    )


def test_iphone_mix_too_early_due_to_constraints_is_direct():
    q = ("Are you seeing any change in consumer behavior on the mix front within "
         "the iPhone series?")
    a = ("It's tough to answer because we've been constrained in October on Pro "
         "and Pro Max. The data we have so far is early, and it is too early in "
         "the curve to call the precise mix. We will have a better read as supply "
         "normalizes.")
    label = _classify(q, a)
    assert label.category == DodgeCategory.DIRECT, (
        f"Expected DIRECT (current-state read plus constraint explanation), "
        f"got {label.category.value}: {label.reasoning}"
    )
