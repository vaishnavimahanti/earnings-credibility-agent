from src.agents import dodge
from src.schemas import CallSection, DodgeCategory, QAPair, SpeakerRole, Turn


def _pair(question: str, answer: str) -> QAPair:
    q = Turn(
        turn_id="q",
        speaker_name="Analyst",
        speaker_role=SpeakerRole.ANALYST,
        section=CallSection.QA,
        text=question,
        word_count=len(question.split()),
        position=0,
    )
    a = Turn(
        turn_id="a",
        speaker_name="Exec",
        speaker_role=SpeakerRole.EXECUTIVE,
        section=CallSection.QA,
        text=answer,
        word_count=len(answer.split()),
        position=1,
    )
    return QAPair(pair_id="qa", question_turn=q, answer_turns=[a])


def test_procedural_transition_is_non_answer_without_llm():
    pair = _pair(
        "Can you talk about demand trends?",
        "Thanks, operator. Next question, please.",
    )
    label = dodge.classify(pair)
    assert label.category == DodgeCategory.NON_ANSWER


def test_explicit_offline_followup_is_deferred_without_llm():
    pair = _pair(
        "Can you quantify the impact on fiscal 2025 margins?",
        "We will follow up with you offline on the specific margin bridge.",
    )
    label = dodge.classify(pair)
    assert label.category == DodgeCategory.DEFERRED


def test_substantive_no_disclosure_is_partial_without_llm():
    pair = _pair(
        "Can you break out the growth rate by product for 2025?",
        (
            "We do not break out product-level growth rates. What I can say is "
            "that total revenue grew 8% year over year, demand improved "
            "sequentially, backlog increased, and pricing was positive across "
            "the portfolio. We expect those drivers to remain supportive."
        ),
    )
    label = dodge.classify(pair)
    assert label.category == DodgeCategory.PARTIAL


def test_guardrails_question_with_generic_innovation_frame_is_reframed():
    pair = _pair(
        (
            "What guardrails or accountability are in place around innovation, "
            "such as percentage of sales from new products or number of launches?"
        ),
        (
            "The first thing is to bear in mind that there are different types "
            "of innovation at play. One is renovation of core brands, and all "
            "of those efforts are focused on driving the business."
        ),
    )
    label = dodge.classify(pair)
    assert label.category == DodgeCategory.REFRAMED


def test_why_versus_question_with_mechanism_context_is_reframed():
    pair = _pair(
        (
            "Why might amylin be preferred versus GIP as a maintenance regimen "
            "for obesity, and how would it differ on half-life?"
        ),
        (
            "On GIP, we were excited to show the benefits of isolated GIP "
            "agonism to answer mechanism-of-action questions. There is "
            "potential for that molecule in other indications or combinations."
        ),
    )
    label = dodge.classify(pair)
    assert label.category == DodgeCategory.REFRAMED


def test_primary_strategy_question_with_only_event_timing_is_non_answer():
    pair = _pair(
        (
            "How are you assessing the Phase II NASH data and is tirzepatide "
            "or retatrutide the right target molecule to move forward?"
        ),
        (
            "The abstract was accepted and will be presented at EASL in June. "
            "That will be the opportunity to see the full NASH package."
        ),
    )
    label = dodge.classify(pair)
    assert label.category == DodgeCategory.NON_ANSWER


def test_wait_until_future_guidance_is_deferred():
    pair = _pair(
        "Can you give us a range for 2024 cash flow and deliveries?",
        (
            "We are going to wait until January to describe any range for "
            "cash flow next year. We will be specific then."
        ),
    )
    label = dodge.classify(pair)
    assert label.category == DodgeCategory.DEFERRED
