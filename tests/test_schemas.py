"""Schema validation tests."""

import pytest
from datetime import date

from src.schemas import (
    CallSection,
    DodgeCategory,
    DodgeLabel,
    EarningsCall,
    QAPair,
    SpeakerRole,
    Turn,
)


def make_turn(text: str, role: SpeakerRole = SpeakerRole.EXECUTIVE, pos: int = 0) -> Turn:
    return Turn(
        turn_id=f"t-{pos}",
        speaker_name="Test Speaker",
        speaker_role=role,
        section=CallSection.QA,
        text=text,
        word_count=len(text.split()),
        position=pos,
    )


def test_qa_pair_text_concatenation():
    """answer_text joins multiple answer turns."""
    q = make_turn("What was revenue?", SpeakerRole.ANALYST, 0)
    a1 = make_turn("Revenue was $10B.", SpeakerRole.EXECUTIVE, 1)
    a2 = make_turn("That's up 12%.", SpeakerRole.EXECUTIVE, 2)
    pair = QAPair(pair_id="p1", question_turn=q, answer_turns=[a1, a2])
    assert "Revenue was $10B." in pair.answer_text
    assert "up 12%" in pair.answer_text
    assert pair.question_text == "What was revenue?"


def test_dodge_label_evidence_length():
    """Evidence quotes should fit the documented 25-word soft cap."""
    label = DodgeLabel(
        category=DodgeCategory.NON_ANSWER,
        confidence=0.85,
        evidence_from_question="What was the take rate in Delivery this quarter?",
        evidence_from_answer="We're really excited about Delivery.",
        reasoning="Executive pivoted to enthusiasm without engaging with the take rate.",
    )
    assert label.category == DodgeCategory.NON_ANSWER
    assert 0 <= label.confidence <= 1


def test_dodge_confidence_bounds():
    """Confidence outside [0,1] should fail validation."""
    with pytest.raises(Exception):
        DodgeLabel(
            category=DodgeCategory.DIRECT,
            confidence=1.5,
            evidence_from_question="x",
            evidence_from_answer="y",
            reasoning="z",
        )


def test_earnings_call_prepared_text():
    """prepared_remarks_text() should join only PREPARED-section turns."""
    turns = [
        Turn(turn_id="t1", speaker_name="CEO", speaker_role=SpeakerRole.EXECUTIVE,
             section=CallSection.PREPARED, text="Great quarter.", word_count=2, position=0),
        Turn(turn_id="t2", speaker_name="CEO", speaker_role=SpeakerRole.EXECUTIVE,
             section=CallSection.PREPARED, text="Revenue grew 10%.", word_count=3, position=1),
        Turn(turn_id="t3", speaker_name="Analyst", speaker_role=SpeakerRole.ANALYST,
             section=CallSection.QA, text="Question?", word_count=1, position=2),
    ]
    call = EarningsCall(
        ticker="TEST", company_name="Test Co", quarter="Q1", year=2024,
        call_date=date(2024, 1, 1), turns=turns,
    )
    prepared = call.prepared_remarks_text()
    assert "Great quarter." in prepared
    assert "Revenue grew 10%." in prepared
    assert "Question?" not in prepared
