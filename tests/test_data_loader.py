"""Tests for transcript parsing logic — doesn't hit the network."""

from datetime import date

from src.schemas import CallSection, SpeakerRole, Turn
from src.utils.data_loader import classify_speaker, pair_qa, split_sections


def test_classify_speaker_operator():
    assert classify_speaker("Operator", None) == SpeakerRole.OPERATOR


def test_classify_speaker_executive():
    assert classify_speaker("Tim Cook", "Chief Executive Officer") == SpeakerRole.EXECUTIVE
    assert classify_speaker("Luca Maestri", "CFO") == SpeakerRole.EXECUTIVE


def test_classify_speaker_analyst():
    assert classify_speaker("Toni Sacconaghi", "Bernstein") == SpeakerRole.ANALYST
    assert classify_speaker("Brian Nowak", "Morgan Stanley") == SpeakerRole.ANALYST


def test_classify_speaker_unknown():
    assert classify_speaker("Jane Doe", None) == SpeakerRole.UNKNOWN


def _make_turn(name, role, text, pos):
    return Turn(
        turn_id=f"t-{pos}",
        speaker_name=name,
        speaker_role=role,
        section=CallSection.OTHER,
        text=text,
        word_count=len(text.split()),
        position=pos,
    )


def test_split_sections_uses_operator_qa_cue():
    """When the operator says 'first question', everything after is Q&A."""
    turns = [
        _make_turn("CEO", SpeakerRole.EXECUTIVE, "Welcome to our call.", 0),
        _make_turn("CFO", SpeakerRole.EXECUTIVE, "Our revenue was strong.", 1),
        _make_turn("Operator", SpeakerRole.OPERATOR, "We will now take your first question.", 2),
        _make_turn("Analyst", SpeakerRole.ANALYST, "Thanks. My question is...", 3),
        _make_turn("CEO", SpeakerRole.EXECUTIVE, "Great question. The answer is...", 4),
    ]
    result = split_sections(turns)
    assert result[0].section == CallSection.PREPARED
    assert result[1].section == CallSection.PREPARED
    assert result[3].section == CallSection.QA
    assert result[4].section == CallSection.QA


def test_context_classifier_handles_missing_titles():
    """The HF dataset has no titles — roles must come from call structure.

    Regression test for the empty-call bug: prepared-remarks speakers become
    executives, new Q&A voices become analysts, and Q&A pairs are built.
    """
    from src.utils.data_loader import classify_speakers_by_context, pair_qa

    raw = [
        _make_turn("Operator", SpeakerRole.UNKNOWN,
                   "Welcome to the call. Joining us is Jane CEO.", 0),
        _make_turn("Jane Smith", SpeakerRole.UNKNOWN,
                   "Revenue was $4 billion, up 12 percent.", 1),
        _make_turn("Operator", SpeakerRole.UNKNOWN,
                   "We will now begin the question-and-answer session. "
                   "Our first question comes from Bob Analyst.", 2),
        _make_turn("Bob Jones", SpeakerRole.UNKNOWN,
                   "What was the gross margin in the quarter?", 3),
        _make_turn("Jane Smith", SpeakerRole.UNKNOWN,
                   "Gross margin was 42 percent, up 100 basis points.", 4),
    ]
    classified = classify_speakers_by_context(raw)
    by_name = {t.speaker_name: t.speaker_role for t in classified}
    assert by_name["Operator"] == SpeakerRole.OPERATOR
    assert by_name["Jane Smith"] == SpeakerRole.EXECUTIVE  # spoke in prepared remarks
    assert by_name["Bob Jones"] == SpeakerRole.ANALYST     # new voice in Q&A

    pairs = pair_qa(classified)
    assert len(pairs) == 1
    assert pairs[0].question_turn.speaker_name == "Bob Jones"
    assert pairs[0].answer_turns[0].speaker_name == "Jane Smith"


def test_record_to_call_parses_titleless_structured_content():
    """A realistic HF record ({speaker, text}, no titles) must yield Q&A pairs."""
    from src.utils.data_loader import _record_to_call

    rec = {
        "symbol": "TEST", "company_name": "Test Co", "quarter": 3, "year": 2024,
        "date": "2024-10-01 17:00:00", "content": "",
        "structured_content": [
            {"speaker": "Operator", "text": "Welcome. Joining us is Pat Exec."},
            {"speaker": "Pat Exec", "text": "Revenue grew 10 percent to $2 billion this quarter."},
            {"speaker": "Operator", "text": "Our first question comes from Sam Analyst."},
            {"speaker": "Sam Analyst", "text": "What drove the margin expansion specifically?"},
            {"speaker": "Pat Exec", "text": "Mix shift and operating leverage drove 80 basis points."},
        ],
    }
    call = _record_to_call(rec)
    assert len(call.qa_pairs) == 1, "titleless record must still produce Q&A pairs"
    assert call.ticker == "TEST"


def test_record_to_call_raw_content_fallback():
    """When structured_content is empty, parse the raw content string."""
    from src.utils.data_loader import _record_to_call

    rec = {
        "symbol": "RAW", "company_name": "Raw Co", "quarter": 1, "year": 2024,
        "date": "2024-01-15 17:00:00", "structured_content": [],
        "content": (
            "Operator: Welcome. Joining us is Dana Boss.\n"
            "Dana Boss: We earned $500 million in net income this quarter.\n"
            "Operator: First question from Lee Researcher.\n"
            "Lee Researcher: Can you break out the segment revenue?\n"
            "Dana Boss: Sure, the cloud segment was $300 million of the total."
        ),
    }
    call = _record_to_call(rec)
    assert len(call.qa_pairs) == 1, "raw-content fallback must produce Q&A pairs"
    """Q&A pairing should group exec answers under the most recent analyst question."""
    turns = [
        _make_turn("Analyst A", SpeakerRole.ANALYST, "What was revenue?", 0),
        _make_turn("CEO", SpeakerRole.EXECUTIVE, "Answer to Q1.", 1),
        _make_turn("CFO", SpeakerRole.EXECUTIVE, "Adding on...", 2),
        _make_turn("Analyst B", SpeakerRole.ANALYST, "What was margin?", 3),
        _make_turn("CEO", SpeakerRole.EXECUTIVE, "Answer to Q2.", 4),
    ]
    # Set all to QA section
    turns = [t.model_copy(update={"section": CallSection.QA}) for t in turns]
    pairs = pair_qa(turns)
    assert len(pairs) == 2
    assert pairs[0].question_text == "What was revenue?"
    assert len(pairs[0].answer_turns) == 2  # CEO + CFO
    assert pairs[1].question_text == "What was margin?"
    assert len(pairs[1].answer_turns) == 1


def test_pair_qa_skips_courtesy_closings():
    """Short analyst thank-yous should not become classifier inputs."""
    turns = [
        _make_turn("Analyst A", SpeakerRole.ANALYST, "What was revenue growth in China?", 0),
        _make_turn("CEO", SpeakerRole.EXECUTIVE, "China revenue grew 8 percent.", 1),
        _make_turn("Analyst A", SpeakerRole.ANALYST, "Great. Thank you, Tim.", 2),
        _make_turn("CEO", SpeakerRole.EXECUTIVE, "Thanks. Can we have the next question, please?", 3),
        _make_turn("Analyst B", SpeakerRole.ANALYST, "Could you talk about component pricing for December?", 4),
        _make_turn("CFO", SpeakerRole.EXECUTIVE, "Memory prices are expected to increase in December.", 5),
        _make_turn("Analyst B", SpeakerRole.ANALYST, "Okay. Thank you. I.ll leave it there. Thank you.", 6),
        _make_turn("CFO", SpeakerRole.EXECUTIVE, "Thank you. Next question, please.", 7),
    ]
    turns = [t.model_copy(update={"section": CallSection.QA}) for t in turns]

    pairs = pair_qa(turns)

    assert len(pairs) == 2
    assert [p.question_text for p in pairs] == [
        "What was revenue growth in China?",
        "Could you talk about component pricing for December?",
    ]


def test_substantive_question_filter_keeps_real_question_fragments():
    from src.utils.data_loader import _is_substantive_question

    assert not _is_substantive_question("Thanks, Tim.")
    assert not _is_substantive_question("Okay. Thank you. I.ll leave it there. Thank you.")
    assert _is_substantive_question("curious whether you have a better read on iPhone demand")
    assert _is_substantive_question("Great, could you talk about CapEx outlook?")
