"""
Build a bundled sample earnings call for offline development and testing.

The sample is synthetic but realistic — modeled on the patterns documented
in eval/labeled_set.jsonl. This lets you run the full pipeline without any
data downloads.

Usage:
    python scripts/build_sample.py
"""

from __future__ import annotations
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import (
    CallSection,
    EarningsCall,
    QAPair,
    SpeakerRole,
    Turn,
)


def _turn(turn_id, name, role, title, section, text, pos):
    return Turn(
        turn_id=turn_id,
        speaker_name=name,
        speaker_role=role,
        speaker_title=title,
        section=section,
        text=text,
        word_count=len(text.split()),
        position=pos,
    )


def build_sample() -> EarningsCall:
    turns = [
        _turn("t-0", "Operator", SpeakerRole.OPERATOR, None, CallSection.PREPARED,
              "Welcome to Acme Corp's Q3 2024 earnings call. Joining us today are CEO Jane Mitchell and CFO Robert Lee.", 0),
        _turn("t-1", "Jane Mitchell", SpeakerRole.EXECUTIVE, "CEO", CallSection.PREPARED,
              "Thank you. We delivered another strong quarter. Revenue grew 14% year over year to $2.8 billion, "
              "operating margin expanded 180 basis points to 28.4%, and free cash flow was $720 million. "
              "Our Cloud segment continues to lead the growth story, with revenue up 32% to $980 million.", 1),
        _turn("t-2", "Robert Lee", SpeakerRole.EXECUTIVE, "CFO", CallSection.PREPARED,
              "For the fourth quarter, we expect revenue of $2.95 to $3.05 billion, representing 11 to 14% growth. "
              "We continue to expect full year operating margin to expand 150 to 200 basis points. "
              "Our previously announced $1 billion cost optimization program is on track to deliver the full target by year end.", 2),
        _turn("t-3", "Operator", SpeakerRole.OPERATOR, None, CallSection.PREPARED,
              "We will now begin the question-and-answer session. Our first question comes from Sarah Chen at Goldman Sachs.", 3),

        # Q1 — DIRECT
        _turn("t-4", "Sarah Chen", SpeakerRole.ANALYST, "Goldman Sachs", CallSection.QA,
              "Thanks for taking my question. Could you give us the Cloud segment operating margin in the quarter and how it trended sequentially?", 4),
        _turn("t-5", "Robert Lee", SpeakerRole.EXECUTIVE, "CFO", CallSection.QA,
              "Cloud operating margin was 21.4% in Q3, up from 19.8% in Q2 and 16.2% in the year-ago period. "
              "The 160 basis points of sequential improvement was driven by scale leverage and continued infrastructure efficiency.", 5),

        # Q2 — REFRAMED
        _turn("t-6", "Mike Torres", SpeakerRole.ANALYST, "JPMorgan", CallSection.QA,
              "What's the magnitude of the AI-related revenue contribution in Q3, and how should we think about that scaling into 2025?", 6),
        _turn("t-7", "Jane Mitchell", SpeakerRole.EXECUTIVE, "CEO", CallSection.QA,
              "We're incredibly excited about the AI opportunity in front of us. Customer demand has been extraordinary, "
              "and the team is executing brilliantly. We have multi-year roadmaps with our top 100 customers that all "
              "include AI components, and the platform momentum has never been stronger.", 7),

        # Q3 — DEFERRED
        _turn("t-8", "Priya Patel", SpeakerRole.ANALYST, "Morgan Stanley", CallSection.QA,
              "Can you quantify the percentage of Cloud revenue coming from your top 10 customers, and how that concentration has changed?", 8),
        _turn("t-9", "Robert Lee", SpeakerRole.EXECUTIVE, "CFO", CallSection.QA,
              "We don't disclose customer concentration at that level of granularity. What I can say is that our customer "
              "base is well diversified across industries and geographies. We'll get back to you with any additional color "
              "we can share offline.", 9),

        # Q4 — PARTIAL
        _turn("t-10", "Aisha Williams", SpeakerRole.ANALYST, "Bernstein", CallSection.QA,
              "On the cost optimization program, how much of the $1 billion has hit the P&L through Q3, and what's the cadence for Q4?", 10),
        _turn("t-11", "Robert Lee", SpeakerRole.EXECUTIVE, "CFO", CallSection.QA,
              "We've delivered approximately $700 million of the $1 billion through the first three quarters. "
              "The remainder will flow through Q4. We're not breaking down the specific bucket-level cadence at this time, "
              "but we feel very confident in the full year target.", 11),

        # Q5 — NON_ANSWER
        _turn("t-12", "Tom Bradshaw", SpeakerRole.ANALYST, "UBS", CallSection.QA,
              "Are you considering any changes to the dividend policy given the strong free cash flow?", 12),
        _turn("t-13", "Jane Mitchell", SpeakerRole.EXECUTIVE, "CEO", CallSection.QA,
              "Capital allocation is something the board reviews regularly. We're focused on driving long-term shareholder value, "
              "and we've consistently invested in the highest-return opportunities across organic growth, M&A, and shareholder returns. "
              "Our framework hasn't changed.", 13),

        # Q6 — DIRECT
        _turn("t-14", "Sarah Chen", SpeakerRole.ANALYST, "Goldman Sachs", CallSection.QA,
              "One follow-up — what was the FX impact on revenue in the quarter?", 14),
        _turn("t-15", "Robert Lee", SpeakerRole.EXECUTIVE, "CFO", CallSection.QA,
              "FX was a 130 basis point headwind to reported revenue growth in Q3. On a constant currency basis, "
              "revenue grew 15.3% versus the 14% reported figure.", 15),
    ]

    pairs = [
        QAPair(pair_id="p1", question_turn=turns[4], answer_turns=[turns[5]]),
        QAPair(pair_id="p2", question_turn=turns[6], answer_turns=[turns[7]]),
        QAPair(pair_id="p3", question_turn=turns[8], answer_turns=[turns[9]]),
        QAPair(pair_id="p4", question_turn=turns[10], answer_turns=[turns[11]]),
        QAPair(pair_id="p5", question_turn=turns[12], answer_turns=[turns[13]]),
        QAPair(pair_id="p6", question_turn=turns[14], answer_turns=[turns[15]]),
    ]

    return EarningsCall(
        ticker="ACME",
        company_name="Acme Corp",
        quarter="Q3",
        year=2024,
        call_date=date(2024, 10, 24),
        turns=turns,
        qa_pairs=pairs,
    )


def main():
    out_dir = Path(__file__).parent.parent / "data" / "sample_transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_call.json"

    call = build_sample()
    out_path.write_text(call.model_dump_json(indent=2))
    print(f"Sample call written to {out_path}")
    print(f"  Ticker: {call.ticker} {call.quarter} {call.year}")
    print(f"  Turns: {len(call.turns)}")
    print(f"  Q&A pairs: {len(call.qa_pairs)}")


if __name__ == "__main__":
    main()
