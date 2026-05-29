"""
Brief Writer agent.

Takes everything the other agents produced and writes the analyst memo.
Every claim is grounded with a citation that points back to a transcript
turn — no free-floating prose.
"""

from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field

from ..schemas import (
    AnalystBrief,
    BriefSection,
    Citation,
    Contradiction,
    CredibilityScore,
    DodgeCategory,
    DodgeLabel,
    EarningsCall,
    QAPair,
)
from ..llm import default_client


class _DraftSection(BaseModel):
    heading: str
    content: str = Field(max_length=2400)
    cited_turn_ids: List[str] = Field(default_factory=list)


class _DraftBrief(BaseModel):
    headline: str = Field(max_length=500)
    key_concerns: List[_DraftSection] = Field(default_factory=list, max_length=4)
    positive_signals: List[_DraftSection] = Field(default_factory=list, max_length=3)


_SYSTEM_PROMPT = """You are an experienced sell-side equity research analyst
writing a short internal memo for your PM after an earnings call.

Constraints:
  - Be specific and concrete. Cite numbers and direct quotes.
  - Every section's content must reference at least one transcript turn by ID.
  - Tone: factual, hedged where appropriate, not promotional.
  - Sentence case in headings (no Title Case, no ALL CAPS).
  - No emoji, no marketing language.
  - Concerns first, positives second. The PM cares more about risk.
  - Limit each section to ~150 words.

Return the JSON object matching the schema."""


def _build_writer_prompt(
    call: EarningsCall,
    credibility: CredibilityScore,
    dodge_labels: List[DodgeLabel],
    contradictions: List[Contradiction],
    qa_pairs: List[QAPair],
) -> str:
    # Surface the most diagnostic Q&A turns for the writer
    flagged_pairs = [
        (p, lab)
        for p, lab in zip(qa_pairs, dodge_labels)
        if lab.category in {DodgeCategory.NON_ANSWER, DodgeCategory.REFRAMED, DodgeCategory.PARTIAL}
    ][:5]

    flagged_summary = "\n\n".join(
        f"[{p.question_turn.turn_id}] {lab.category.value.upper()} "
        f"(conf={lab.confidence:.2f}):\n"
        f"  Q: {p.question_text[:300]}\n"
        f"  A: {p.answer_text[:400]}\n"
        f"  Why: {lab.reasoning}"
        for p, lab in flagged_pairs
    ) or "(no flagged Q&A pairs)"

    contradiction_summary = "\n\n".join(
        f"vs {c.prior_quarter} ({c.severity}): {c.reasoning}\n"
        f"  Now: {c.current_claim[:200]}\n"
        f"  Then: {c.prior_claim[:200]}"
        for c in contradictions
    ) or "(no contradictions detected)"

    hedge_level = (
        "low" if credibility.avg_hedging_density < 2.0
        else "moderate" if credibility.avg_hedging_density < 5.0
        else "high"
    )

    return f"""COMPANY: {call.company_name} ({call.ticker})
QUARTER: {call.quarter} {call.year}
CALL DATE: {call.call_date}

CREDIBILITY METRICS:
  - Overall score: {credibility.overall_score}/100
  - Dodge rate: {credibility.dodge_rate:.1%}
  - Avg hedging density: {credibility.avg_hedging_density} ({hedge_level})
  - Contradiction count: {len(contradictions)}

FLAGGED Q&A PAIRS (in order of severity):
{flagged_summary}

CONTRADICTIONS WITH PRIOR QUARTERS:
{contradiction_summary}

Write the analyst memo. Concerns first, positives second. Do not cite low hedging density as evidence of evasiveness; if hedging density is low, say it partly offsets the dodge-rate concern or omit it."""


def write_brief(
    call: EarningsCall,
    qa_pairs: List[QAPair],
    dodge_labels: List[DodgeLabel],
    contradictions: List[Contradiction],
    credibility: CredibilityScore,
) -> AnalystBrief:
    prompt = _build_writer_prompt(call, credibility, dodge_labels, contradictions, qa_pairs)
    draft = default_client().structured(
        prompt=prompt,
        schema=_DraftBrief,
        system=_SYSTEM_PROMPT,
        max_tokens=2500,
    )

    # Hydrate citations from cited_turn_ids
    turn_by_id = {t.turn_id: t for t in call.turns}

    def _section_with_cites(s: _DraftSection) -> BriefSection:
        cites: List[Citation] = []
        for tid in s.cited_turn_ids:
            t = turn_by_id.get(tid)
            if t:
                quote = t.text[:380]
                cites.append(Citation(turn_id=t.turn_id, speaker_name=t.speaker_name, quote=quote))
        return BriefSection(heading=s.heading, content=s.content, citations=cites)

    credibility_with_count = credibility.model_copy(
        update={"contradiction_count": len(contradictions)}
    )

    qa_dodges = [
        (p.pair_id, lab)
        for p, lab in zip(qa_pairs, dodge_labels)
        if lab.category != DodgeCategory.DIRECT
    ]
    qa_labels = [
        (p.pair_id, lab)
        for p, lab in zip(qa_pairs, dodge_labels)
    ]

    return AnalystBrief(
        ticker=call.ticker,
        quarter=f"{call.quarter} {call.year}",
        call_date=call.call_date,
        headline=draft.headline,
        credibility=credibility_with_count,
        key_concerns=[_section_with_cites(s) for s in draft.key_concerns],
        positive_signals=[_section_with_cites(s) for s in draft.positive_signals],
        contradictions=contradictions,
        qa_dodges=qa_dodges,
        qa_labels=qa_labels,
    )
