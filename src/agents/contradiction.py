"""
Contradiction Detector agent.

For each significant claim management makes this quarter, retrieve the
same-topic claim from prior quarters and ask an LLM whether they conflict.

Output: a list of `Contradiction` objects, each with severity and reasoning.
"""

from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field

from ..schemas import Contradiction, EarningsCall, QAPair
from ..llm import default_client
from . import retriever


class _ContradictionJudgment(BaseModel):
    """LLM output for a single comparison."""
    is_contradiction: bool
    severity: str = Field(description="'low', 'medium', or 'high'; ignored if not a contradiction")
    reasoning: str = Field(max_length=1200)


_SYSTEM_PROMPT = """You are a senior buy-side analyst comparing management
statements across quarters.

You will be given a CURRENT claim from this quarter's earnings call and a
RELATED claim from a prior quarter on what appears to be the same topic.

Decide whether the two claims CONTRADICT each other in a way that matters
to an investor. Be conservative — natural evolution of guidance, refinement
of estimates, or shift in time horizon is NOT a contradiction. A
contradiction is when the substance of the claim has materially flipped.

Severity:
  - high   — a hard reversal on a number, deadline, or commitment
  - medium — a softer reversal (e.g., "highest priority" → "one of our priorities")
  - low    — a minor shift in framing that an attentive analyst would still flag

If they don't contradict, set is_contradiction=False; severity is ignored.

Return the JSON object."""


def _judge_pair(current_claim: str, prior_claim: str, prior_quarter_label: str) -> _ContradictionJudgment:
    prompt = f"""CURRENT CLAIM (this quarter):
{current_claim}

PRIOR CLAIM (from {prior_quarter_label}):
{prior_claim}

Do these contradict? Return the JSON object."""
    return default_client().structured(
        prompt=prompt,
        schema=_ContradictionJudgment,
        system=_SYSTEM_PROMPT,
        max_tokens=400,
    )


def detect_contradictions(
    call: EarningsCall,
    qa_pairs: List[QAPair] | None = None,
    max_claims: int = 8,
) -> List[Contradiction]:
    """
    For each Q&A pair (up to max_claims), find prior-quarter context and
    judge whether the current answer contradicts any of it.
    """
    pairs = qa_pairs or call.qa_pairs
    pairs = pairs[:max_claims]
    out: List[Contradiction] = []

    for pair in pairs:
        # Use the question as the topic anchor — it's a clean handle on
        # what the analyst cared about.
        prior_chunks = retriever.retrieve_prior_context(
            ticker=call.ticker,
            query=pair.question_text,
            current_quarter=call.quarter,
            current_year=call.year,
            n_results=2,
        )
        if not prior_chunks:
            continue

        for prior in prior_chunks:
            judgment = _judge_pair(
                current_claim=pair.answer_text[:1500],
                prior_claim=prior["text"][:1500],
                prior_quarter_label=f"{prior['meta']['quarter']} {prior['meta']['year']}",
            )
            if judgment.is_contradiction:
                out.append(
                    Contradiction(
                        current_claim=pair.answer_text[:500],
                        prior_claim=prior["text"][:500],
                        prior_quarter=f"{prior['meta']['quarter']} {prior['meta']['year']}",
                        severity=judgment.severity if judgment.severity in {"low", "medium", "high"} else "low",
                        reasoning=judgment.reasoning,
                    )
                )
                break  # one contradiction per pair is enough

    return out
