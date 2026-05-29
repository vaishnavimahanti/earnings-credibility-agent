"""
Schemas for all structured agent outputs.
Every LLM call returns a typed object — no raw strings cross module boundaries.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Literal
from datetime import date
from pydantic import BaseModel, Field, ConfigDict


class SpeakerRole(str, Enum):
    OPERATOR = "operator"
    EXECUTIVE = "executive"
    ANALYST = "analyst"
    UNKNOWN = "unknown"


class CallSection(str, Enum):
    PREPARED = "prepared_remarks"
    QA = "qa"
    OTHER = "other"


class Turn(BaseModel):
    """A single speaker turn in an earnings call transcript."""
    model_config = ConfigDict(frozen=True)

    turn_id: str
    speaker_name: str
    speaker_role: SpeakerRole
    speaker_title: Optional[str] = None
    section: CallSection
    text: str
    word_count: int = Field(ge=0)
    position: int = Field(ge=0, description="0-indexed order in the call")


class QAPair(BaseModel):
    """An analyst question paired with the management response that followed."""
    pair_id: str
    question_turn: Turn
    answer_turns: List[Turn] = Field(min_length=1)

    @property
    def question_text(self) -> str:
        return self.question_turn.text

    @property
    def answer_text(self) -> str:
        return " ".join(t.text for t in self.answer_turns)


class DodgeCategory(str, Enum):
    """The five-way classification of how management handled an analyst question."""
    DIRECT = "direct_answer"
    PARTIAL = "partial_answer"
    REFRAMED = "reframed_question"
    DEFERRED = "deferred"
    NON_ANSWER = "non_answer"


class DodgeLabel(BaseModel):
    """Structured output from the Dodge Classifier agent."""
    category: DodgeCategory
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_from_question: str = Field(
        description="A short verbatim quote (<=25 words) of what was asked"
    )
    evidence_from_answer: str = Field(
        description="A short verbatim quote (<=25 words) showing the response pattern"
    )
    reasoning: str = Field(
        description="One sentence explaining the classification",
        max_length=800,
    )


class HedgingScore(BaseModel):
    """Output of the linguistic feature extractors."""
    hedging_density: float = Field(ge=0.0, description="Hedge words per 100 tokens")
    uncertainty_density: float = Field(ge=0.0, description="L-M uncertainty words per 100 tokens")
    specificity_score: float = Field(ge=0.0, le=1.0, description="Numeric/named-entity density")
    on_topic_score: float = Field(ge=0.0, le=1.0, description="Cosine sim Q vs A")
    script_adherence: float = Field(ge=0.0, le=1.0, description="Cosine sim A vs prepared remarks")


class Contradiction(BaseModel):
    """A potential contradiction with a prior quarter's claim."""
    current_claim: str
    prior_claim: str
    prior_quarter: str = Field(description="e.g. 'Q2 2024'")
    severity: Literal["low", "medium", "high"]
    reasoning: str


class CredibilityScore(BaseModel):
    """Composite credibility output for a full call."""
    overall_score: float = Field(ge=0.0, le=100.0, description="0=very evasive, 100=very direct")
    dodge_rate: float = Field(ge=0.0, le=1.0, description="Fraction of Q&A that was not a direct answer")
    avg_hedging_density: float = Field(ge=0.0)
    contradiction_count: int = Field(ge=0)
    flagged_questions: List[str] = Field(
        default_factory=list,
        description="pair_ids of Q&A pairs that scored as non-answer or reframed"
    )


class Citation(BaseModel):
    """A pointer back to evidence in the transcript."""
    turn_id: str
    speaker_name: str
    quote: str = Field(max_length=400)


class BriefSection(BaseModel):
    """One section of the analyst brief."""
    heading: str
    content: str
    citations: List[Citation] = Field(default_factory=list)


class AnalystBrief(BaseModel):
    """The final deliverable — what an analyst would actually read."""
    ticker: str
    quarter: str
    call_date: date
    headline: str = Field(description="One-sentence summary of the call's credibility profile")
    credibility: CredibilityScore
    key_concerns: List[BriefSection] = Field(default_factory=list)
    positive_signals: List[BriefSection] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list)
    qa_dodges: List[tuple[str, DodgeLabel]] = Field(
        default_factory=list,
        description="(pair_id, label) for every non-direct response"
    )
    qa_labels: List[tuple[str, DodgeLabel]] = Field(
        default_factory=list,
        description="(pair_id, label) for every classified Q&A response"
    )


class EarningsCall(BaseModel):
    """The structured representation of one full earnings call."""
    ticker: str
    company_name: str
    quarter: str
    year: int
    call_date: date
    turns: List[Turn]
    qa_pairs: List[QAPair] = Field(default_factory=list)

    def prepared_remarks_text(self) -> str:
        return "\n\n".join(
            t.text for t in self.turns if t.section == CallSection.PREPARED
        )
