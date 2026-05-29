"""
Dodge Classifier agent.

Given a single analyst Q&A pair, classify the management response into one of:
  - DIRECT      — addressed the specific question with concrete details
  - PARTIAL     — addressed part of it, hedged or ignored the rest
  - REFRAMED    — answered a related but different question
  - DEFERRED    — promised to follow up offline / referred to IR
  - NON_ANSWER  — pivoted entirely or gave only boilerplate

This is the feature no commercial earnings tool (AlphaSense, Hebbia, Rogo) leads with.
Every label comes with quoted evidence so an analyst can verify in one click.
"""

from __future__ import annotations
import re
from ..schemas import DodgeLabel, DodgeCategory, QAPair, HedgingScore
from ..llm import default_client


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "do", "does", "for", "from", "how", "i", "if", "in", "is", "it", "just",
    "maybe", "of", "on", "or", "so", "that", "the", "then", "there", "this",
    "to", "was", "we", "what", "when", "where", "whether", "with", "would",
    "you", "your",
}
_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z'\-]*\b")
_NUMBER_OR_DATE_RE = re.compile(
    r"\b(?:\d[\d,.]*|q[1-4]|20\d{2}|'\d{2}|first|second|third|fourth)\b",
    re.I,
)
_BUSINESS_SPECIFIC_RE = re.compile(
    r"\b("
    r"basis points|bps|percent|margin|revenue|sales|growth|grew|declined|"
    r"increased|decreased|flat|sequential|year over year|price|pricing|"
    r"volume|demand|orders|backlog|share|capital|expense|deposit|loan|"
    r"guidance|expect|expected|outlook|pipeline|launch|contract|inventory"
    r")\b",
    re.I,
)
_WHOLE_REFUSAL_RE = re.compile(
    r"\b("
    r"not going to (?:comment|speculate|provide|give)|don't comment|"
    r"do not comment|don't provide|do not provide|not providing|don't disclose|"
    r"do not disclose|don't break out|do not break out|we'll get back|"
    r"follow up|take that offline|connect offline|talk to ir|investor relations|"
    r"i don't have that|too early to (?:say|tell|call|predict)|"
    r"difficult to (?:say|predict|answer)|hard to (?:say|predict|answer)|"
    r"wouldn't want to speculate|can't speculate|cannot speculate|"
    r"ongoing litigation|ongoing case"
    r")\b",
    re.I,
)
_PROCEDURAL_ONLY_RE = re.compile(
    r"^\s*(?:thanks?|thank you|okay|all right|yeah|great|hi|good morning|"
    r"operator|next question|we'll take|can we have|may we have|you got it)"
    r"[\s,!.]*(?:"
    r"(?:operator,?\s*)?(?:next question|we'll take|can we have|may we have)"
    r".*)?$",
    re.I,
)
_PIVOT_ONLY_RE = re.compile(
    r"\b(excited|long-term opportunity|our philosophy|customer experience|"
    r"innovation|best than being most|best versus|tune out the noise|"
    r"strategic priority|focused on being|mission|brand strength)\b",
    re.I,
)
_QUANT_OR_FORWARD_ASK_RE = re.compile(
    r"\b(how much|how many|what percent|percentage|quantify|range|specific|"
    r"break out|mix|growth rate|revenue growth|margin|guidance|forecast|"
    r"outlook|next year|going forward|cadence|trajectory|contribution|"
    r"impact|2024|2025|2026|2027)\b",
    re.I,
)
_NORMAL_DISCLOSURE_RE = re.compile(
    r"\b(don't provide|do not provide|not providing|don't disclose|"
    r"do not disclose|don't break out|do not break out|not guide|"
    r"don't guide|no guidance|not forecast|won't forecast|don't forecast|"
    r"not prepared to give|not ready to give|not going to give|"
    r"not going to provide)\b",
    re.I,
)
_TOO_EARLY_RE = re.compile(
    r"\b(too early|difficult to predict|hard to predict|hard to say|"
    r"difficult to say|we'll see|not prepared to)\b",
    re.I,
)
_EXPLICIT_PUNT_RE = re.compile(
    r"\b(punt|take that offline|connect offline|follow up|circle back|"
    r"get back to you|talk to ir|investor relations|wouldn't want to speculate|"
    r"won't speculate|can't speculate|cannot speculate|ongoing litigation|"
    r"ongoing case|save that for another day|wait until|be specific in|"
    r"share at the beginning of|when we have news)\b",
    re.I,
)
_STRONG_DEFER_RE = re.compile(
    r"\b("
    r"punt|take that offline|connect offline|wouldn't want to speculate|"
    r"won't speculate|can't speculate|cannot speculate|ongoing litigation|"
    r"ongoing case|save that for another day|wait until|be specific in|"
    r"share at the beginning of|when we have news|not going to answer|"
    r"not going to talk about|don't have that in front|do not have that in front|"
    r"i don't have that in front|i do not have that in front"
    r")\b",
    re.I,
)
_SOFT_FOLLOWUP_RE = re.compile(
    r"\b(follow up|circle back|get back to you|talk to ir|"
    r"investor relations|ir team|help you model)\b",
    re.I,
)
_REFRAMED_SPECIFIC_ASK_RE = re.compile(
    r"\b("
    r"why|versus|vs\.?|preferred|differentiate|differ|relative to|compared to|"
    r"guardrails|accountability|track|measure|metric|target|threshold|"
    r"consumer environment|what are you seeing|what changed|what might change|"
    r"how do you think about the opportunity"
    r")\b",
    re.I,
)
_GENERIC_FRAME_RE = re.compile(
    r"\b("
    r"first thing|bear in mind|different types|from a .* angle|"
    r"we are excited|we're excited|excited to show|mechanism-of-action|"
    r"mechanism of action|long-term success|strong system|"
    r"focused on|opportunity broadly|building out|there will be other programs|"
    r"our strategy|our philosophy"
    r")\b",
    re.I,
)
_PRIMARY_STRATEGIC_ASK_RE = re.compile(
    r"\b("
    r"how (?:are|you're|you are) (?:thinking|assessing)|how do you assess|"
    r"opportunity for|right target molecule|preferred molecule|"
    r"which molecule|molecule to move forward|strategy|strategic|"
    r"plan b|minimum price|what's your minimum|pros and cons"
    r")\b",
    re.I,
)
_SECONDARY_TIMING_ONLY_RE = re.compile(
    r"\b("
    r"abstract was accepted|will be presented|presented at|"
    r"opportunity to see the full|wait for the data|"
    r"when we have news|early days|too early in this process"
    r")\b",
    re.I,
)

DODGE_SYSTEM_PROMPT = """You are an experienced sell-side equity research analyst.

Your job: classify how an executive responded to an analyst's question on an earnings call.

The five categories:

1. DIRECT_ANSWER — The executive addressed the specific question asked, with
   concrete details (numbers, dates, names, plans). The analyst would walk away
   with their question genuinely answered.

2. PARTIAL_ANSWER — The executive engaged with the PRIMARY question but left
   the single most important part of it materially unresolved. Reserve this for
   cases where the core ask — not a secondary or add-on sub-question — went
   unanswered. If they answered what was primarily asked, it is NOT partial.

3. REFRAMED_QUESTION — The executive answered a related-but-different question.
   They restated the topic in their own terms and addressed THAT, not what was
   actually asked. This is the classic dodge.

4. DEFERRED — The executive explicitly said they'd follow up offline, refer to
   the IR team, or punt to a future quarter / Investor Day. Information was
   withheld, not given.

5. NON_ANSWER — The executive pivoted entirely to another topic, gave only
   boilerplate ("we're really excited about the long-term opportunity"), or
   acknowledged the question without engaging with it.

PRIMARY-QUESTION DOCTRINE (read this before classifying):
  Analysts routinely stack a main question with one or more forward-looking or
  secondary sub-questions. Classify against the PRIMARY question — the central
  thing the analyst most wanted to know — not against every clause they uttered.

Important calibration rules:
  - "Great question" + a real answer = still DIRECT_ANSWER.
  - If the executive answered the primary question with concrete information,
    it is DIRECT_ANSWER even if they declined a secondary or forward-looking
    add-on. Declining ONE sub-clause does not demote a real answer to PARTIAL.
  - If your reasoning says the primary question was answered but a secondary
    question was declined, the correct label is DIRECT_ANSWER, not PARTIAL.
    Example: current China demand metrics answered + stimulus speculation
    declined = DIRECT_ANSWER.
  - Declining to forecast or quantify something the company does not guide to
    (e.g., a specific future segment growth rate, product-level revenue
    direction, a metric they never disclose) is NORMAL DISCLOSURE DISCIPLINE,
    not a dodge. If the rest of the answer was substantive, prefer
    DIRECT_ANSWER. Only use DEFERRED/NON_ANSWER when the refusal IS the whole
    response to the primary ask.
  - Treat Apple-style no-forward-guide answers as DIRECT_ANSWER when they give
    a real framework, timeline, or current-state read. Worked examples:
    CapEx question answered with hybrid model + 10-K reference = DIRECT_ANSWER;
    language rollout answered with current/December/April timing = DIRECT_ANSWER;
    iPhone mix answered with supply-constraint context and too-early caveat =
    DIRECT_ANSWER. Do not demote these to PARTIAL solely because no forecast
    or percentage was given.
  - Providing real, specific metrics (numbers, dates, named drivers) is strong
    evidence FOR direct_answer even if the analyst wanted still more detail.
    Analysts always want more detail; that craving alone is not a dodge.
  - A genuine "I don't know" with no offer to follow up is NON_ANSWER.
  - An answer that pivots to strategy/enthusiasm when the question was about a
    number, attribution, or a specific fact is REFRAMED_QUESTION.
  - Reserve PARTIAL_ANSWER for genuine cases where the core ask was left
    materially open — not for every answer that omits a tangent. When torn
    between DIRECT and PARTIAL, default to DIRECT if the primary question was
    answered with specifics.

Quote rules for evidence fields:
  - Verbatim text from the transcript only — no paraphrasing.
  - Each evidence field must be 25 words or fewer.
  - Pick the most diagnostic sentence — not the longest."""


def _build_user_prompt(pair: QAPair, features: HedgingScore | None) -> str:
    feature_summary = ""
    if features is not None:
        feature_summary = (
            f"\n\nPre-computed linguistic signals (use as priors, not as proof):\n"
            f"  - hedging_density: {features.hedging_density:.2f} (per 100 tokens)\n"
            f"  - specificity_score: {features.specificity_score:.2f} (0=vague, 1=concrete)\n"
            f"  - on_topic_score: {features.on_topic_score:.2f} (0=off-topic, 1=on-topic)\n"
            f"  - script_adherence: {features.script_adherence:.2f} "
            f"(higher = repeating prepared remarks)"
        )

    return f"""ANALYST QUESTION (from {pair.question_turn.speaker_name}):
{pair.question_text}

MANAGEMENT RESPONSE:
{pair.answer_text}
{feature_summary}

Classify this Q&A pair. Return the JSON object."""


def _content_overlap(question: str, answer: str) -> float:
    """Cheap topicality proxy used only for post-classification calibration."""
    q_terms = {
        t.lower()
        for t in _TOKEN_RE.findall(question)
        if len(t) > 3 and t.lower() not in _STOPWORDS
    }
    if not q_terms:
        return 0.0
    a_terms = {t.lower() for t in _TOKEN_RE.findall(answer)}
    return len(q_terms & a_terms) / len(q_terms)


def _has_substantive_specifics(answer: str, features: HedgingScore | None) -> bool:
    marker_count = len(_NUMBER_OR_DATE_RE.findall(answer)) + len(_BUSINESS_SPECIFIC_RE.findall(answer))
    feature_specific = bool(features and features.specificity_score >= 0.08)
    return marker_count >= 2 or feature_specific


def _is_whole_response_refusal(answer: str) -> bool:
    if not _WHOLE_REFUSAL_RE.search(answer):
        return False
    return len(answer.split()) < 90 or not _has_substantive_specifics(answer, None)


def _procedural_non_answer(pair: QAPair) -> DodgeLabel | None:
    answer = pair.answer_text.strip()
    lower = answer.lower()
    is_next_question_transition = (
        len(answer.split()) <= 18
        and "next question" in lower
        and any(cue in lower for cue in ("thank", "thanks", "operator"))
    )
    if len(answer.split()) <= 18 and (_PROCEDURAL_ONLY_RE.search(answer) or is_next_question_transition):
        return DodgeLabel(
            category=DodgeCategory.NON_ANSWER,
            confidence=1.0,
            evidence_from_question=pair.question_text[:120],
            evidence_from_answer=answer[:120],
            reasoning="Procedural transition or courtesy response; no substantive answer was provided.",
        )
    return None


def _short_quote(text: str) -> str:
    return " ".join(text.split()[:25])


def _rule_based_label(pair: QAPair, features: HedgingScore | None) -> DodgeLabel | None:
    """
    High-precision rules for cases the LLM tends to over-smooth as DIRECT.

    These rules intentionally cover only explicit refusal/procedure patterns.
    The LLM still handles ordinary semantic boundary cases.
    """
    question = pair.question_text
    answer = pair.answer_text
    answer_words = len(answer.split())
    substantive = _has_substantive_specifics(answer, features)
    forward_or_quant = bool(_QUANT_OR_FORWARD_ASK_RE.search(question))

    if _EXPLICIT_PUNT_RE.search(answer):
        strong_defer = bool(_STRONG_DEFER_RE.search(answer))
        soft_followup = bool(_SOFT_FOLLOWUP_RE.search(answer))
        if strong_defer and (answer_words < 180 or not substantive):
            return DodgeLabel(
                category=DodgeCategory.DEFERRED,
                confidence=0.95,
                evidence_from_question=_short_quote(question),
                evidence_from_answer=_short_quote(answer),
                reasoning="Explicit punt, offline follow-up, legal/speculation refusal, or IR referral withheld the requested information.",
            )
        if soft_followup and (answer_words < 60 or not substantive):
            return DodgeLabel(
                category=DodgeCategory.DEFERRED,
                confidence=0.9,
                evidence_from_question=_short_quote(question),
                evidence_from_answer=_short_quote(answer),
                reasoning="Management offered follow-up or IR referral instead of a substantive answer.",
            )

    if (
        _PRIMARY_STRATEGIC_ASK_RE.search(question)
        and _SECONDARY_TIMING_ONLY_RE.search(answer)
        and _content_overlap(question, answer) < 0.16
    ):
        return DodgeLabel(
            category=DodgeCategory.NON_ANSWER,
            confidence=0.88,
            evidence_from_question=_short_quote(question),
            evidence_from_answer=_short_quote(answer),
            reasoning=(
                "The primary question asked for management's strategic view, "
                "but the answer only addressed timing or process."
            ),
        )

    if forward_or_quant and _NORMAL_DISCLOSURE_RE.search(answer):
        if substantive:
            return DodgeLabel(
                category=DodgeCategory.PARTIAL,
                confidence=0.82,
                evidence_from_question=_short_quote(question),
                evidence_from_answer=_short_quote(answer),
                reasoning="Management gave some substantive context but explicitly declined the requested quantitative disclosure or guidance.",
            )
        return DodgeLabel(
            category=DodgeCategory.DEFERRED,
            confidence=0.9,
            evidence_from_question=_short_quote(question),
            evidence_from_answer=_short_quote(answer),
            reasoning="Management explicitly declined to provide the requested quantitative disclosure or guidance.",
        )

    if forward_or_quant and _TOO_EARLY_RE.search(answer):
        if answer_words >= 120 and substantive:
            return DodgeLabel(
                category=DodgeCategory.PARTIAL,
                confidence=0.78,
                evidence_from_question=_short_quote(question),
                evidence_from_answer=_short_quote(answer),
                reasoning="Management gave useful context but said the requested forward-looking read was too early or difficult to provide.",
            )
        return DodgeLabel(
            category=DodgeCategory.DEFERRED,
            confidence=0.86,
            evidence_from_question=_short_quote(question),
            evidence_from_answer=_short_quote(answer),
            reasoning="Management mainly deferred the forward-looking question as too early or difficult to answer.",
        )

    if (
        forward_or_quant
        and _PIVOT_ONLY_RE.search(answer)
        and not substantive
        and _content_overlap(question, answer) < 0.12
    ):
        return DodgeLabel(
            category=DodgeCategory.REFRAMED,
            confidence=0.82,
            evidence_from_question=_short_quote(question),
            evidence_from_answer=_short_quote(answer),
            reasoning="The answer pivots to broad strategy or enthusiasm instead of the requested quantitative or factual point.",
        )

    if (
        _REFRAMED_SPECIFIC_ASK_RE.search(question)
        and _GENERIC_FRAME_RE.search(answer)
        and _content_overlap(question, answer) < 0.18
    ):
        return DodgeLabel(
            category=DodgeCategory.REFRAMED,
            confidence=0.78,
            evidence_from_question=_short_quote(question),
            evidence_from_answer=_short_quote(answer),
            reasoning=(
                "The question asked for a specific comparison, guardrail, or "
                "business read, but the answer reframed into broad context."
            ),
        )

    return None


def _calibrate_label(
    label: DodgeLabel,
    pair: QAPair,
    features: HedgingScore | None,
) -> DodgeLabel:
    """
    Post-classification guardrail for the observed held-out failure mode:
    the LLM over-demotes concrete, on-topic answers to PARTIAL/REFRAMED just
    because an analyst asked for still more detail.
    """
    if label.category not in {DodgeCategory.PARTIAL, DodgeCategory.REFRAMED}:
        return label

    answer = pair.answer_text
    answer_words = len(answer.split())
    substantive = _has_substantive_specifics(answer, features)
    overlap = _content_overlap(pair.question_text, answer)
    on_topic = bool(features and features.on_topic_score >= 0.35)
    whole_refusal = _is_whole_response_refusal(answer)
    pivot_only = bool(_PIVOT_ONLY_RE.search(answer)) and not substantive
    specific_reframe_risk = bool(
        _REFRAMED_SPECIFIC_ASK_RE.search(pair.question_text)
        and _GENERIC_FRAME_RE.search(answer)
        and overlap < 0.18
    )

    if label.category == DodgeCategory.REFRAMED and specific_reframe_risk:
        return label

    if answer_words >= 80 and substantive and not whole_refusal and not pivot_only:
        if label.category == DodgeCategory.PARTIAL and (overlap >= 0.06 or on_topic):
            return label.model_copy(
                update={
                    "category": DodgeCategory.DIRECT,
                    "confidence": min(label.confidence, 0.78),
                    "reasoning": (
                        "Calibrated to DIRECT: the response gave substantive, "
                        "on-topic metrics or drivers; any missing detail appears secondary."
                    ),
                }
            )
        if label.category == DodgeCategory.REFRAMED and (overlap >= 0.10 or on_topic):
            return label.model_copy(
                update={
                    "category": DodgeCategory.DIRECT,
                    "confidence": min(label.confidence, 0.72),
                    "reasoning": (
                        "Calibrated to DIRECT: despite some framing language, the "
                        "answer materially engaged the question with concrete content."
                    ),
                }
            )

    return label


def classify(pair: QAPair, features: HedgingScore | None = None) -> DodgeLabel:
    """Classify a single Q&A pair into one of the five dodge categories."""
    procedural = _procedural_non_answer(pair)
    if procedural is not None:
        return procedural
    rule_label = _rule_based_label(pair, features)
    if rule_label is not None:
        return rule_label

    # Cheap pre-filter using HARD deflection cues only. Soft politeness cues
    # ("great question", "as we said", "i appreciate the question") are excluded
    # because they appear constantly in perfectly direct answers and were a
    # major source of over-flagging. We only nudge on explicit refusal/punt
    # language, and even then we frame it neutrally so the LLM still decides.
    hard_cues = [
        "we don't comment on", "we don't provide", "we don't disclose",
        "we don't break out", "we'll get back to you", "we'll follow up",
        "i'll have to get back", "i don't have that in front of me",
        "let's take that offline", "you can talk to ir",
        "talk to investor relations",
    ]
    low = pair.answer_text.lower().replace("’", "'")
    cue_count = sum(low.count(c) for c in hard_cues)

    prompt = _build_user_prompt(pair, features)
    if cue_count >= 2:
        prompt += (
            f"\n\nNote: this response contains explicit refusal/punt language "
            f"in {cue_count} places. Consider whether the refusal is the WHOLE "
            f"response to the primary question (then DEFERRED/NON_ANSWER) or "
            f"whether substantive information was still provided alongside it "
            f"(then DIRECT_ANSWER or PARTIAL_ANSWER). Decide from the content."
        )

    label = default_client().structured(
        prompt=prompt,
        schema=DodgeLabel,
        system=DODGE_SYSTEM_PROMPT,
        max_tokens=800,
    )
    return _calibrate_label(label, pair, features)

def is_dodge(label: DodgeLabel) -> bool:
    """A 'dodge' for reporting purposes is anything but DIRECT."""
    return label.category != DodgeCategory.DIRECT
