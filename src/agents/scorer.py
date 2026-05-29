"""
Credibility Scorer.

Combines the classical features (hedging, specificity) with the dodge-label
distribution into a single 0-100 score and a set of derived metrics.

Score philosophy: high = direct, specific, on-topic. Low = evasive, hedged,
off-topic. Calibration is documented in eval/calibration.md.
"""

from __future__ import annotations
from typing import List

from ..schemas import (
    CredibilityScore,
    DodgeCategory,
    DodgeLabel,
    HedgingScore,
    QAPair,
)


# Score weights for each dodge category (higher = more credible)
DODGE_WEIGHT = {
    DodgeCategory.DIRECT: 100.0,
    DodgeCategory.PARTIAL: 75.0,
    DodgeCategory.REFRAMED: 35.0,
    DodgeCategory.DEFERRED: 55.0,   # explicit refusal, but less severe than reframing
    DodgeCategory.NON_ANSWER: 15.0,
}


def score_call(
    pairs: List[QAPair],
    labels: List[DodgeLabel],
    features: List[HedgingScore],
) -> CredibilityScore:
    """Aggregate per-pair signals into a call-level credibility score."""
    if not pairs:
        return CredibilityScore(
            overall_score=50.0,
            dodge_rate=0.0,
            avg_hedging_density=0.0,
            contradiction_count=0,
        )

    n = len(pairs)
    # Dodge component: weighted avg of category scores, weighted by LLM confidence
    dodge_component = sum(
        DODGE_WEIGHT[lab.category] * lab.confidence for lab in labels
    ) / max(sum(lab.confidence for lab in labels), 1e-6)

    # Hedging component: lower hedging = more credible. Hedge density of 5/100
    # words is roughly the corpus median; map to 50.
    avg_hedge = sum(f.hedging_density for f in features) / n
    hedge_component = max(0.0, 100.0 - (avg_hedge * 8.0))

    # Specificity component: 0..1 → 0..100
    avg_spec = sum(f.specificity_score for f in features) / n
    spec_component = avg_spec * 100.0

    # Weighted blend: dodge category dominates (it's the most discriminative)
    overall = 0.55 * dodge_component + 0.20 * hedge_component + 0.25 * spec_component

    flagged = [
        pair.pair_id
        for pair, lab in zip(pairs, labels)
        if lab.category in {DodgeCategory.NON_ANSWER, DodgeCategory.REFRAMED}
    ]
    dodge_rate = sum(1 for lab in labels if lab.category != DodgeCategory.DIRECT) / n

    return CredibilityScore(
        overall_score=round(max(0.0, min(100.0, overall)), 1),
        dodge_rate=round(dodge_rate, 3),
        avg_hedging_density=round(avg_hedge, 2),
        contradiction_count=0,  # filled in by orchestrator
        flagged_questions=flagged,
    )
