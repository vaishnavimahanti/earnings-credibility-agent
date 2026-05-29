"""
Loughran-McDonald financial sentiment word lists (abridged, public domain).
Source: https://sraf.nd.edu/textual-analysis/resources/

We include the most commonly-cited subsets used in earnings-call NLP literature:
  - UNCERTAINTY: words signaling vagueness/ambiguity
  - MODAL_WEAK: weak modal verbs (could, might, may)
  - HEDGE_PHRASES: multi-word hedges

This is intentionally not the full 80k-word list — we want fast lookups for the
feature extractor and reproducibility without external downloads.
"""

UNCERTAINTY = {
    "approximate", "approximately", "assume", "assumed", "assumes", "assuming",
    "believe", "believed", "believes", "believing", "conceivable", "conceivably",
    "could", "depend", "depended", "dependent", "depending", "depends",
    "estimate", "estimated", "estimates", "estimating", "estimation",
    "fluctuate", "fluctuated", "fluctuates", "fluctuating", "fluctuation",
    "indefinite", "indefinitely", "indeterminable", "indeterminate",
    "likelihood", "likely", "may", "maybe", "might", "nearly", "occasionally",
    "perhaps", "possibility", "possible", "possibly", "predict", "predicted",
    "predicting", "prediction", "predictions", "predicts", "probable",
    "probably", "risk", "risked", "riskier", "riskiest", "risks", "risky",
    "roughly", "seems", "seldom", "seldomly", "should", "sometimes",
    "somewhat", "speculate", "speculated", "speculates", "speculating",
    "speculation", "speculations", "speculative", "sporadic", "sporadically",
    "suggest", "suggested", "suggesting", "suggests", "suspect", "suspected",
    "suspects", "tend", "tended", "tending", "tends", "tentative",
    "tentatively", "uncertain", "uncertainly", "uncertainties", "uncertainty",
    "unclear", "unclearly", "uncommon", "undecided", "undefined",
    "undeterminable", "undetermined", "unforecasted", "unforeseeable",
    "unforeseen", "unguaranteed", "unidentifiable", "unidentified",
    "unknowable", "unknown", "unobservable", "unpredictability",
    "unpredictable", "unpredictably", "unproved", "unproven", "unquantifiable",
    "unquantified", "unreconciled", "unsettled", "unspecific", "unspecified",
    "untested", "unusual", "unusually", "vagaries", "vague", "vaguely",
    "vagueness", "variability", "variable", "variables", "variably",
    "variant", "variation", "variations", "varied", "varies", "vary",
    "varying", "volatile", "volatilities", "volatility",
}

MODAL_WEAK = {"could", "may", "might", "would", "should", "can", "perhaps"}

HEDGE_PHRASES = [
    "we believe", "we think", "we expect", "we anticipate", "we hope",
    "should be", "could be", "might be", "may be", "tend to", "kind of",
    "sort of", "to some extent", "in some ways", "more or less",
    "i guess", "i suppose", "i would say", "i would think",
    "as far as i can tell", "to the best of our knowledge",
    "directionally", "broadly speaking", "generally speaking",
    "ballpark", "ish",
]

# Non-answer cues — phrases that signal a question is being deflected
NON_ANSWER_CUES = [
    "we don't comment on", "we don't provide", "we don't disclose",
    "we don't break out", "we'll get back to you", "we'll follow up",
    "i'll have to get back", "i don't have that in front of me",
    "let's take that offline", "you can talk to ir", "talk to investor relations",
    "as we said", "as i mentioned", "as we mentioned",
    "i appreciate the question", "great question",
]


def _normalize(text: str) -> str:
    """Lowercase and collapse common contractions (don't/do not) for matching."""
    t = text.lower()
    # Treat curly quotes as straight, strip apostrophes (handles do not / don't)
    t = t.replace("\u2019", "'").replace("'", "")
    return t


def hedge_phrase_count(text: str) -> int:
    """Count multi-word hedge phrase occurrences (case-insensitive)."""
    low = _normalize(text)
    return sum(low.count(_normalize(p)) for p in HEDGE_PHRASES)


def non_answer_cue_count(text: str) -> int:
    """Count phrases that flag deflection."""
    low = _normalize(text)
    return sum(low.count(_normalize(p)) for p in NON_ANSWER_CUES)
