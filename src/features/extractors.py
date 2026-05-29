"""
Feature extractors for management responses.

Implements the five signals from finance-linguistics literature:
  1. hedging_density       — Loughran-McDonald uncertainty word ratio
  2. modal_weak_density    — weak modals (could/might/may) per 100 tokens
  3. specificity_score     — ratio of numbers/dates/entities to tokens
  4. on_topic_score        — cosine(question_embedding, answer_embedding)
  5. script_adherence      — cosine(answer_embedding, prepared_remarks_embedding)

These are CHEAP signals — they run in milliseconds without an LLM call.
The LLM only gets called when classical signals are ambiguous (see agents/dodge.py).
"""

from __future__ import annotations
import re
from functools import lru_cache
from typing import Iterable

from .lm_dictionary import (
    UNCERTAINTY,
    MODAL_WEAK,
    hedge_phrase_count,
)
from ..schemas import HedgingScore, QAPair, EarningsCall


_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
_PCT_RE = re.compile(r"\b\d[\d,.]*\s*%")
_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z'\-]*\b")


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer — lowercase, alpha-only."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def hedging_density(text: str) -> float:
    """Loughran-McDonald uncertainty words + multi-word hedges per 100 tokens."""
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return 0.0
    uncertainty_hits = sum(1 for t in toks if t in UNCERTAINTY)
    phrase_hits = hedge_phrase_count(text)
    return ((uncertainty_hits + phrase_hits) / n) * 100.0


def modal_weak_density(text: str) -> float:
    """Weak modal verbs per 100 tokens."""
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return 0.0
    hits = sum(1 for t in toks if t in MODAL_WEAK)
    return (hits / n) * 100.0


def specificity_score(text: str) -> float:
    """
    Ratio of specific-information markers (numbers, percentages, proper nouns)
    to total tokens. Higher = more concrete, lower = more vague.
    Scaled to [0, 1] via a soft cap at 0.20.
    """
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return 0.0
    num_count = len(_NUMBER_RE.findall(text)) + len(_PCT_RE.findall(text))
    # Proper-noun proxy: capitalized words that aren't sentence-initial
    proper = sum(1 for w in re.findall(r"\b[A-Z][a-z]+\b", text)) // 2
    raw = (num_count + proper) / n
    return min(1.0, raw / 0.20)


@lru_cache(maxsize=1)
def _get_embedder():
    """Lazy-load the sentence-transformers model (small + fast)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _cosine(a, b) -> float:
    import numpy as np
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)


def on_topic_score(question: str, answer: str) -> float:
    """Semantic similarity between question and answer. Higher = more on-topic."""
    model = _get_embedder()
    embs = model.encode([question, answer])
    score = _cosine(embs[0], embs[1])
    # MiniLM cosine sims live ~[0.2, 0.85] for real Q&A; rescale to [0, 1]
    return max(0.0, min(1.0, (score - 0.2) / 0.65))


def script_adherence(answer: str, prepared_remarks: str) -> float:
    """
    Cosine similarity between answer and the most similar prepared-remarks chunk.
    High score = management is repeating the prepared script (Lee 2016 signal).
    """
    if not prepared_remarks.strip():
        return 0.0
    model = _get_embedder()
    # Chunk prepared remarks into ~200-word windows
    words = prepared_remarks.split()
    chunks = [" ".join(words[i : i + 200]) for i in range(0, len(words), 100)]
    chunks = [c for c in chunks if len(c.split()) > 20] or [prepared_remarks]

    ans_emb = model.encode([answer])[0]
    chunk_embs = model.encode(chunks)
    sims = [_cosine(ans_emb, c) for c in chunk_embs]
    best = max(sims)
    return max(0.0, min(1.0, (best - 0.2) / 0.65))


def extract_features(pair: QAPair, call: EarningsCall) -> HedgingScore:
    """Run all extractors against one Q&A pair."""
    answer = pair.answer_text
    question = pair.question_text
    prepared = call.prepared_remarks_text()

    return HedgingScore(
        hedging_density=hedging_density(answer),
        uncertainty_density=hedging_density(answer),  # alias for clarity
        specificity_score=specificity_score(answer),
        on_topic_score=on_topic_score(question, answer),
        script_adherence=script_adherence(answer, prepared),
    )


def batch_extract(pairs: Iterable[QAPair], call: EarningsCall) -> list[HedgingScore]:
    """Run feature extraction across many pairs (more efficient — batches embeddings)."""
    pairs = list(pairs)
    if not pairs:
        return []
    model = _get_embedder()
    prepared = call.prepared_remarks_text()

    questions = [p.question_text for p in pairs]
    answers = [p.answer_text for p in pairs]
    all_embs = model.encode(questions + answers)
    q_embs = all_embs[: len(pairs)]
    a_embs = all_embs[len(pairs) :]

    # Prepared remarks chunks
    if prepared.strip():
        words = prepared.split()
        chunks = [" ".join(words[i : i + 200]) for i in range(0, len(words), 100)]
        chunks = [c for c in chunks if len(c.split()) > 20] or [prepared]
        chunk_embs = model.encode(chunks)
    else:
        chunk_embs = None

    out = []
    for i, p in enumerate(pairs):
        on_topic = _cosine(q_embs[i], a_embs[i])
        on_topic = max(0.0, min(1.0, (on_topic - 0.2) / 0.65))

        if chunk_embs is not None:
            script = max(_cosine(a_embs[i], c) for c in chunk_embs)
            script = max(0.0, min(1.0, (script - 0.2) / 0.65))
        else:
            script = 0.0

        out.append(
            HedgingScore(
                hedging_density=hedging_density(p.answer_text),
                uncertainty_density=hedging_density(p.answer_text),
                specificity_score=specificity_score(p.answer_text),
                on_topic_score=on_topic,
                script_adherence=script,
            )
        )
    return out
