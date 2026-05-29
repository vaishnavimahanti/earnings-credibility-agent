"""
Retriever agent.

Indexes prior-quarter transcripts in Chroma, so for any claim in the current
call we can pull the same-topic claim from prior quarters and feed it to the
Contradiction Detector.

Why Chroma: free, local, zero-config. Swap to Pinecone in production.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .. import config
from ..schemas import EarningsCall, Turn, CallSection


_COLLECTION_NAME = "earnings_history"


def _client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _embed_fn():
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _collection():
    return _client().get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_embed_fn(),
    )


def _chunk_turn(turn: Turn, max_words: int = 120) -> list[str]:
    """Split a long turn into smaller chunks for retrieval."""
    words = turn.text.split()
    if len(words) <= max_words:
        return [turn.text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def index_call(call: EarningsCall) -> int:
    """Add one earnings call to the retrieval index. Returns # chunks indexed."""
    coll = _collection()
    docs: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []
    for turn in call.turns:
        for ci, chunk in enumerate(_chunk_turn(turn)):
            docs.append(chunk)
            ids.append(f"{call.ticker}-{call.quarter}-{call.year}-{turn.turn_id}-{ci}")
            metadatas.append(
                {
                    "ticker": call.ticker,
                    "quarter": call.quarter,
                    "year": call.year,
                    "call_date": call.call_date.isoformat(),
                    "speaker": turn.speaker_name,
                    "speaker_role": turn.speaker_role.value,
                    "section": turn.section.value,
                }
            )
    if not docs:
        return 0
    coll.upsert(documents=docs, ids=ids, metadatas=metadatas)
    return len(docs)


def index_calls(calls: Iterable[EarningsCall]) -> int:
    """Batch-index many calls."""
    return sum(index_call(c) for c in calls)


def retrieve_prior_context(
    ticker: str,
    query: str,
    current_quarter: str,
    current_year: int,
    n_results: int = 4,
) -> list[dict]:
    """
    Return chunks from PRIOR quarters of the same ticker that are most
    relevant to `query`. Excludes the current quarter to prevent leakage.
    """
    coll = _collection()
    results = coll.query(
        query_texts=[query],
        n_results=n_results * 3,  # over-fetch then filter
        where={"ticker": ticker},
    )

    out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Exclude current quarter
        if meta["year"] == current_year and meta["quarter"] == current_quarter:
            continue
        # Exclude future quarters (leakage prevention)
        if (meta["year"], _q_to_int(meta["quarter"])) >= (
            current_year,
            _q_to_int(current_quarter),
        ):
            continue
        out.append({"text": doc, "meta": meta, "distance": dist})
        if len(out) >= n_results:
            break
    return out


def _q_to_int(q: str) -> int:
    """'Q3' -> 3."""
    return int(str(q).strip("Qq") or 0)


def reset_index() -> None:
    """Wipe the collection (useful for tests)."""
    try:
        _client().delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
