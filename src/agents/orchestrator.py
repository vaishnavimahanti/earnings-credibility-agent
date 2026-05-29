"""
Orchestrator — wires the agents into a LangGraph workflow.

Flow:
    parse → features → dodge → retrieve_prior → contradictions → score → write

If the call has no Q&A section, the pipeline short-circuits with a brief
that only covers prepared remarks.
"""

from __future__ import annotations
from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from ..schemas import (
    AnalystBrief,
    Contradiction,
    CredibilityScore,
    DodgeLabel,
    EarningsCall,
    HedgingScore,
    QAPair,
)
from ..features import extractors
from . import dodge as dodge_agent
from . import contradiction as contradiction_agent
from . import retriever
from . import scorer
from . import writer


class _State(TypedDict, total=False):
    call: EarningsCall
    qa_pairs: List[QAPair]
    features: List[HedgingScore]
    dodge_labels: List[DodgeLabel]
    contradictions: List[Contradiction]
    credibility: CredibilityScore
    brief: AnalystBrief


def _node_features(state: _State) -> _State:
    call = state["call"]
    state["qa_pairs"] = call.qa_pairs
    state["features"] = extractors.batch_extract(call.qa_pairs, call)
    return state


def _node_dodge(state: _State) -> _State:
    pairs = state["qa_pairs"]
    feats = state["features"]
    labels = [dodge_agent.classify(p, f) for p, f in zip(pairs, feats)]
    state["dodge_labels"] = labels
    return state


def _node_index_self(state: _State) -> _State:
    """Index the current call so contradiction-detection has access to it next quarter."""
    retriever.index_call(state["call"])
    return state


def _node_contradictions(state: _State) -> _State:
    state["contradictions"] = contradiction_agent.detect_contradictions(
        state["call"], qa_pairs=state["qa_pairs"]
    )
    return state


def _node_score(state: _State) -> _State:
    state["credibility"] = scorer.score_call(
        state["qa_pairs"], state["dodge_labels"], state["features"]
    )
    return state


def _node_write(state: _State) -> _State:
    state["brief"] = writer.write_brief(
        call=state["call"],
        qa_pairs=state["qa_pairs"],
        dodge_labels=state["dodge_labels"],
        contradictions=state["contradictions"],
        credibility=state["credibility"],
    )
    return state


def build_graph():
    g = StateGraph(_State)
    g.add_node("features", _node_features)
    g.add_node("dodge", _node_dodge)
    g.add_node("index_self", _node_index_self)
    g.add_node("contradictions", _node_contradictions)
    g.add_node("score", _node_score)
    g.add_node("write", _node_write)

    g.set_entry_point("features")
    g.add_edge("features", "dodge")
    g.add_edge("dodge", "index_self")
    g.add_edge("index_self", "contradictions")
    g.add_edge("contradictions", "score")
    g.add_edge("score", "write")
    g.add_edge("write", END)
    return g.compile()


class EmptyCallError(ValueError):
    """Raised when a call parsed to zero Q&A pairs — we refuse to fabricate a brief."""


def analyze_call(call: EarningsCall) -> AnalystBrief:
    """Single entrypoint: take a parsed call, return a finished brief.

    Refuses to run if the call has no Q&A pairs. A 0-pair call means parsing
    failed upstream; scoring it would yield a meaningless 50/100 and the writer
    would hallucinate analysis around an empty input. Fail loudly instead.
    """
    if not call.qa_pairs:
        raise EmptyCallError(
            f"{call.ticker} {call.quarter} {call.year} parsed to 0 Q&A pairs "
            f"({len(call.turns)} turns total). The transcript parser could not "
            f"identify an analyst Q&A section, so there is nothing to score. "
            f"This is a parsing failure, not a credibility signal — inspect the "
            f"raw transcript and the speaker-role assignment before trusting any "
            f"output. Refusing to generate a brief from an empty call."
        )
    graph = build_graph()
    final_state = graph.invoke({"call": call})
    return final_state["brief"]
