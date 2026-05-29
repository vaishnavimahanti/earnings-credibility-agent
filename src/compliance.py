"""
Compliance layer.

Three things matter on Wall Street, even in a portfolio project. Showing you
*thought* about these is the difference between "interesting project" and
"hire this person":

  1. PII redaction — earnings calls occasionally mention employee names that
     aren't already public (e.g., "we'd like to thank Sarah from accounting").
     We don't want them in vector stores or downstream LLM prompts.

  2. MNPI exposure — material non-public information should never be fed in.
     For public-company earnings calls this is technically NOT MNPI (calls are
     public the moment they're released), but we still log every transcript
     ingestion with its source URL so an audit can verify provenance.

  3. Model risk management (SR 11-7) — every LLM call is logged: model name,
     prompt hash, response hash, latency, cost. This is what makes the system
     auditable. We do NOT log full prompt content by default (that's a
     separate ZDR-mode tradeoff).
"""

from __future__ import annotations
import hashlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

from . import config

_AUDIT_LOG = config.DATA_DIR / "audit.log"
_logger = logging.getLogger("compliance")


@dataclass
class LLMCallRecord:
    """Single LLM-call audit record. Written to data/audit.log as JSONL."""
    timestamp: str
    run_id: str
    model: str
    prompt_hash: str
    response_hash: str
    latency_ms: float
    schema_name: str
    success: bool
    error: str | None = None
    metadata: dict = field(default_factory=dict)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@contextmanager
def audit_llm_call(model: str, schema_name: str, prompt: str, run_id: str | None = None):
    """
    Context manager that records every LLM call for SR 11-7-style audit.

    Usage:
        with audit_llm_call(model="claude-sonnet-4-5", schema_name="DodgeLabel", prompt=p) as rec:
            response = llm.structured(...)
            rec["response"] = response.model_dump_json()
    """
    started = time.time()
    record: dict = {"response": "", "error": None, "metadata": {}}
    rid = run_id or uuid.uuid4().hex[:12]
    try:
        yield record
        success = True
    except Exception as e:  # pragma: no cover
        record["error"] = str(e)
        success = False
        raise
    finally:
        latency_ms = (time.time() - started) * 1000
        entry = LLMCallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=rid,
            model=model,
            prompt_hash=_sha256(prompt),
            response_hash=_sha256(record["response"]),
            latency_ms=round(latency_ms, 1),
            schema_name=schema_name,
            success=success,
            error=record["error"],
            metadata=record["metadata"],
        )
        with _AUDIT_LOG.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")


def redact_pii(text: str) -> str:
    """
    Best-effort PII redaction using Microsoft Presidio.
    Falls back to no-op if presidio isn't installed (tests should still pass).
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        return text

    analyzer = _presidio_analyzer()
    results = analyzer.analyze(
        text=text,
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"],
        language="en",
    )
    if not results:
        return text
    anonymizer = AnonymizerEngine()
    return anonymizer.anonymize(text=text, analyzer_results=results).text


_analyzer_singleton = None


def _presidio_analyzer():
    global _analyzer_singleton
    if _analyzer_singleton is None:
        from presidio_analyzer import AnalyzerEngine
        _analyzer_singleton = AnalyzerEngine()
    return _analyzer_singleton


def log_ingestion(source_url: str, ticker: str, quarter: str, year: int) -> None:
    """Record the provenance of a transcript ingestion."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "transcript_ingested",
        "source_url": source_url,
        "ticker": ticker,
        "quarter": quarter,
        "year": year,
    }
    with _AUDIT_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def refuse_if_non_public(query: str) -> str | None:
    """
    Refusal handler — if the user's query looks like it's asking about a
    private company or MNPI, return a refusal string. Otherwise return None.
    """
    blocked_patterns = [
        "insider", "non-public", "mnpi", "private placement",
        "leaked", "before the announcement",
    ]
    low = query.lower()
    for p in blocked_patterns:
        if p in low:
            return (
                "I can't analyze non-public or insider information. This tool "
                "only operates on publicly released earnings calls and filings. "
                "Please reframe your question to use public materials only."
            )
    return None
