"""Tests for compliance and audit-logging behavior."""

import json
from pathlib import Path

import pytest

from src import compliance


def test_audit_writes_jsonl_entry(tmp_path, monkeypatch):
    """audit_llm_call should write one JSON line per call."""
    log_path = tmp_path / "audit.log"
    monkeypatch.setattr(compliance, "_AUDIT_LOG", log_path)
    with compliance.audit_llm_call(
        model="test-model", schema_name="TestSchema", prompt="hello world"
    ) as rec:
        rec["response"] = '{"ok": true}'

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["model"] == "test-model"
    assert entry["schema_name"] == "TestSchema"
    assert entry["success"] is True


def test_audit_records_error_on_exception(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.log"
    monkeypatch.setattr(compliance, "_AUDIT_LOG", log_path)
    with pytest.raises(RuntimeError):
        with compliance.audit_llm_call(
            model="m", schema_name="s", prompt="p"
        ):
            raise RuntimeError("boom")
    entry = json.loads(log_path.read_text().strip())
    assert entry["success"] is False
    assert "boom" in entry["error"]


def test_refuse_if_non_public_blocks_insider_query():
    out = compliance.refuse_if_non_public("Show me insider sentiment before the announcement")
    assert out is not None
    assert "non-public" in out or "insider" in out


def test_refuse_if_non_public_allows_normal_query():
    out = compliance.refuse_if_non_public("Summarize the Q3 2024 Apple earnings call")
    assert out is None


def test_log_ingestion_records_provenance(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.log"
    monkeypatch.setattr(compliance, "_AUDIT_LOG", log_path)
    compliance.log_ingestion(
        source_url="https://example.com/aapl-q3.html",
        ticker="AAPL", quarter="Q3", year=2024,
    )
    entry = json.loads(log_path.read_text().strip())
    assert entry["event"] == "transcript_ingested"
    assert entry["ticker"] == "AAPL"
