"""Centralized config — reads from .env and exposes typed settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
DUCKDB_PATH = Path(os.getenv("DUCKDB_PATH", DATA_DIR / "earnings.duckdb"))
EVAL_DIR = PROJECT_ROOT / "eval"

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "earnings-credibility-agent")

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_llm_provider() -> str:
    """Return 'anthropic' or 'openai' based on which key is set."""
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if OPENAI_API_KEY:
        return "openai"
    raise RuntimeError(
        "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
    )
