"""
Load earnings call transcripts from free public sources.

Primary source: `kurry/sp500_earnings_transcripts` on HuggingFace.
Fallback: a small bundled sample (data/sample_transcripts/) for offline dev.

Each call is parsed into our Pydantic `EarningsCall` schema:
    - speaker turns
    - sections (prepared remarks vs Q&A)
    - paired Q&A
"""

from __future__ import annotations
import json
import re
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Optional

from ..schemas import (
    CallSection,
    EarningsCall,
    QAPair,
    SpeakerRole,
    Turn,
)
from .. import config


# Heuristics for identifying speaker role from titles like
# "Tim Cook -- Chief Executive Officer" or "Operator" or "Toni Sacconaghi -- Bernstein"
_EXEC_TITLES = re.compile(
    r"(CEO|CFO|COO|CTO|chief|president|founder|chair|head of|"
    r"officer|vice president|VP)",
    re.IGNORECASE,
)
_ANALYST_TITLES = re.compile(
    r"(analyst|JPMorgan|Goldman|Morgan Stanley|Bernstein|Citi|"
    r"Bank of America|Wells Fargo|Barclays|Credit Suisse|UBS|"
    r"Wedbush|Cowen|Jefferies|Evercore|Raymond James|Stifel)",
    re.IGNORECASE,
)


def _is_operator(name: str) -> bool:
    return bool(name) and name.strip().lower() in {"operator", "host", "moderator"}


def classify_speaker(name: str, title: str | None) -> SpeakerRole:
    """
    Title-based fast path, kept for callers that DO have titles (the synthetic
    sample and the eval set). The HuggingFace dataset carries NO titles — for
    that data, role assignment happens in `classify_speakers_by_context()`,
    which uses call structure instead.
    """
    if _is_operator(name):
        return SpeakerRole.OPERATOR
    blob = f"{name} {title or ''}"
    if title and _EXEC_TITLES.search(blob):
        return SpeakerRole.EXECUTIVE
    if title and _ANALYST_TITLES.search(blob):
        return SpeakerRole.ANALYST
    return SpeakerRole.UNKNOWN


def classify_speakers_by_context(turns: list[Turn]) -> list[Turn]:
    """
    Assign speaker roles using CALL STRUCTURE rather than titles.

    Mirrors how a real earnings call is shaped:
      - "Operator"/"Host" turns are the operator.
      - Anyone who speaks during prepared remarks (before Q&A) and isn't the
        operator is an EXECUTIVE — management delivers the prepared remarks.
      - In Q&A, the operator introduces each analyst, who asks, then management
        answers. A Q&A speaker already known as an executive stays EXECUTIVE;
        any other non-operator Q&A speaker is an ANALYST.

    Robust to the missing-title problem in the HF dataset.
    """
    qa_start = _find_qa_start(turns)

    exec_names: set[str] = set()
    for t in turns[:qa_start]:
        if not _is_operator(t.speaker_name):
            exec_names.add(t.speaker_name.strip().lower())

    out: list[Turn] = []
    expecting_analyst = True
    for i, t in enumerate(turns):
        section = CallSection.QA if i >= qa_start else CallSection.PREPARED
        name_key = t.speaker_name.strip().lower()
        if _is_operator(t.speaker_name):
            role = SpeakerRole.OPERATOR
            if section == CallSection.QA:
                expecting_analyst = True
        elif section != CallSection.QA:
            role = SpeakerRole.EXECUTIVE
        elif name_key in exec_names:
            role = SpeakerRole.EXECUTIVE
            expecting_analyst = False
        elif expecting_analyst:
            role = SpeakerRole.ANALYST
            expecting_analyst = False
        else:
            # Some executives do not speak in prepared remarks, so they are not
            # in exec_names. Within an operator-delimited Q&A block, the first
            # unknown speaker is usually the analyst; subsequent unknown
            # speakers are management answering the question.
            role = SpeakerRole.EXECUTIVE
        out.append(t.model_copy(update={"section": section, "speaker_role": role}))
    return out


def split_sections(turns: list[Turn]) -> list[Turn]:
    """Assign sections WITHOUT changing roles (title-based callers)."""
    qa_start = _find_qa_start(turns)
    out = []
    for i, t in enumerate(turns):
        section = CallSection.QA if i >= qa_start else CallSection.PREPARED
        out.append(t.model_copy(update={"section": section}))
    return out


def _find_qa_start(turns: list[Turn]) -> int:
    """Return the index of the first Q&A turn, or len(turns) if no Q&A."""
    qa_cue = re.compile(
        r"(question[- ]and[- ]answer|first question|begin the q\s*&\s*a|"
        r"open .{0,20} questions|take .{0,20} questions|"
        r"question comes from|question is from)",
        re.IGNORECASE,
    )
    for i, t in enumerate(turns):
        if _is_operator(t.speaker_name) and qa_cue.search(t.text):
            return i + 1
    for i, t in enumerate(turns):
        if t.speaker_role == SpeakerRole.ANALYST:
            return i
    return len(turns)


_QUESTION_CUES = re.compile(
    r"\b(what|when|where|why|how|can|could|should|would|do|does|did|is|are|"
    r"was|were|whether|if|any|which|who|whose|curious|help us|talk about|"
    r"walk us through|give us|share with us|color on|view of|outlook|guidance)\b",
    re.IGNORECASE,
)
_COURTESY_CLOSING = re.compile(
    r"\b(thanks?|thank you|appreciate|best of luck|congrats|congratulations|"
    r"i.?ll leave it there|that.?s helpful|got it|okay|great)\b",
    re.IGNORECASE,
)


def _is_substantive_question(text: str) -> bool:
    """Return False for courtesy closings that are not real analyst questions."""
    stripped = " ".join(text.split())
    if not stripped:
        return False

    words = re.findall(r"\b[\w.]+\b", stripped)
    has_question_mark = "?" in stripped
    has_question_cue = bool(_QUESTION_CUES.search(stripped))
    looks_like_closing = bool(_COURTESY_CLOSING.search(stripped))

    if len(words) < 8 and looks_like_closing and not has_question_mark:
        return False
    if looks_like_closing and not has_question_mark and not has_question_cue:
        return False
    if not has_question_mark and not has_question_cue:
        return len(words) >= 12
    return True


def pair_qa(turns: list[Turn]) -> list[QAPair]:
    """
    Build Q&A pairs from the Q&A section.
    A question is an analyst turn; the answer is every subsequent executive turn
    until the next analyst turn or operator handoff.
    """
    pairs: list[QAPair] = []
    qa_turns = [t for t in turns if t.section == CallSection.QA]
    i = 0
    while i < len(qa_turns):
        if (
            qa_turns[i].speaker_role == SpeakerRole.ANALYST
            and _is_substantive_question(qa_turns[i].text)
        ):
            question = qa_turns[i]
            answers: list[Turn] = []
            j = i + 1
            while j < len(qa_turns) and qa_turns[j].speaker_role != SpeakerRole.ANALYST:
                if qa_turns[j].speaker_role == SpeakerRole.EXECUTIVE:
                    answers.append(qa_turns[j])
                j += 1
            if answers:
                pairs.append(
                    QAPair(
                        pair_id=f"qa-{question.turn_id}",
                        question_turn=question,
                        answer_turns=answers,
                    )
                )
            i = j
        else:
            i += 1
    return pairs


def load_from_huggingface(
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    limit: int | None = None,
) -> Iterator[EarningsCall]:
    """
    Stream earnings calls from `kurry/sp500_earnings_transcripts`.
    Filter by ticker and year if specified.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install with: pip install datasets") from e

    ds = load_dataset("kurry/sp500_earnings_transcripts", split="train", streaming=True)
    seen = 0
    for rec in ds:
        if tickers and rec.get("symbol") not in tickers:
            continue
        if years and rec.get("year") not in years:
            continue
        try:
            call = _record_to_call(rec)
        except Exception:
            continue
        yield call
        seen += 1
        if limit and seen >= limit:
            return


def _parse_raw_content(content: str) -> list[tuple[str, str]]:
    """
    Fallback parser for the raw `content` string when `structured_content`
    is empty. Lines look like "Speaker Name: utterance text...".
    """
    segments: list[tuple[str, str]] = []
    pattern = re.compile(r"(?m)^([A-Z][A-Za-z.''\-]+(?:\s+[A-Z][A-Za-z.''\-]+){0,4})\s*:\s")
    matches = list(pattern.finditer(content))
    for idx, m in enumerate(matches):
        speaker = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        if text:
            segments.append((speaker, text))
    return segments


def _record_to_call(rec: dict) -> EarningsCall:
    """Map a HuggingFace record into our EarningsCall schema."""
    structured = rec.get("structured_content") or []

    segments: list[tuple[str, str]] = []
    for seg in structured:
        speaker_name = (seg.get("speaker") or "Unknown").strip()
        text = (seg.get("text") or "").strip()
        if text:
            segments.append((speaker_name, text))
    if not segments:
        segments = _parse_raw_content(rec.get("content") or "")

    turns: list[Turn] = []
    for pos, (speaker_name, text) in enumerate(segments):
        turns.append(
            Turn(
                turn_id=f"t-{pos}-{uuid.uuid4().hex[:6]}",
                speaker_name=speaker_name,
                speaker_role=SpeakerRole.UNKNOWN,  # assigned by context below
                speaker_title=None,
                section=CallSection.OTHER,
                text=text,
                word_count=len(text.split()),
                position=pos,
            )
        )

    # The HF dataset carries no titles → classify by call structure.
    turns = classify_speakers_by_context(turns)
    pairs = pair_qa(turns)

    call_date_str = rec.get("date")
    if isinstance(call_date_str, str):
        call_date = datetime.strptime(call_date_str[:10], "%Y-%m-%d").date()
    else:
        call_date = date.today()

    return EarningsCall(
        ticker=rec.get("symbol", "UNK"),
        company_name=rec.get("company_name", rec.get("symbol", "Unknown")),
        quarter=f"Q{rec.get('quarter', 1)}",
        year=int(rec.get("year", call_date.year)),
        call_date=call_date,
        turns=turns,
        qa_pairs=pairs,
    )


def load_sample(sample_dir: Optional[Path] = None) -> EarningsCall:
    """Load a single bundled sample call for tests and offline dev."""
    sample_dir = sample_dir or (config.DATA_DIR / "sample_transcripts")
    sample_file = sample_dir / "sample_call.json"
    if not sample_file.exists():
        raise FileNotFoundError(
            f"No sample at {sample_file}. Run scripts/build_sample.py first."
        )
    with sample_file.open() as f:
        data = json.load(f)
    return EarningsCall.model_validate(data)
