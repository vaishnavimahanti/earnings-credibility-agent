"""
Build a HELD-OUT labeling set from calls you did NOT use while tuning.

This is the bridge from "30-example regression set" to a real out-of-sample
benchmark. It reads a roster of (ticker, year, quarter), pulls each call via
the HuggingFace loader, runs pair_qa() to get candidate Q&A pairs, and writes
a CSV with EMPTY ground_truth/notes columns for hand labeling.

Usage:
    python scripts/build_heldout_set.py --roster eval/heldout_roster.md
    python scripts/build_heldout_set.py --tickers JPM KO PFE --years 2023 \
        --out eval/heldout_to_label.csv --max-per-call 15
    python scripts/build_heldout_set.py --to-jsonl \
        eval/heldout_to_label.csv eval/heldout_100.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import data_loader

CSV_COLUMNS = [
    "call_id", "ticker", "year", "quarter", "pair_id",
    "analyst_name", "analyst_firm", "exec_name", "exec_title",
    "question", "answer", "ground_truth", "notes",
]

VALID_LABELS = {
    "direct_answer", "partial_answer", "reframed_question",
    "deferred", "non_answer",
}


def _parse_roster(path: Path) -> list[tuple[str, int, str | None]]:
    """Parse lines containing `TICKER YEAR Qn` from a markdown roster."""
    out = []
    pat = re.compile(r"\b([A-Z]{1,5})\s+(20\d{2})(?:\s+(Q[1-4]))?\b")
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3)))
    return out


def _emit_csv(roster: list[tuple[str, int, str | None]], out_path: Path, max_per_call: int) -> None:
    rows = []
    tickers = sorted({t for t, _, _ in roster})
    years = sorted({y for _, y, _ in roster})
    wanted_quarters = {(t, y, q) for t, y, q in roster if q}

    print(f"Streaming calls for tickers={tickers}, years={years} ...")
    seen_calls = 0
    for call in data_loader.load_from_huggingface(tickers=tickers, years=years):
        key = (call.ticker, call.year, call.quarter)
        roster_has_quarter = any(t == call.ticker and y == call.year and q for t, y, q in roster)
        if roster_has_quarter and key not in wanted_quarters:
            continue
        if not call.qa_pairs:
            print(f"  [skip] {call.ticker} {call.quarter} {call.year}: no Q&A pairs parsed")
            continue

        seen_calls += 1
        call_id = f"{call.ticker}-{call.quarter}-{call.year}"
        n = 0
        for pair in call.qa_pairs:
            if n >= max_per_call:
                break
            q = pair.question_turn
            a0 = pair.answer_turns[0]
            rows.append({
                "call_id": call_id,
                "ticker": call.ticker,
                "year": call.year,
                "quarter": call.quarter,
                "pair_id": pair.pair_id,
                "analyst_name": q.speaker_name,
                "analyst_firm": q.speaker_title or "",
                "exec_name": a0.speaker_name,
                "exec_title": a0.speaker_title or "",
                "question": pair.question_text.strip(),
                "answer": pair.answer_text.strip(),
                "ground_truth": "",
                "notes": "",
            })
            n += 1
        print(f"  [ok]   {call_id}: wrote {n} pairs")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} unlabeled pairs from {seen_calls} calls -> {out_path}")
    print("Next: open the CSV and fill the `ground_truth` column by hand.")
    print(f"Valid labels: {', '.join(sorted(VALID_LABELS))}")


def _to_jsonl(csv_path: Path, jsonl_path: Path) -> None:
    """Convert a hand-labeled CSV into the JSONL the eval harness expects."""
    kept, skipped = 0, 0
    with csv_path.open() as f, jsonl_path.open("w") as out:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth = (row.get("ground_truth") or "").strip()
            if ground_truth not in VALID_LABELS:
                skipped += 1
                continue
            rec = {
                "ticker": row["ticker"],
                "analyst_name": row.get("analyst_name", ""),
                "analyst_firm": row.get("analyst_firm", ""),
                "exec_name": row.get("exec_name", ""),
                "exec_title": row.get("exec_title", ""),
                "question": row["question"],
                "answer": row["answer"],
                "ground_truth": ground_truth,
            }
            out.write(json.dumps(rec) + "\n")
            kept += 1
    print(f"Wrote {kept} labeled examples -> {jsonl_path} (skipped {skipped})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build held-out labeling CSV/JSONL")
    parser.add_argument("--roster", help="Path to heldout_roster.md")
    parser.add_argument("--tickers", nargs="+", help="Override: tickers to pull")
    parser.add_argument("--years", nargs="+", type=int, help="Override: years to pull")
    parser.add_argument("--out", default="eval/heldout_to_label.csv")
    parser.add_argument("--max-per-call", type=int, default=15)
    parser.add_argument("--to-jsonl", nargs=2, metavar=("CSV", "JSONL"))
    args = parser.parse_args()

    if args.to_jsonl:
        _to_jsonl(Path(args.to_jsonl[0]), Path(args.to_jsonl[1]))
        return 0

    if args.roster:
        roster = _parse_roster(Path(args.roster))
    elif args.tickers and args.years:
        roster = [(ticker, year, None) for ticker in args.tickers for year in args.years]
    else:
        parser.error("Provide --roster, or both --tickers and --years, or --to-jsonl")

    if not roster:
        sys.exit("Roster is empty. Add lines like '- JPM 2023 Q2'.")

    print(f"Roster: {len(roster)} call specs")
    _emit_csv(roster, Path(args.out), args.max_per_call)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
