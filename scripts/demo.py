"""
End-to-end CLI demo.

Usage:
    python scripts/demo.py --ticker AAPL --year 2024
    python scripts/demo.py --sample
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator import EmptyCallError, analyze_call
from pydantic import ValidationError
from src.utils import data_loader


def main():
    parser = argparse.ArgumentParser(description="Earnings credibility agent demo")
    parser.add_argument("--ticker", help="Ticker (e.g. AAPL)")
    parser.add_argument("--year", type=int, help="Year (e.g. 2024)")
    parser.add_argument("--sample", action="store_true", help="Use bundled sample call")
    parser.add_argument("--out", help="Write brief JSON to this path")
    args = parser.parse_args()

    if args.sample:
        print("Loading sample call…")
        call = data_loader.load_sample()
    else:
        if not args.ticker or not args.year:
            parser.error("--ticker and --year required (or use --sample)")
        print(f"Streaming {args.ticker} {args.year} from HuggingFace…")
        calls = list(
            data_loader.load_from_huggingface(
                tickers=[args.ticker], years=[args.year], limit=1
            )
        )
        if not calls:
            sys.exit(f"No call found for {args.ticker} {args.year}")
        call = calls[0]

    print(f"Loaded {call.ticker} {call.quarter} {call.year} — "
          f"{len(call.turns)} turns, {len(call.qa_pairs)} Q&A pairs")
    print("Running multi-agent analysis…")
    try:
        brief = analyze_call(call)
    except EmptyCallError as e:
        sys.exit(
            f"\nCould not analyze this call:\n  {e}\n\n"
            f"Tip: inspect the transcript's speaker labels. If the Q&A section "
            f"wasn't detected, the credibility score would be meaningless."
        )
    except ValidationError as e:
        sys.exit(
            f"\nCould not analyze this call because an LLM response did not match "
            f"the expected schema:\n  {e}\n\n"
            f"Tip: retry once; if it repeats, inspect the schema constraint named "
            f"in the error above."
        )

    print("\n" + "=" * 70)
    print(f"BRIEF — {brief.ticker} {brief.quarter}")
    print("=" * 70)
    print(f"\nHeadline: {brief.headline}\n")
    print(f"Credibility score: {brief.credibility.overall_score}/100")
    print(f"Dodge rate: {brief.credibility.dodge_rate:.0%}")
    print(f"Contradictions found: {brief.credibility.contradiction_count}")
    print(f"Flagged Q&A pairs: {len(brief.qa_dodges)}")

    if brief.key_concerns:
        print("\n--- KEY CONCERNS ---")
        for s in brief.key_concerns:
            print(f"\n{s.heading}")
            print(s.content)

    if brief.qa_dodges:
        print("\n--- FLAGGED Q&A ---")
        for pair_id, label in brief.qa_dodges:
            print(f"\n[{label.category.value.upper()} | conf={label.confidence:.0%}]")
            print(f"  Q: {label.evidence_from_question}")
            print(f"  A: {label.evidence_from_answer}")
            print(f"  → {label.reasoning}")

    if args.out:
        Path(args.out).write_text(brief.model_dump_json(indent=2))
        print(f"\nFull brief written to {args.out}")


if __name__ == "__main__":
    main()
