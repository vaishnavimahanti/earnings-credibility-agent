"""
Print the highest-signal classifier errors from an eval result JSON.

Usage:
    python -m eval.audit_errors
    python -m eval.audit_errors --results eval/results/dodge_eval_latest.json --limit 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit eval misclassifications")
    parser.add_argument("--results", type=Path, default=Path("eval/results/dodge_eval_latest.json"))
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    errors = results.get("errors", [])
    print(f"{len(errors)} errors / {results.get('n_examples')} examples")
    print(f"Binary: {results.get('binary_dodge')}")
    if results.get("material_dodge"):
        print(f"Material: {results.get('material_dodge')}")

    for i, err in enumerate(errors[: args.limit], 1):
        print("\n" + "=" * 88)
        print(
            f"{i}. {err.get('ticker', '')} "
            f"truth={err['ground_truth']} predicted={err['predicted']} "
            f"conf={err.get('confidence')}"
        )
        print(f"Q: {err.get('question_full', err.get('question', ''))}")
        print(f"A: {err.get('answer_full', err.get('answer', ''))}")
        if err.get("reasoning"):
            print(f"WHY: {err['reasoning']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
