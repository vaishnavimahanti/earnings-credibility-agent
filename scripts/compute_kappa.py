"""
Compute inter-rater agreement (Cohen's kappa) between two CSV label columns.

Use this for the double-labeled subset of the held-out evaluation set.

Usage:
    python scripts/compute_kappa.py eval/kappa_subset.csv
    python scripts/compute_kappa.py eval/kappa_subset.csv --col-a label_a --col-b label_b
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def cohens_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be non-empty and equal length")
    n = len(a)
    labels = sorted(set(a) | set(b))
    observed = sum(x == y for x, y in zip(a, b)) / n
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _binary(labels: list[str]) -> list[str]:
    return ["direct" if label == "direct_answer" else "dodge" for label in labels]


def _interpret(kappa: float) -> str:
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa from CSV label columns")
    parser.add_argument("csv_path")
    parser.add_argument("--col-a", default="label_a")
    parser.add_argument("--col-b", default="label_b")
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.csv_path).open()))
    labels_a = [(row.get(args.col_a) or "").strip() for row in rows]
    labels_b = [(row.get(args.col_b) or "").strip() for row in rows]
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a and b]
    if not pairs:
        sys.exit(f"No rows with both {args.col_a} and {args.col_b} filled.")

    labels_a = [a for a, _ in pairs]
    labels_b = [b for _, b in pairs]
    kappa_5way = cohens_kappa(labels_a, labels_b)
    kappa_binary = cohens_kappa(_binary(labels_a), _binary(labels_b))
    raw_agreement = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)

    print(f"\nInter-rater agreement on {len(labels_a)} double-labeled pairs")
    print("=" * 55)
    print(f"Raw agreement (5-way):   {raw_agreement:.1%}")
    print(f"Cohen's kappa (5-way):   {kappa_5way:.3f}  ({_interpret(kappa_5way)})")
    print(f"Cohen's kappa (binary):  {kappa_binary:.3f}  ({_interpret(kappa_binary)})")
    print("=" * 55)

    disagreements = Counter(f"{a} vs {b}" for a, b in zip(labels_a, labels_b) if a != b)
    if disagreements:
        print("\nMost common disagreements (rater A vs rater B):")
        for pair, count in disagreements.most_common(5):
            print(f"  {count:2d}x  {pair}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
