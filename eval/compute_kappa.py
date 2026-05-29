"""
Compute Cohen's kappa for two label columns, with per-category agreement.

Inputs are JSONL files. Use either:

    python -m eval.compute_kappa labels_a.jsonl labels_b.jsonl

or compare two columns in the same file:

    python -m eval.compute_kappa double_labeled.jsonl --col-a labeler_a --col-b labeler_b

Rows are matched by `id` when present, then exact `question`, then line order.
Comment lines starting with # are ignored.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

VALID_LABELS = [
    "direct_answer",
    "partial_answer",
    "reframed_question",
    "deferred",
    "non_answer",
]


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            row["_line_index"] = len(rows)
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def _key(row: dict) -> str:
    if row.get("id"):
        return f"id:{row['id']}"
    if row.get("question"):
        return f"q:{row['question']}"
    return f"line:{row['_line_index']}"


def _label(row: dict, col: str, path: Path) -> str:
    label = row.get(col)
    if label not in VALID_LABELS:
        raise ValueError(
            f"{path}:{row['_source_line']} has invalid {col}={label!r}; "
            f"expected one of {VALID_LABELS}"
        )
    return label


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    n = len(labels_a)
    if n != len(labels_b):
        raise ValueError("label lists must have the same length")
    if n == 0:
        raise ValueError("no overlapping labels")

    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[x] / n) * (counts_b[x] / n) for x in VALID_LABELS)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def per_category(labels_a: list[str], labels_b: list[str]) -> list[dict]:
    rows = []
    for label in VALID_LABELS:
        a_count = sum(x == label for x in labels_a)
        b_count = sum(x == label for x in labels_b)
        both = sum(a == label and b == label for a, b in zip(labels_a, labels_b))
        either = sum(a == label or b == label for a, b in zip(labels_a, labels_b))
        rows.append({
            "label": label,
            "labeler_a": a_count,
            "labeler_b": b_count,
            "both": both,
            "either": either,
            "agreement_when_either": both / either if either else 1.0,
        })
    return rows


def compare(path_a: Path, path_b: Path, col_a: str, col_b: str) -> dict:
    rows_a = {_key(row): row for row in _load(path_a)}
    rows_b = {_key(row): row for row in _load(path_b)}
    keys = sorted(set(rows_a) & set(rows_b))
    if not keys:
        raise ValueError("no overlapping examples between label inputs")

    labels_a = [_label(rows_a[k], col_a, path_a) for k in keys]
    labels_b = [_label(rows_b[k], col_b, path_b) for k in keys]
    agreements = sum(a == b for a, b in zip(labels_a, labels_b))
    return {
        "n_overlap": len(keys),
        "agreements": agreements,
        "raw_agreement": agreements / len(keys),
        "kappa": cohen_kappa(labels_a, labels_b),
        "per_category": per_category(labels_a, labels_b),
    }


def _print(result: dict) -> None:
    print(f"Overlapping examples: {result['n_overlap']}")
    print(f"Raw agreement: {result['raw_agreement']:.3f} ({result['agreements']}/{result['n_overlap']})")
    print(f"Cohen's kappa: {result['kappa']:.3f}")
    print("\nPer-category agreement")
    print("Category | A count | B count | Both | Either | Agreement when either")
    print("---|---:|---:|---:|---:|---:")
    for row in result["per_category"]:
        print(
            f"{row['label']} | {row['labeler_a']} | {row['labeler_b']} | "
            f"{row['both']} | {row['either']} | {row['agreement_when_either']:.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa for two label columns")
    parser.add_argument("labeler_a", type=Path, help="JSONL file for labeler A, or double-labeled JSONL")
    parser.add_argument("labeler_b", type=Path, nargs="?", help="Optional JSONL file for labeler B")
    parser.add_argument("--col-a", default="ground_truth", help="Label column in first file")
    parser.add_argument("--col-b", default="ground_truth", help="Label column in second file, or same file if omitted")
    args = parser.parse_args()

    path_b = args.labeler_b or args.labeler_a
    _print(compare(args.labeler_a, path_b, args.col_a, args.col_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
