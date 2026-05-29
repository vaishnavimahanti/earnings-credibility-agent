"""
Evaluation harness for the Dodge Classifier.

Reads eval/labeled_set.jsonl (hand-labeled Q&A pairs with ground-truth dodge
categories) and reports precision, recall, F1 per category plus confusion matrix.

Usage:
    python -m eval.run_eval

Output:
    - eval/results/dodge_eval_<timestamp>.json
    - eval/results/dodge_eval_latest.md
"""

from __future__ import annotations
import json
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.schemas import (
    CallSection,
    DodgeCategory,
    QAPair,
    SpeakerRole,
    Turn,
)
from src.agents import dodge as dodge_agent
from src.features import extractors


EVAL_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EVAL_MILESTONES = [
    {
        "track": "sample_regression",
        "stage": "sample_25",
        "target_examples": 25,
        "target_calls": "sample/calibration examples",
        "status": "measured",
        "purpose": "Initial prototype regression guard",
    },
    {
        "track": "sample_regression",
        "stage": "sample_50",
        "target_examples": 50,
        "target_calls": "sample/calibration examples",
        "status": "planned",
        "purpose": "Larger calibration regression set; still not a real held-out benchmark",
    },
    {
        "track": "real_heldout",
        "stage": "real_50",
        "target_examples": 50,
        "target_calls": "~4 unseen calls",
        "status": "planned",
        "purpose": "First small real held-out smoke test",
    },
    {
        "track": "real_heldout",
        "stage": "real_200",
        "target_examples": 200,
        "target_calls": "~16 unseen calls",
        "status": "planned",
        "purpose": "First credible real held-out estimate with enough rows to inspect false positives",
    },
    {
        "track": "real_heldout",
        "stage": "real_300",
        "target_examples": 300,
        "target_calls": "~24 unseen calls",
        "status": "planned",
        "purpose": "Sector and quarter coverage; tighter false-positive estimates",
    },
    {
        "track": "real_heldout",
        "stage": "real_500",
        "target_examples": 500,
        "target_calls": "~40 unseen calls",
        "status": "planned",
        "purpose": "Model-risk-ready eval with double-label subset and holdout discipline",
    },
]



def _load_labeled_set(path: Path) -> list[dict]:
    """Read JSONL of {question, answer, ground_truth} records."""
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(json.loads(line))
    return out


def _record_to_pair(rec: dict, idx: int) -> QAPair:
    """Build a QAPair from a flat eval record."""
    q_turn = Turn(
        turn_id=f"eval-q-{idx}",
        speaker_name=rec.get("analyst_name", "Analyst"),
        speaker_role=SpeakerRole.ANALYST,
        speaker_title=rec.get("analyst_firm"),
        section=CallSection.QA,
        text=rec["question"],
        word_count=len(rec["question"].split()),
        position=idx * 2,
    )
    a_turn = Turn(
        turn_id=f"eval-a-{idx}",
        speaker_name=rec.get("exec_name", "Executive"),
        speaker_role=SpeakerRole.EXECUTIVE,
        speaker_title=rec.get("exec_title"),
        section=CallSection.QA,
        text=rec["answer"],
        word_count=len(rec["answer"].split()),
        position=idx * 2 + 1,
    )
    return QAPair(pair_id=f"eval-{idx}", question_turn=q_turn, answer_turns=[a_turn])


def _confusion(pred: list[str], truth: list[str]) -> dict:
    """Build a confusion matrix as nested dict."""
    cats = sorted({*pred, *truth})
    matrix = {t: {p: 0 for p in cats} for t in cats}
    for p, t in zip(pred, truth):
        matrix[t][p] += 1
    return matrix


def _per_class_metrics(pred: list[str], truth: list[str]) -> dict:
    """Per-class precision, recall, F1."""
    out = {}
    for cat in sorted({*pred, *truth}):
        tp = sum(1 for p, t in zip(pred, truth) if p == cat and t == cat)
        fp = sum(1 for p, t in zip(pred, truth) if p == cat and t != cat)
        fn = sum(1 for p, t in zip(pred, truth) if p != cat and t == cat)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for t in truth if t == cat)
        out[cat] = {"precision": round(prec, 3), "recall": round(rec, 3),
                    "f1": round(f1, 3), "support": support}
    return out


def _binary_dodge_metrics(pred: list[str], truth: list[str]) -> dict:
    """
    Collapse 5-way labels to binary (DIRECT vs DODGE).
    This is the most-cited number for the project.
    """
    DIRECT = DodgeCategory.DIRECT.value
    pred_bin = ["dodge" if p != DIRECT else "direct" for p in pred]
    truth_bin = ["dodge" if t != DIRECT else "direct" for t in truth]
    tp = sum(1 for p, t in zip(pred_bin, truth_bin) if p == "dodge" and t == "dodge")
    fp = sum(1 for p, t in zip(pred_bin, truth_bin) if p == "dodge" and t == "direct")
    fn = sum(1 for p, t in zip(pred_bin, truth_bin) if p == "direct" and t == "dodge")
    tn = sum(1 for p, t in zip(pred_bin, truth_bin) if p == "direct" and t == "direct")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
    }


def _material_dodge_metrics(pred: list[str], truth: list[str]) -> dict:
    """
    Collapse labels to MATERIAL_DODGE vs NOT_MATERIAL_DODGE.

    PARTIAL_ANSWER is intentionally treated as not-material here. It is still
    tracked in the 5-way confusion matrix, but reporting it as the same severity
    as NON_ANSWER/REFRAMED/DEFERRED was the source of inflated headline dodge
    rates during held-out testing.
    """
    material = {
        DodgeCategory.REFRAMED.value,
        DodgeCategory.DEFERRED.value,
        DodgeCategory.NON_ANSWER.value,
    }
    pred_bin = ["material_dodge" if p in material else "not_material" for p in pred]
    truth_bin = ["material_dodge" if t in material else "not_material" for t in truth]
    tp = sum(1 for p, t in zip(pred_bin, truth_bin) if p == "material_dodge" and t == "material_dodge")
    fp = sum(1 for p, t in zip(pred_bin, truth_bin) if p == "material_dodge" and t == "not_material")
    fn = sum(1 for p, t in zip(pred_bin, truth_bin) if p == "not_material" and t == "material_dodge")
    tn = sum(1 for p, t in zip(pred_bin, truth_bin) if p == "not_material" and t == "not_material")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
    }


def _infer_eval_track(labeled_path: Path) -> str:
    name = labeled_path.name.lower()
    if "heldout" in name or "holdout" in name or "real" in name:
        return "real_heldout"
    return "sample_regression"


def _milestones_for_result(eval_track: str, n_examples: int) -> list[dict]:
    """Mark the matching milestone as measured when a held-out file is evaluated."""
    out = []
    for m in EVAL_MILESTONES:
        item = dict(m)
        if item["track"] == eval_track and item["target_examples"] == n_examples:
            item["status"] = "measured"
        out.append(item)
    return out


def run_evaluation(labeled_path: Path | None = None, eval_track: str | None = None) -> dict:
    """Run the dodge classifier against the labeled set and return metrics."""
    labeled_path = labeled_path or EVAL_DIR / "labeled_set.jsonl"
    eval_track = eval_track or _infer_eval_track(labeled_path)
    records = _load_labeled_set(labeled_path)
    if not records:
        raise RuntimeError(f"No records found at {labeled_path}")

    pred_labels: list[str] = []
    truth_labels: list[str] = []
    pred_confidences: list[float] = []
    examples: list[dict] = []

    for i, rec in enumerate(records):
        pair = _record_to_pair(rec, i)
        # We need a "fake" call to extract features — use prepared_remarks=""
        from src.schemas import EarningsCall
        fake_call = EarningsCall(
            ticker=rec.get("ticker", "TEST"),
            company_name=rec.get("ticker", "TEST"),
            quarter="Q1",
            year=2024,
            call_date=datetime.now().date(),
            turns=[pair.question_turn, pair.answer_turns[0]],
            qa_pairs=[pair],
        )
        feats = extractors.extract_features(pair, fake_call)
        label = dodge_agent.classify(pair, feats)
        pred_labels.append(label.category.value)
        truth_labels.append(rec["ground_truth"])
        pred_confidences.append(label.confidence)
        examples.append({
            "ticker": rec.get("ticker"),
            "question": rec["question"][:120],
            "answer": rec["answer"][:200],
            "question_full": rec["question"],
            "answer_full": rec["answer"],
            "ground_truth": rec["ground_truth"],
            "predicted": label.category.value,
            "confidence": label.confidence,
            "evidence_from_question": label.evidence_from_question,
            "evidence_from_answer": label.evidence_from_answer,
            "reasoning": label.reasoning,
            "correct": label.category.value == rec["ground_truth"],
        })

    n = len(records)
    accuracy = sum(1 for p, t in zip(pred_labels, truth_labels) if p == t) / n

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_examples": n,
        "accuracy": round(accuracy, 3),
        "binary_dodge": _binary_dodge_metrics(pred_labels, truth_labels),
        "material_dodge": _material_dodge_metrics(pred_labels, truth_labels),
        "per_class": _per_class_metrics(pred_labels, truth_labels),
        "confusion_matrix": _confusion(pred_labels, truth_labels),
        "avg_confidence": round(sum(pred_confidences) / n, 3),
        "eval_track": eval_track,
        "eval_stage": "sample_regression" if eval_track == "sample_regression" else "real_heldout",
        "eval_caveat": (
            "Sample/prototype regression eval; not a production generalization estimate."
            if eval_track == "sample_regression"
            else "Real held-out eval result. Confirm roster was not used during prompt tuning."
        ),
        "labeled_path": str(labeled_path),
        "milestones": _milestones_for_result(eval_track, n),
        "examples": examples,
        "errors": [e for e in examples if not e["correct"]],
    }
    return results


def write_markdown_report(results: dict, out_path: Path | None = None) -> Path:
    """Render a markdown report from the results dict."""
    out_path = out_path or RESULTS_DIR / "dodge_eval_latest.md"
    lines = [
        "# Dodge classifier evaluation",
        f"_Run at {results['timestamp']} on {results['n_examples']} hand-labeled Q&A pairs_",
        "",
        "## Headline binary metric (DIRECT vs DODGE)",
        "",
        f"- **F1: {results['binary_dodge']['f1']}**",
        f"- Precision: {results['binary_dodge']['precision']}",
        f"- Recall: {results['binary_dodge']['recall']}",
        f"- Overall accuracy (5-way): {results['accuracy']}",
        f"- Avg model confidence: {results['avg_confidence']}",
        f"- Eval stage: {results.get('eval_stage', 'prototype_regression')}",
        f"- Caveat: {results.get('eval_caveat', 'Prototype regression eval; not a production generalization estimate.')}",
        "",
        "## Material-dodge metric (REFRAMED/DEFERRED/NON_ANSWER only)",
        "",
        f"- **F1: {results.get('material_dodge', {}).get('f1', 'n/a')}**",
        f"- Precision: {results.get('material_dodge', {}).get('precision', 'n/a')}",
        f"- Recall: {results.get('material_dodge', {}).get('recall', 'n/a')}",
        "",
        "## Evaluation maturity",
        "",
        "### Sample/prototype regression milestones",
        "",
        "| Stage | Target examples | Calls | Status | Purpose |",
        "|---|---:|---|---|---|",
    ]
    for m in results.get("milestones", EVAL_MILESTONES):
        if m.get("track") != "sample_regression":
            continue
        lines.append(
            f"| {m['stage']} | {m['target_examples']} | {m['target_calls']} | "
            f"{m['status']} | {m['purpose']} |"
        )

    lines += [
        "",
        "### Real held-out dataset milestones",
        "",
        "| Stage | Target examples | Calls | Status | Purpose |",
        "|---|---:|---|---|---|",
    ]
    for m in results.get("milestones", EVAL_MILESTONES):
        if m.get("track") != "real_heldout":
            continue
        lines.append(
            f"| {m['stage']} | {m['target_examples']} | {m['target_calls']} | "
            f"{m['status']} | {m['purpose']} |"
        )

    lines += [
        "",
        "The sample rows are prototype/regression milestones. The real held-out rows are planned dataset milestones until a held-out JSONL is labeled and evaluated.",
        "",
        "## Per-class metrics",
        "",
        "| Category | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for cat, m in results["per_class"].items():
        lines.append(
            f"| {cat} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |"
        )

    lines += ["", "## Confusion matrix", "", "_Rows = truth, columns = predicted_", ""]
    matrix = results["confusion_matrix"]
    cats = list(matrix.keys())
    lines.append("| | " + " | ".join(cats) + " |")
    lines.append("|" + "---|" * (len(cats) + 1))
    for t in cats:
        row = [matrix[t].get(p, 0) for p in cats]
        lines.append(f"| **{t}** | " + " | ".join(str(x) for x in row) + " |")

    out_path.write_text("\n".join(lines))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run dodge classifier evaluation")
    parser.add_argument(
        "--labeled",
        type=Path,
        default=EVAL_DIR / "labeled_set.jsonl",
        help="Path to labeled JSONL eval set",
    )
    parser.add_argument(
        "--eval-track",
        choices=["sample_regression", "real_heldout"],
        default=None,
        help="Override eval track; inferred from --labeled by default",
    )
    args = parser.parse_args()

    print("Running dodge classifier evaluation…")
    results = run_evaluation(args.labeled, eval_track=args.eval_track)
    out_json = RESULTS_DIR / f"dodge_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    latest_json = RESULTS_DIR / "dodge_eval_latest.json"
    payload = json.dumps(results, indent=2)
    out_json.write_text(payload)
    latest_json.write_text(payload)
    md = write_markdown_report(results)
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote latest JSON: {latest_json}")
    print(f"Wrote markdown: {md}")
    print(f"\nBinary F1: {results['binary_dodge']['f1']}")
    print(f"5-way accuracy: {results['accuracy']}")


if __name__ == "__main__":
    main()
