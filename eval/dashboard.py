"""
Generate a local HTML dashboard for dodge-classifier evaluation results.

Usage:
    python -m eval.dashboard
    python -m eval.dashboard --results eval/results/dodge_eval_latest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
DEFAULT_JSON = RESULTS_DIR / "dodge_eval_latest.json"
DEFAULT_OUT = RESULTS_DIR / "dashboard.html"

DEFAULT_MILESTONES = [
    {"track": "sample_regression", "stage": "sample_25", "target_examples": 25, "status": "measured"},
    {"track": "real_heldout", "stage": "real_50", "target_examples": 50, "status": "measured"},
    {"track": "real_heldout", "stage": "real_100", "target_examples": 100, "status": "measured"},
    {"track": "real_heldout", "stage": "real_200", "target_examples": 200, "status": "measured"},
    {"track": "real_heldout", "stage": "real_500", "target_examples": 500, "status": "measured"},
]

REAL_DATASET_SOURCES = [
    {
        "stage": "real_50",
        "examples": 50,
        "method": "fresh live run",
        "result": RESULTS_DIR / "dodge_eval_20260528_181051.json",
        "labels": ROOT / "heldout_50_balanced_reviewed.jsonl",
    },
    {
        "stage": "real_100",
        "examples": 100,
        "method": "fresh live run",
        "result": RESULTS_DIR / "dodge_eval_20260529_020144.json",
        "labels": ROOT / "heldout_100_balanced_reviewed.jsonl",
    },
    {
        "stage": "real_200",
        "examples": 200,
        "method": "cached-prediction rescore",
        "result": RESULTS_DIR / "dodge_eval_20260529_010123.json",
        "labels": ROOT / "heldout_200_balanced_reviewed.jsonl",
    },
    {
        "stage": "real_500",
        "examples": 500,
        "method": "cached-prediction rescore",
        "result": RESULTS_DIR / "dodge_eval_20260529_014605.json",
        "labels": ROOT / "heldout_500_balanced_reviewed.jsonl",
    },
]


def _latest_json() -> Path:
    if DEFAULT_JSON.exists():
        return DEFAULT_JSON
    candidates = sorted(RESULTS_DIR.glob("dodge_eval_*.json"))
    if not candidates:
        raise FileNotFoundError("No eval JSON found. Run python -m eval.run_eval first.")
    return candidates[-1]


def _bar(value: float) -> str:
    pct = max(0.0, min(1.0, value))
    return f'<div class="bar"><span style="width:{pct * 100:.1f}%"></span></div>'


def _round(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _binary_metrics(truth: list[str], pred: list[str]) -> dict:
    def is_dodge(label: str) -> bool:
        return label != "direct_answer"

    tp = sum(is_dodge(t) and is_dodge(p) for t, p in zip(truth, pred))
    fp = sum((not is_dodge(t)) and is_dodge(p) for t, p in zip(truth, pred))
    fn = sum(is_dodge(t) and (not is_dodge(p)) for t, p in zip(truth, pred))
    tn = sum((not is_dodge(t)) and (not is_dodge(p)) for t, p in zip(truth, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def _material_metrics(truth: list[str], pred: list[str]) -> dict:
    material = {"reframed_question", "deferred", "non_answer"}
    tp = sum(t in material and p in material for t, p in zip(truth, pred))
    fp = sum(t not in material and p in material for t, p in zip(truth, pred))
    fn = sum(t in material and p not in material for t, p in zip(truth, pred))
    tn = sum(t not in material and p not in material for t, p in zip(truth, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def _load_reviewed_truth(path: Path) -> dict[tuple[str, str], str]:
    truth = {}
    if not path.exists():
        return truth
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        truth[(row["question"].strip(), row["answer"].strip())] = row["ground_truth"]
    return truth


def _rescore_cached_predictions(result_path: Path, reviewed_labels: Path) -> dict | None:
    if not result_path.exists() or not reviewed_labels.exists():
        return None
    result = json.loads(result_path.read_text())
    truth_by_key = _load_reviewed_truth(reviewed_labels)
    truth: list[str] = []
    pred: list[str] = []
    for example in result.get("examples", []):
        q = (example.get("question_full") or example.get("question") or "").strip()
        a = (example.get("answer_full") or example.get("answer") or "").strip()
        key = (q, a)
        if key not in truth_by_key:
            continue
        truth.append(truth_by_key[key])
        pred.append(example["predicted"])
    if not truth:
        return None
    return {
        "n_examples": len(truth),
        "accuracy": round(sum(t == p for t, p in zip(truth, pred)) / len(truth), 3),
        "binary_dodge": _binary_metrics(truth, pred),
        "material_dodge": _material_metrics(truth, pred),
    }


def _real_dataset_results() -> list[dict]:
    rows = []
    for source in REAL_DATASET_SOURCES:
        if source["method"] == "cached-prediction rescore":
            metrics = _rescore_cached_predictions(source["result"], source["labels"])
        elif source["result"].exists():
            metrics = json.loads(source["result"].read_text())
        else:
            metrics = None
        rows.append({**source, "metrics": metrics})
    return rows


def _latest_real_metrics() -> dict:
    for item in reversed(_real_dataset_results()):
        if item["metrics"] is not None:
            return item["metrics"]
    return {}


def _milestone_rows(results: dict, track: str) -> str:
    rows = []
    for m in DEFAULT_MILESTONES:
        if m.get("track") != track:
            continue
        target = m["target_examples"]
        display_n = target if m.get("status") == "measured" else 0
        progress = display_n / target if target else 0.0
        rows.append(
            "<tr>"
            f"<td>{m['stage']}</td>"
            f"<td>{target}</td>"
            f"<td>{m.get('status', 'planned')}</td>"
            f"<td>{_bar(progress)}<small>{display_n}/{target}</small></td>"
            "</tr>"
        )
    return "\n".join(rows)


def _real_result_rows() -> str:
    rows = []
    for item in _real_dataset_results():
        metrics = item["metrics"]
        if metrics is None:
            rows.append(
                "<tr>"
                f"<td>{item['stage']}</td><td>{item['examples']}</td>"
                f"<td>{item['method']}</td><td colspan='5'>missing source result</td>"
                "</tr>"
            )
            continue
        binary = metrics["binary_dodge"]
        material = metrics.get("material_dodge", {})
        rows.append(
            "<tr>"
            f"<td>{item['stage']}</td>"
            f"<td>{item['examples']}</td>"
            f"<td>{item['method']}</td>"
            f"<td>{_round(binary.get('f1'))}</td>"
            f"<td>{_round(binary.get('precision'))}</td>"
            f"<td>{_round(binary.get('recall'))}</td>"
            f"<td>{_round(metrics.get('accuracy'))}</td>"
            f"<td>{_round(material.get('f1'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _trend_chart(items: list[dict]) -> str:
    chart_items = [i for i in items if i["metrics"] is not None]
    if not chart_items:
        return "<div class='empty'>No real dataset metrics available.</div>"

    width, height = 760, 270
    left, right, top, bottom = 58, 28, 20, 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [item["examples"] for item in chart_items]
    min_x, max_x = min(xs), max(xs)

    def x_pos(x: int) -> float:
        if min_x == max_x:
            return left + plot_w / 2
        return left + ((x - min_x) / (max_x - min_x)) * plot_w

    def y_pos(y: float) -> float:
        return top + (1 - max(0.0, min(1.0, y))) * plot_h

    series = [
        ("Binary F1", "#1f6f8b", lambda m: m["binary_dodge"]["f1"]),
        ("5-way accuracy", "#4f7f52", lambda m: m["accuracy"]),
        ("Material F1", "#9a5b22", lambda m: m.get("material_dodge", {}).get("f1", 0.0)),
    ]
    lines = []
    legend = []
    for idx, (name, color, getter) in enumerate(series):
        points = [(x_pos(i["examples"]), y_pos(getter(i["metrics"]))) for i in chart_items]
        point_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        circles = "".join(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4.5' fill='{color}' />"
            for x, y in points
        )
        lines.append(
            f"<polyline points='{point_attr}' fill='none' stroke='{color}' "
            f"stroke-width='3' stroke-linecap='round' stroke-linejoin='round' />{circles}"
        )
        lx = left + idx * 170
        legend.append(
            f"<g><rect x='{lx}' y='{height - 18}' width='12' height='12' fill='{color}' />"
            f"<text x='{lx + 18}' y='{height - 8}'>{name}</text></g>"
        )

    grid = []
    for val in [0.6, 0.7, 0.8, 0.9, 1.0]:
        y = y_pos(val)
        grid.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' class='grid' />"
            f"<text x='12' y='{y + 4:.1f}'>{val:.1f}</text>"
        )
    xlabels = []
    for item in chart_items:
        x = x_pos(item["examples"])
        xlabels.append(
            f"<line x1='{x:.1f}' y1='{height-bottom}' x2='{x:.1f}' y2='{height-bottom+5}' class='axis' />"
            f"<text x='{x:.1f}' y='{height-bottom+22}' text-anchor='middle'>{item['examples']}</text>"
        )

    return (
        f"<svg class='trend' viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='Metrics by real dataset size'>"
        f"<line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' class='axis' />"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' class='axis' />"
        + "".join(grid)
        + "".join(xlabels)
        + "".join(lines)
        + "".join(legend)
        + "</svg>"
    )


def _metric_bar_panel(items: list[dict]) -> str:
    rows = []
    for item in items:
        metrics = item["metrics"]
        if metrics is None:
            continue
        binary = metrics["binary_dodge"]["f1"]
        acc = metrics["accuracy"]
        material = metrics.get("material_dodge", {}).get("f1", 0.0)
        rows.append(
            "<div class='metric-row'>"
            f"<div class='metric-name'>{item['examples']}</div>"
            "<div class='metric-pack'>"
            f"<div class='metric-cell'><span>Binary F1</span>{_bar(binary)}<b>{_round(binary)}</b></div>"
            f"<div class='metric-cell'><span>5-way accuracy</span>{_bar(acc)}<b>{_round(acc)}</b></div>"
            f"<div class='metric-cell'><span>Material F1</span>{_bar(material)}<b>{_round(material)}</b></div>"
            "</div>"
            "</div>"
        )
    return "\n".join(rows)


def _label_counts(path: Path) -> dict[str, int]:
    labels = ["direct_answer", "partial_answer", "reframed_question", "deferred", "non_answer"]
    counts = {label: 0 for label in labels}
    if not path.exists():
        return counts
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        counts[row.get("ground_truth", "")] = counts.get(row.get("ground_truth", ""), 0) + 1
    return counts


def _label_mix_panel(items: list[dict]) -> str:
    colors = {
        "direct_answer": "#4f7f52",
        "partial_answer": "#c49a3a",
        "reframed_question": "#b96b3c",
        "deferred": "#8f3f45",
        "non_answer": "#5a445f",
    }
    order = ["direct_answer", "partial_answer", "reframed_question", "deferred", "non_answer"]
    rows = []
    for item in items:
        counts = _label_counts(item["labels"])
        total = sum(counts.values()) or 1
        segments = "".join(
            f"<span style='width:{counts[label] / total * 100:.2f}%;background:{colors[label]}' "
            f"title='{label}: {counts[label]}'></span>"
            for label in order
            if counts.get(label, 0)
        )
        detail = " | ".join(f"{label.replace('_', ' ')} {counts[label]}" for label in order if counts.get(label, 0))
        rows.append(
            "<div class='mix-row'>"
            f"<div class='metric-name'>{item['examples']}</div>"
            f"<div class='stack'>{segments}</div>"
            f"<small>{detail}</small>"
            "</div>"
        )
    legend = "".join(
        f"<span class='legend'><i style='background:{colors[label]}'></i>{label.replace('_', ' ')}</span>"
        for label in order
    )
    return f"<div class='legend-row'>{legend}</div>" + "\n".join(rows)


def _heat(value: float) -> str:
    value = max(0.0, min(1.0, value))
    if value >= 0.85:
        return "#dcebd8"
    if value >= 0.70:
        return "#f2e8c7"
    if value >= 0.50:
        return "#f2d3bc"
    return "#ecc4c4"


def _per_class_heatmap(results: dict) -> str:
    rows = []
    for label, metrics in results.get("per_class", {}).items():
        rows.append(
            "<tr>"
            f"<td>{label.replace('_', ' ')}</td>"
            f"<td>{metrics['support']}</td>"
            f"<td style='background:{_heat(metrics['precision'])}'>{_round(metrics['precision'])}</td>"
            f"<td style='background:{_heat(metrics['recall'])}'>{_round(metrics['recall'])}</td>"
            f"<td style='background:{_heat(metrics['f1'])}'>{_round(metrics['f1'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _per_class_rows(results: dict) -> str:
    rows = []
    for label, m in results.get("per_class", {}).items():
        rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td>{m['support']}</td>"
            f"<td>{m['precision']}</td>"
            f"<td>{m['recall']}</td>"
            f"<td>{m['f1']}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render(results: dict) -> str:
    real_items = _real_dataset_results()
    latest_real = _latest_real_metrics() or results
    binary = latest_real["binary_dodge"]
    material = latest_real.get("material_dodge", {})
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Dodge classifier eval dashboard</title>
  <style>
    :root {{ --ink:#172026; --muted:#65727d; --line:#d9e0e6; --panel:#fbfcfd; --head:#f3f5f7; --blue:#1f6f8b; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: var(--ink); background:#ffffff; }}
    h1 {{ margin-bottom: 4px; font-size: 30px; }}
    h2 {{ margin-top: 30px; font-size: 18px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin: 24px 0; }}
    .card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: var(--panel); }}
    .card div:first-child {{ color: var(--muted); font-size: 13px; }}
    .value {{ font-size: 30px; font-weight: 750; margin-top: 4px; font-variant-numeric: tabular-nums; }}
    .grid2 {{ display:grid; grid-template-columns: minmax(520px, 1.35fr) minmax(420px, .9fr); gap:16px; align-items:stretch; }}
    .panel {{ border:1px solid var(--line); border-radius:8px; padding:14px; background:var(--panel); }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid #e2e7eb; padding: 10px; text-align: left; vertical-align: middle; }}
    th {{ background: var(--head); font-size: 12px; text-transform: uppercase; color:#4b5963; }}
    .bar {{ height: 10px; background: #edf1f4; border-radius: 999px; overflow: hidden; min-width: 120px; }}
    .bar span {{ display: block; height: 100%; background: var(--blue); }}
    small {{ color: var(--muted); display: block; margin-top: 3px; }}
    .note {{ border-left: 4px solid #b7791f; background: #fff8eb; padding: 12px 14px; margin: 18px 0; }}
    .trend {{ width:100%; height:auto; }}
    .trend text {{ fill:#586673; font-size:12px; }}
    .axis {{ stroke:#7f8b94; stroke-width:1; }}
    .grid {{ stroke:#e6ebef; stroke-width:1; }}
    .metric-row {{ display:grid; grid-template-columns:56px minmax(0, 1fr); gap:12px; align-items:start; margin:12px 0; }}
    .metric-name {{ font-weight:700; font-variant-numeric: tabular-nums; padding-top:2px; white-space:nowrap; }}
    .metric-pack {{ display:grid; gap:7px; min-width:0; }}
    .metric-cell {{ display:grid; grid-template-columns:102px minmax(120px, 1fr) 46px; gap:8px; align-items:center; font-size:12px; min-width:0; }}
    .metric-cell span {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#4b5963; }}
    .metric-cell b {{ text-align:right; font-variant-numeric: tabular-nums; }}
    .stack {{ display:flex; height:14px; overflow:hidden; border-radius:999px; background:#edf1f4; border:1px solid #d8e0e6; }}
    .stack span {{ display:block; }}
    .mix-row {{ display:grid; grid-template-columns:64px minmax(220px, 1fr); gap:10px; align-items:center; margin:10px 0; }}
    .mix-row small {{ grid-column:2; margin-top:-5px; }}
    .legend-row {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:10px; }}
    .legend {{ font-size:12px; color:#4b5963; }}
    .legend i {{ display:inline-block; width:10px; height:10px; margin-right:5px; border-radius:2px; vertical-align:-1px; }}
    .empty {{ color:var(--muted); padding:20px; }}
    @media (max-width: 1050px) {{ .cards, .grid2 {{ grid-template-columns:1fr; }} .metric-cell {{ grid-template-columns:102px minmax(120px, 1fr) 46px; }} }}
  </style>
</head>
<body>
  <h1>Dodge classifier eval dashboard</h1>
  <div class="muted">Reviewed real-dataset model tear sheet. Static HTML generated from saved eval JSONs; no LLM calls are made by this dashboard.</div>

  <div class="note">Dashboard shows sample/prototype milestones separately from reviewed real-dataset results. The 200 and 500 rows are cached-prediction rescores because the live Anthropic rerun stopped when credits were exhausted.</div>

  <div class="cards">
    <div class="card"><div>Binary F1, largest set</div><div class="value">{binary['f1']}</div></div>
    <div class="card"><div>Material F1, largest set</div><div class="value">{material.get('f1', '-')}</div></div>
    <div class="card"><div>Binary precision</div><div class="value">{binary['precision']}</div></div>
    <div class="card"><div>Binary recall</div><div class="value">{binary['recall']}</div></div>
  </div>
  <div class="muted">Material dodge counts only reframed, deferred, and non-answer labels. Partial answers remain visible in the 5-way table but are not treated as full dodges.</div>

  <h2>Metric Ladder</h2>
  <div class="grid2">
    <div class="panel">{_trend_chart(real_items)}</div>
    <div class="panel">{_metric_bar_panel(real_items)}</div>
  </div>

  <h2>Ground-Truth Label Mix</h2>
  <div class="panel">{_label_mix_panel(real_items)}</div>

  <h2>Reviewed Real Dataset Results</h2>
  <table>
    <tr><th>Stage</th><th>Examples</th><th>Method</th><th>Binary F1</th><th>Binary precision</th><th>Binary recall</th><th>5-way accuracy</th><th>Material F1</th></tr>
    {_real_result_rows()}
  </table>

  <h2>Sample/prototype regression milestones</h2>
  <table>
    <tr><th>Stage</th><th>Target examples</th><th>Status</th><th>Progress</th></tr>
    {_milestone_rows(results, 'sample_regression')}
  </table>

  <h2>Real held-out dataset milestones</h2>
  <table>
    <tr><th>Stage</th><th>Target examples</th><th>Status</th><th>Progress</th></tr>
    {_milestone_rows(results, 'real_heldout')}
  </table>

  <h2>Latest Per-Class Heatmap</h2>
  <table>
    <tr><th>Category</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
    {_per_class_heatmap(results)}
  </table>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate eval dashboard HTML")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    result_path = args.results or _latest_json()
    with result_path.open() as f:
        results = json.load(f)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(results))
    print(f"Dashboard written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
