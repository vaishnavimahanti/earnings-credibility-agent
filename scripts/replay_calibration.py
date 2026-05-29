"""
Replay a small calibration baseline against the dodge classifier.

By default this script only lists/validates the replay cases. Use --run-live to
call the configured LLM and compare predicted labels against each case's allowed
labels. This keeps normal local use free of accidental API spend.

Examples:
    python scripts/replay_calibration.py --list
    python scripts/replay_calibration.py --run-live
    python scripts/replay_calibration.py --run-live --json-out data/replay/latest_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import CallSection, DodgeCategory, QAPair, SpeakerRole, Turn

ROOT = Path(__file__).parent.parent
DEFAULT_BASELINE = ROOT / "data" / "replay" / "aapl_fq4_2024_baseline.json"


def _load_baseline(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    if "cases" not in data or not isinstance(data["cases"], list):
        raise ValueError(f"{path} must contain a 'cases' list")
    for idx, case in enumerate(data["cases"]):
        missing = {"id", "question", "answer", "allowed_labels"} - set(case)
        if missing:
            raise ValueError(f"case {idx} missing required keys: {sorted(missing)}")
        for label in case["allowed_labels"]:
            DodgeCategory(label)
    return data


def _pair(case: dict) -> QAPair:
    question = case["question"]
    answer = case["answer"]
    q = Turn(
        turn_id=f"{case['id']}-q",
        speaker_name="Analyst",
        speaker_role=SpeakerRole.ANALYST,
        section=CallSection.QA,
        text=question,
        word_count=len(question.split()),
        position=0,
    )
    a = Turn(
        turn_id=f"{case['id']}-a",
        speaker_name="Executive",
        speaker_role=SpeakerRole.EXECUTIVE,
        section=CallSection.QA,
        text=answer,
        word_count=len(answer.split()),
        position=1,
    )
    return QAPair(pair_id=case["id"], question_turn=q, answer_turns=[a])


def _run_case(case: dict) -> dict:
    from src.agents import dodge as dodge_agent

    label = dodge_agent.classify(_pair(case))
    allowed = set(case["allowed_labels"])
    return {
        "id": case["id"],
        "expected": sorted(allowed),
        "predicted": label.category.value,
        "confidence": label.confidence,
        "passed": label.category.value in allowed,
        "reasoning": label.reasoning,
    }


def _print_cases(data: dict) -> None:
    print(f"Baseline: {data.get('name', 'unnamed')}")
    print(f"Cases: {len(data['cases'])}")
    for case in data["cases"]:
        allowed = ", ".join(case["allowed_labels"])
        print(f"- {case['id']}: allowed={allowed}")


def _print_results(results: list[dict]) -> None:
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"Replay result: {passed}/{total} passed")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(
            f"[{mark}] {r['id']} -> {r['predicted']} "
            f"(allowed: {', '.join(r['expected'])}; conf={r['confidence']:.0%})"
        )
        if not r["passed"]:
            print(f"       {r['reasoning']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay dodge-classifier calibration cases")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--list", action="store_true", help="List cases without LLM calls")
    parser.add_argument("--run-live", action="store_true", help="Call the configured LLM")
    parser.add_argument("--json-out", type=Path, help="Write live replay results to JSON")
    args = parser.parse_args()

    data = _load_baseline(args.baseline)
    if args.list or not args.run_live:
        _print_cases(data)
        if not args.run_live:
            print("\nUse --run-live to call the classifier.")
        return 0

    results = [_run_case(case) for case in data["cases"]]
    _print_results(results)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "baseline": str(args.baseline),
            "results": results,
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json_out}")

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
