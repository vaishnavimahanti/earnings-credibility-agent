"""
Backtest: does the credibility score predict forward returns?

For each scored call in data/scored_calls/, fetch SPY-adjusted abnormal
returns at T+1, T+5, T+30 via yfinance, then bucket by credibility decile
and report mean abnormal return per decile.

Honest framing: this is a CORRELATIONAL exercise, not a trading strategy.
If the signal is real but small, that's still a useful project finding.
If there's no signal, that's also a real result — report it honestly.

Usage:
    python scripts/backtest.py
"""

from __future__ import annotations
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_scored_calls() -> list[dict]:
    """Read pre-computed briefs from data/scored_calls/."""
    path = Path(__file__).parent.parent / "data" / "scored_calls"
    if not path.exists():
        sys.exit(
            "No scored calls found at data/scored_calls/. "
            "Run `python scripts/score_all.py` first."
        )
    out = []
    for p in sorted(path.glob("*.json")):
        out.append(json.loads(p.read_text()))
    return out


def _abnormal_return(ticker: str, call_date_str: str, horizon_days: int) -> float | None:
    """Compute (stock return − SPY return) over `horizon_days` after call."""
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        sys.exit("Install yfinance and pandas: pip install yfinance pandas")

    from datetime import datetime
    call_date = datetime.fromisoformat(call_date_str).date()
    start = call_date - timedelta(days=2)
    end = call_date + timedelta(days=horizon_days + 5)

    try:
        stock = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        if stock.empty or spy.empty:
            return None
        s_ret = (stock["Close"].iloc[-1] - stock["Close"].iloc[0]) / stock["Close"].iloc[0]
        m_ret = (spy["Close"].iloc[-1] - spy["Close"].iloc[0]) / spy["Close"].iloc[0]
        return float(s_ret - m_ret)
    except Exception:
        return None


def main():
    briefs = _load_scored_calls()
    if not briefs:
        sys.exit("No briefs available.")
    print(f"Loaded {len(briefs)} briefs")

    rows = []
    for b in briefs:
        ar1 = _abnormal_return(b["ticker"], str(b["call_date"]), 1)
        ar5 = _abnormal_return(b["ticker"], str(b["call_date"]), 5)
        ar30 = _abnormal_return(b["ticker"], str(b["call_date"]), 30)
        rows.append({
            "ticker": b["ticker"],
            "quarter": b["quarter"],
            "credibility": b["credibility"]["overall_score"],
            "abn_return_t1": ar1,
            "abn_return_t5": ar5,
            "abn_return_t30": ar30,
        })

    # Bucket into deciles by credibility score
    import statistics
    rows = [r for r in rows if r["abn_return_t5"] is not None]
    rows.sort(key=lambda r: r["credibility"])
    n = len(rows)
    deciles = []
    for d in range(10):
        chunk = rows[int(n * d / 10) : int(n * (d + 1) / 10)]
        if not chunk:
            continue
        deciles.append({
            "decile": d + 1,
            "n": len(chunk),
            "mean_credibility": round(statistics.mean(r["credibility"] for r in chunk), 1),
            "mean_abn_return_t5": round(statistics.mean(r["abn_return_t5"] for r in chunk) * 100, 2),
            "mean_abn_return_t30": round(
                statistics.mean(r["abn_return_t30"] for r in chunk if r["abn_return_t30"] is not None) * 100, 2
            ),
        })

    print("\nDecile | n | mean credibility | abn return T+5 (%) | abn return T+30 (%)")
    print("-" * 75)
    for d in deciles:
        print(f"  {d['decile']:2d}  | {d['n']:2d}|     {d['mean_credibility']:5.1f}      |"
              f"      {d['mean_abn_return_t5']:6.2f}        |       {d['mean_abn_return_t30']:6.2f}")

    out_path = Path(__file__).parent.parent / "data" / "backtest_results.json"
    out_path.write_text(json.dumps({"deciles": deciles, "raw": rows}, indent=2, default=str))
    print(f"\nResults written to {out_path}")
    print("\nReminder: this is correlational, not a trading strategy. Report what you find honestly.")


if __name__ == "__main__":
    main()
