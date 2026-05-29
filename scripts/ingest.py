"""
Bulk ingest earnings calls from HuggingFace into the Chroma index.

Usage:
    python scripts/ingest.py --tickers AAPL MSFT NVDA --years 2022 2023 2024
    python scripts/ingest.py --tickers AAPL --years 2024 --limit 4
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import retriever
from src.compliance import log_ingestion
from src.utils import data_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    print(f"Ingesting calls for tickers={args.tickers}, years={args.years}, limit={args.limit}")
    n_calls = 0
    n_chunks = 0
    for call in data_loader.load_from_huggingface(
        tickers=args.tickers, years=args.years, limit=args.limit
    ):
        chunks = retriever.index_call(call)
        log_ingestion(
            source_url=f"huggingface://kurry/sp500_earnings_transcripts/{call.ticker}/{call.year}/{call.quarter}",
            ticker=call.ticker, quarter=call.quarter, year=call.year,
        )
        n_calls += 1
        n_chunks += chunks
        print(f"  [{n_calls}] {call.ticker} {call.quarter} {call.year} → {chunks} chunks")

    print(f"\nDone. Indexed {n_calls} calls, {n_chunks} chunks total.")


if __name__ == "__main__":
    main()
