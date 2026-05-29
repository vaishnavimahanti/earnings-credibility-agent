# Setup guide

Step-by-step instructions to get the project running on your machine. If
anything here is unclear, file an issue or grep for the relevant test file —
the tests are the source of truth on how the code is meant to behave.

---

## Prerequisites

You need:

- **Python 3.10 or newer.** Check with `python --version`. If you're on 3.9
  or older, install a newer version from python.org or via pyenv.
- **About 2 GB of disk space** for dependencies (sentence-transformers and
  spaCy models are the main contributors).
- **An Anthropic API key** (recommended) or an OpenAI API key. The full
  end-to-end demo costs about $0.05–$0.10 in API calls.

Optional but recommended:

- **`git`** for cloning and version control.
- **A virtual environment tool.** I use `venv` (built-in); `conda` and
  `uv` also work.

---

## Step 1 — Clone the project

```bash
git clone <your-fork-url> earnings-credibility-agent
cd earnings-credibility-agent
```

If you received the project as a zip, unzip it and `cd` into the resulting
folder.

---

## Step 2 — Create and activate a virtual environment

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your shell prompt afterwards. If you don't,
the rest of the commands will install into your system Python — which works
but is not recommended.

---

## Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs about 35 packages and takes 3–5 minutes on a typical
connection. The largest ones are `sentence-transformers` (downloads a
~90 MB embedding model on first use) and `chromadb`.

If `presidio-analyzer` fails to install on your platform, it's optional —
PII redaction will fall back to a no-op. You can comment that line out
of `requirements.txt` and the project will still run.

---

## Step 4 — Configure your API key

```bash
cp .env.example .env
```

Open `.env` in your editor and set:

```ini
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get a key at https://console.anthropic.com. Free-tier credits are enough
to run the demo and the eval harness several times.

If you prefer OpenAI, comment the Anthropic line and uncomment:

```ini
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini
```

The system auto-detects which provider's key is set.

---

## Step 5 — Run the test suite

```bash
pytest tests/ -v
```

You should see **26 offline tests pass**. These don't require an API key —
they cover schema validation, transcript parsing, feature extractors,
and compliance logging.

Expected output (truncated):

```
tests/test_compliance.py::test_audit_writes_jsonl_entry PASSED
tests/test_data_loader.py::test_split_sections_uses_operator_qa_cue PASSED
tests/test_features.py::test_hedging_density_higher_for_vague_text PASSED
tests/test_schemas.py::test_earnings_call_prepared_text PASSED
...
============================== 22 passed in 0.14s ==============================
```

If any test fails at this stage, **stop and fix it before going further**.
The agents depend on these primitives being correct.

---

## Step 6 — Build the sample earnings call

```bash
python scripts/build_sample.py
```

This creates `data/sample_transcripts/sample_call.json` — a synthetic but
realistic earnings call with all five dodge categories represented (one of
each, plus two direct answers). The sample is modeled on the real call
patterns in `eval/labeled_set.jsonl`.

Expected output:

```
Sample call written to .../data/sample_transcripts/sample_call.json
  Ticker: ACME Q3 2024
  Turns: 16
  Q&A pairs: 6
```

---

## Step 7 — Run the end-to-end demo

```bash
python scripts/demo.py --sample
```

This is the first command that hits your API key. It will:

1. Load the sample call.
2. Run linguistic feature extraction on each Q&A pair (~5 seconds; downloads
   the embedding model the first time).
3. Run the dodge classifier on each pair (~10–20 seconds, six LLM calls).
4. Index the call into Chroma (instant).
5. Run contradiction detection (zero contradictions on the first call,
   since there's no prior quarter indexed yet).
6. Score the call.
7. Write the analyst brief (one final LLM call, ~10 seconds).

You'll see the brief print to your terminal, formatted like:

```
======================================================================
BRIEF — ACME Q3 2024
======================================================================

Headline: Acme delivered headline beats but management deflected on
customer concentration and capital allocation, weakening the credibility
of the strong reported numbers.

Credibility score: 62.3/100
Dodge rate: 67%
Contradictions found: 0
Flagged Q&A pairs: 4

--- FLAGGED Q&A ---
[REFRAMED_QUESTION | conf=90%]
  Q: What's the magnitude of the AI-related revenue contribution in Q3...
  A: We're incredibly excited about the AI opportunity in front of us...
  → Executive pivoted to enthusiasm and customer demand without
    quantifying AI-related revenue.
...
```

Total runtime: ~60–90 seconds. Total cost: ~$0.05.

---

## Step 8 — Launch the Streamlit UI

```bash
streamlit run app.py
```

Streamlit will open a browser tab at `http://localhost:8501`. In the
sidebar:

1. Select **Sample call** as the source.
2. Click **Analyze call**.

You'll see the same analysis from Step 7 rendered with color-coded dodge
labels, an expandable brief, and a JSON download button.

---

## Step 9 (optional) — Run the evaluation harness

```bash
python -m eval.run_eval
```

This runs the dodge classifier against the 30-example labeled set in
`eval/labeled_set.jsonl` and produces a markdown report at
`eval/results/dodge_eval_latest.md` plus a timestamped JSON.

Total runtime: ~2 minutes. Total cost: ~$0.10.

The headline metric is **binary F1 on DIRECT vs DODGE**. A working
classifier on this set lands in the **0.80–0.92** range.

---

## Step 10 (optional) — Run against real S&P 500 calls

The data layer streams from `kurry/sp500_earnings_transcripts` on
HuggingFace — a free, open dataset of 20 years of S&P 500 earnings
calls. No HuggingFace account or API key is needed.

```bash
# Index three quarters of AAPL so contradiction detection has prior context
python scripts/ingest.py --tickers AAPL --years 2023 2024 --limit 4

# Analyze the most recent indexed call
python scripts/demo.py --ticker AAPL --year 2024
```

The first run will download a portion of the dataset (~50 MB cached
locally by HuggingFace). Subsequent runs reuse the cache.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`
You're running a script from inside `src/` or `scripts/` directly. Always
run from the project root:

```bash
cd earnings-credibility-agent
python scripts/demo.py --sample
```

### `RuntimeError: No LLM API key found`
You skipped Step 4. Check that `.env` exists in the project root and
contains a valid API key.

### Streamlit shows "No sample call found"
You skipped Step 6. Run `python scripts/build_sample.py`.

### `chromadb` errors about SQLite version
Some Linux distros ship an older SQLite. Fix with:

```bash
pip install pysqlite3-binary
```

Then add this at the top of `src/agents/retriever.py` (before the
chromadb import):

```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

### Tests pass but the demo hangs
Almost always a network issue reaching `api.anthropic.com`. Test with:

```bash
curl https://api.anthropic.com/v1/messages -I
```

If that hangs, you have a firewall or proxy issue, not a code issue.

### HuggingFace streaming is slow
The dataset streams ~5–10 MB per call. If you're on a slow connection,
use `--limit 2` to cap the ingestion or pre-download the dataset:

```bash
python -c "from datasets import load_dataset; load_dataset('kurry/sp500_earnings_transcripts', split='train')"
```

---

## Project layout reference

```
.
├── README.md                   ← Start here for product narrative
├── SETUP.md                    ← You are here
├── requirements.txt            ← Dependencies
├── .env.example                ← Copy to .env and add your API key
├── app.py                      ← Streamlit UI entrypoint
├── src/
│   ├── schemas.py              ← Pydantic types — read this second
│   ├── llm.py                  ← LLM client wrapper
│   ├── compliance.py           ← Audit logging, PII, refusals
│   ├── features/               ← Classical NLP features
│   ├── agents/                 ← The multi-agent system
│   │   ├── dodge.py            ← The headline differentiator
│   │   ├── retriever.py        ← Cross-quarter vector retrieval
│   │   ├── contradiction.py    ← Same-topic claim comparison
│   │   ├── scorer.py           ← Composite credibility score
│   │   ├── writer.py           ← Brief writer with citations
│   │   └── orchestrator.py     ← LangGraph wiring
│   └── utils/data_loader.py    ← HuggingFace dataset parsing
├── tests/                      ← 22 unit tests
├── eval/
│   ├── labeled_set.jsonl       ← 25 hand-labeled Q&A pairs
│   └── run_eval.py             ← Eval harness
├── scripts/
│   ├── build_sample.py         ← Step 6
│   ├── demo.py                 ← Step 7
│   ├── ingest.py               ← Step 10
│   └── backtest.py             ← Optional forward-returns analysis
└── docs/
    ├── architecture.md
    ├── competitive_analysis.md
    └── compliance_and_limitations.md
```

---

## What to read next

1. `README.md` — the product narrative and positioning.
2. `docs/competitive_analysis.md` — what makes this different from
   AlphaSense, Hebbia, Rogo.
3. `docs/architecture.md` — the agent flow and design decisions.
4. `docs/compliance_and_limitations.md` — the honest "what doesn't work
   yet" section.
5. `src/schemas.py` — the data model. Every other file makes sense once
   you've read this.
6. `src/agents/dodge.py` — the headline component.

Total reading time: about 25 minutes.
