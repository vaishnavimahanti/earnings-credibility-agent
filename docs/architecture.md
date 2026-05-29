# Architecture

## High-level flow

```
EarningsCall (parsed)
        │
        ▼
┌───────────────────┐
│  Features         │  Classical NLP: hedging density, specificity,
│  (extractors.py)  │  on-topic score, script adherence — runs in ms
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Dodge Classifier │  Per Q&A pair → DodgeLabel
│  (dodge.py)       │  Uses LLM with structured output + features as priors
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Index self       │  Add this call to Chroma so future quarters have it
│  (retriever.py)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Contradictions   │  For each Q&A → retrieve same-topic prior-quarter chunk
│  (contradiction)  │  → LLM judges if claims contradict
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Scorer           │  Aggregate features + dodge labels → 0-100 score
│  (scorer.py)      │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Brief Writer     │  LLM assembles citation-grounded analyst memo
│  (writer.py)      │
└────────┬──────────┘
         │
         ▼
   AnalystBrief
```

## Why classical features run first

Two reasons:

1. **Cost.** Hedging density, specificity, on-topic similarity, and script
   adherence run in milliseconds without an LLM call. Computing them first
   lets us send the LLM a cheap prior instead of asking it to derive
   linguistic measurements from scratch.

2. **Calibration anchor.** When the LLM and the classical signal agree,
   confidence is high. When they disagree (e.g., low hedging density but
   the LLM labels NON_ANSWER), the disagreement is itself diagnostic and
   shows up in the audit log.

## Why LangGraph over plain function composition

The five-node graph is overkill for sequential execution — `analyze_call`
could be a single function. LangGraph earns its place because:

- Each node can be retried independently on a failure.
- LangSmith tracing slots in with zero changes to node code.
- Extending to a Judge agent (running on the brief output) is one node + one edge.

## Why Chroma over Pinecone

Chroma is local, free, and zero-config. In a production deployment at a bank,
this layer swaps to Pinecone, Weaviate, or an internal vector store with no
agent-code changes. The interface is `index_call()` and `retrieve_prior_context()`.

## Data flow guarantees

- The current call is excluded from contradiction retrieval (no leakage).
- Future-quarter calls are excluded by `(year, quarter)` comparison.
- Every LLM call is logged with `audit_llm_call()` for SR 11-7 compliance.
- PII redaction is a single function call available to any node that
  touches transcript text.

## Where the project would expand in production

- **Audio path.** Run Whisper on the call audio for speaker diarization
  and recover prosodic features (pause length, pitch — Mayew & Venkatachalam 2012).
- **Cross-call retrieval at scale.** Index broker research, prior 10-Qs, and
  proxy statements alongside transcripts.
- **Eval gate in CI.** Run the 30-example regression eval on every PR; block merges
  that drop dodge-F1 by more than 0.05.
- **Human-in-the-loop dataset growth.** Every label an analyst overrides in
  the UI becomes a new eval example.
