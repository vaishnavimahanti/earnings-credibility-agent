# Compliance, limitations, and what I'd need to ship this for real

This document is the section a banking or asset-management hiring manager
will scroll to before any other. It's deliberately written without
sugarcoating.

## 1. Compliance surface this prototype acknowledges (but does not solve)

### Regulation FD
Earnings calls are public the moment they are released. Operating on
post-release transcripts does not create an FD issue. The system explicitly
refuses queries that look like they target non-public materials
(`refuse_if_non_public()` in `src/compliance.py`).

### MNPI
Material non-public information must never enter the system. For this
prototype, that means:
- **Inputs are public transcripts only.** Provenance is logged on every
  ingestion via `log_ingestion()`.
- **No interview notes, no draft transcripts, no internal models.** A
  production deployment at a buy-side firm would need an isolated tenancy
  and a separate ingestion pipeline with stricter source allowlisting.

### SR 11-7 (Model Risk Management Guidance)
Every LLM call is logged to `data/audit.log` with: timestamp, model name,
prompt hash, response hash, latency, schema name, and success/error state.
A production deployment would add:
- Versioned prompt templates with cryptographic signatures
- Reproducibility tests on a frozen eval set per model version
- A model card per release documenting known failure modes
- Human-review queue for outputs above a configurable risk threshold

### Audit trail
The `data/audit.log` file is a JSONL stream of every LLM invocation and every
transcript ingestion. It is meant to satisfy the "auditable decisions" piece
of MRM, not to be the audit itself.

## 2. Known failure modes

### False positives on DIRECT answers
The classifier is more likely to over-call DODGE than under-call it.
Consequence: an analyst skimming the dashboard may scroll past a Q&A pair
that was actually a direct answer. Mitigation: every label carries an
evidence quote and a confidence score, so an analyst can scan and override
in one click.

### Brittle to non-US calls and translated transcripts
The Loughran-McDonald dictionary is English-language, US-finance-specific.
Calls from non-US issuers will likely score with elevated false-positive
rates. Production fix: vertical word lists per jurisdiction; multilingual
embeddings for the script-adherence signal.

### Sentiment-shift confounding
A genuine reversal of guidance ("we no longer expect to be profitable in
2025") will look like a contradiction, which it IS — but it's also
*correct* reporting on management's part. The system flags it; the analyst
decides whether to read it as deception or as appropriate disclosure.

### Survivorship bias in the backtest
The S&P 500 dataset survives whichever companies are currently in the index.
A signal that "works" on this corpus may not generalize to delisted or
smaller-cap issuers. The README backtest reports this limitation explicitly.

### Latency
End-to-end analysis on a typical call (50–80 Q&A pairs after expansion)
runs about 60–120 seconds with parallel LLM calls disabled. With parallel
LLM calls (`asyncio.gather`), it drops to roughly 30 seconds. This is fine
for batch overnight runs and too slow for "live during the call" use.

## 3. What I would need to ship this for real

In rough order of effort:

1. **A real eval set built with domain experts.** The current 30-example set
   is a prototype regression set, not a production-quality estimate. The next
   milestone is ~200 examples across ~16 unseen calls, a few full-call holdouts,
   and a 30-example subset double-labeled by another person with Cohen's-kappa
   inter-rater agreement reporting. A larger production eval would then scale
   toward ~500+ examples labeled by domain experts.

2. **Backtest at scale on a held-out time window.** The included backtest
   uses the same calls that built the index, which is a leakage problem.
   A real backtest would index calls through 2022, then run the system on
   2023-2024 and measure forward returns out-of-sample.

3. **Speech features.** Mayew & Venkatachalam (2012) showed pause patterns
   and pitch contour add incremental information beyond text. Adding Whisper
   + a prosody extractor is a one-week project that would meaningfully
   improve the dodge classifier on borderline cases.

4. **Production model risk documentation.** A model card, a stability test,
   a stress test, and a versioned prompt registry. None of these are hard.
   All of them are required at a real bank before this system would touch a
   PM workflow.

5. **Inter-quarter topic alignment.** The current contradiction detector
   relies on retrieval similarity to find the right "prior claim" to compare
   to. A better approach: cluster Q&A topics across the full call history,
   then compare same-cluster claims directly. This is the single change
   that would most improve contradiction precision.

6. **Continuous learning loop.** Every override an analyst makes in the UI
   should land in a feedback queue that grows the eval set and (eventually)
   feeds a fine-tuned classifier.

## 4. What a vendor (AlphaSense, Hebbia, Rogo) would need to copy this

Honest answer: any of them could build this faster than I can. The work
documented here would take a single AI engineer at one of those companies
roughly two to three weeks. The value of this project is not the IP — it
is demonstrating that I, an applicant with 2.5 years of experience, can
identify the gap, do the academic legwork, and build a defensible prototype
that ships.

That's the bar this project is built against, and that's the honest framing.

## Addendum: the empty-call failure mode (found during testing)

During testing on real S&P 500 transcripts, an AAPL Q4 2024 call returned a
credibility score of exactly 50/100 with zero flagged Q&A pairs — and the
brief writer produced a page of confident-sounding analysis about "China
exposure" and "analysts failing to probe" that **was not grounded in the
transcript at all.**

Root cause: the `kurry/sp500_earnings_transcripts` dataset segments speakers
as `{speaker, text}` with **no title or role field**. The original parser
relied on title hints to distinguish analysts from executives, so every
speaker fell through to UNKNOWN, no Q&A pairs were built, and the scorer
returned its neutral 50/100 fallback. The writer then hallucinated analysis
around the empty result.

This is exactly the "no slop" failure mode this project critiques in
competitors — and it appeared in the tool itself. Three fixes:

1. **Context-based speaker classification** (`classify_speakers_by_context`):
   roles are inferred from call structure — prepared-remarks speakers are
   executives, new voices in the Q&A are analysts — instead of from titles.
2. **Raw-content fallback**: if `structured_content` is empty, the raw
   `content` string is parsed by speaker-prefixed lines.
3. **Empty-call guard** (`EmptyCallError`): the pipeline now refuses to score
   or write a brief for a call with zero Q&A pairs, surfacing the parsing
   failure loudly instead of masking it with a fabricated 50/100.

The lesson worth stating in an interview: a credibility tool that fabricates
output on bad input is worse than one that admits it failed. The guard is the
most important of the three fixes.
