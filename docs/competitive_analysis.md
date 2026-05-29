# Competitive analysis

This project is positioned as a **teardown**, not a startup. The goal is to
demonstrate what an internal AI team at a bank or buy-side firm would build
to plug a specific gap in commercially available earnings-research tools.

## The four incumbents

| Product | Strength | Gap relevant to this project |
|---|---|---|
| **AlphaSense** | Largest proprietary content library (10k+ sources including broker research and expert calls); strong real-time call ingestion; mature search and sentiment. | Sentiment is built on Loughran-McDonald word lists. There is no structured Q&A response taxonomy and no quarter-over-quarter linguistic drift tracking. |
| **Hebbia** | Excellent enterprise search across uploaded internal documents; strong on long-context diligence workflows. | Not earnings-call-specific. No Q&A dodge classification. No time-series credibility tracking. |
| **Rogo** | Built specifically for IB deliverables (pitch books, comps, memos). Forbes AI 50 2026. | Per its own competitive page, no curated content library to match AlphaSense. No tone/credibility analysis. |
| **Bloomberg Terminal** | The default for institutional finance for decades. Recent NLP additions for transcript search. | Heavy UI, no agentic workflow, no Q&A-level classification. |

## The gap this project fills

Across all four products: **none of them publishes a structured taxonomy
of how management answered each analyst question.** They summarize what was
said. They don't tell you *how* it was said relative to what was asked.

Yet finance-linguistics research has shown for over a decade that this signal
matters for predicting subsequent disclosures and stock-price reactions:

- Larcker & Zakolyukina (2012) — abnormally positive managerial tone in
  conference calls predicts later financial restatements.
- Lee (2016) — adherence to prepared scripts during the Q&A is negatively
  associated with market reaction.
- Druz, Petzev, Wagner & Zeckhauser (2020) — manager-tone signals contain
  incremental information beyond the earnings surprise.

The academic methodology has not been productized at the workflow level for
sell-side or buy-side analysts. That is the gap this project demonstrates a
prototype solution for.

## What this project is NOT

- **Not a competitive product.** Nothing here ships at AlphaSense scale.
- **Not a trading signal.** The backtest is a sanity check that the signal
  is correlationally meaningful, not a strategy.
- **Not a replacement for human judgment.** Every label is grounded in
  evidence so an analyst can verify and override in one click.

## What this project IS

A working prototype that demonstrates a 2.5-year AI engineer can:

- Identify a real, underserved gap in a $4B-vendor-dominated market
- Read primary academic research and translate it into a working system
- Build production-quality scaffolding (eval, observability, compliance)
- Communicate honestly about what works and what doesn't
