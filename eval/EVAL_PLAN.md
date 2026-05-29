# Evaluation plan

## Current status: prototype regression eval

The committed eval set currently has 30 hand-labeled Q&A examples. It is useful
for regression testing the dodge taxonomy and the AAPL calibration fixes, but it
is **not** a production-quality estimate of generalization.

Current reported metrics should be framed as prototype metrics:

- 30 hand-labeled examples
- Includes 5 AAPL FQ4 2024 calibration examples added after prompt tuning
- Binary DIRECT-vs-DODGE F1: 0.973 on the latest run
- Five-way accuracy: 0.867 on the latest run

This is a strong prototype signal, not a claim that the classifier will hold at
that level on unseen calls.

## Evaluation milestones

There are two separate tracks. Do not mix them in reporting.

### Sample/prototype regression track

These are calibration and regression checks. They are useful for catching prompt
regressions, but they are not real held-out dataset results.

| Stage | Target examples | Purpose |
|---|---:|---|
| sample_25 | 25 | Initial prototype eval scale |
| sample_50 | 50 | Larger calibration regression set |

### Real held-out dataset track

These are the numbers that can eventually support a serious claim about model
performance.

| Stage | Target examples | Calls | Purpose |
|---|---:|---|---|
| real_50 | 50 | ~4 unseen calls | Small held-out smoke test |
| real_200 | 200 | ~16 unseen calls | First credible real held-out estimate with enough rows to inspect false positives |
| real_300 | 300 | ~24 unseen calls | Sector and quarter coverage; tighter false-positive estimates |
| real_500 | 500 | ~40 unseen calls | Model-risk-ready eval with double-label subset and holdout discipline |

Each held-out stage should include:

- Calls not used during prompt tuning
- Multiple sectors and quarters
- A few complete calls held out entirely until the final run
- A double-labeled subset for Cohen's kappa (30-40 examples at real_200; 50+ by real_500)
- Binary DIRECT-vs-DODGE F1 and five-way accuracy reported separately
- False positives reviewed with special attention to the DIRECT/PARTIAL boundary

The target statement should look like this once the work is done:

> On a 150-example held-out eval across 12 unseen earnings calls, the classifier
> reached 0.91 binary F1, with kappa=0.84 inter-rater agreement on a 30-example
> double-labeled subset.

Until then, use the current sample/regression number as a regression check and
be explicit that it is calibration-informed.

## Labeling protocol sketch

For each analyst question and management response, label exactly one category:

- `direct_answer`: primary question answered with concrete, usable information
- `partial_answer`: primary question engaged, but the core ask remains materially unresolved
- `reframed_question`: executive answers a related but different question
- `deferred`: explicit punt, offline follow-up, future event, IR referral, or legal refusal
- `non_answer`: boilerplate or no substantive engagement

Label against the **primary question**, not every clause in a stacked analyst
question. Normal disclosure discipline should not be counted as a dodge when the
primary question receives a useful answer.

## Holdout discipline

Do not use held-out calls to tune the prompt. Keep them frozen until a final eval
run. If a held-out error leads to a rubric change, move that call into the
calibration set and replace it with another unseen holdout call before quoting a
new out-of-sample number.
