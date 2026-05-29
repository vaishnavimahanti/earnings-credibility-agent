# Dodge classifier evaluation
_Run at 2026-05-29T02:01:44.419420 on 100 hand-labeled Q&A pairs_

## Headline binary metric (DIRECT vs DODGE)

- **F1: 0.812**
- Precision: 0.897
- Recall: 0.743
- Overall accuracy (5-way): 0.86
- Avg model confidence: 0.833
- Eval stage: real_heldout
- Caveat: Real held-out eval result. Confirm roster was not used during prompt tuning.

## Material-dodge metric (REFRAMED/DEFERRED/NON_ANSWER only)

- **F1: 0.778**
- Precision: 0.778
- Recall: 0.778

## Evaluation maturity

### Sample/prototype regression milestones

| Stage | Target examples | Calls | Status | Purpose |
|---|---:|---|---|---|
| sample_25 | 25 | sample/calibration examples | measured | Initial prototype regression guard |
| sample_50 | 50 | sample/calibration examples | planned | Larger calibration regression set; still not a real held-out benchmark |

### Real held-out dataset milestones

| Stage | Target examples | Calls | Status | Purpose |
|---|---:|---|---|---|
| real_50 | 50 | ~4 unseen calls | planned | First small real held-out smoke test |
| real_200 | 200 | ~16 unseen calls | planned | First credible real held-out estimate with enough rows to inspect false positives |
| real_300 | 300 | ~24 unseen calls | planned | Sector and quarter coverage; tighter false-positive estimates |
| real_500 | 500 | ~40 unseen calls | planned | Model-risk-ready eval with double-label subset and holdout discipline |

The sample rows are prototype/regression milestones. The real held-out rows are planned dataset milestones until a held-out JSONL is labeled and evaluated.

## Per-class metrics

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| deferred | 0.5 | 1.0 | 0.667 | 1 |
| direct_answer | 0.873 | 0.954 | 0.912 | 65 |
| non_answer | 0.75 | 0.75 | 0.75 | 4 |
| partial_answer | 0.85 | 0.654 | 0.739 | 26 |
| reframed_question | 1.0 | 0.75 | 0.857 | 4 |

## Confusion matrix

_Rows = truth, columns = predicted_

| | deferred | direct_answer | non_answer | partial_answer | reframed_question |
|---|---|---|---|---|---|
| **deferred** | 1 | 0 | 0 | 0 | 0 |
| **direct_answer** | 0 | 62 | 0 | 3 | 0 |
| **non_answer** | 0 | 1 | 3 | 0 | 0 |
| **partial_answer** | 1 | 7 | 1 | 17 | 0 |
| **reframed_question** | 0 | 1 | 0 | 0 | 3 |