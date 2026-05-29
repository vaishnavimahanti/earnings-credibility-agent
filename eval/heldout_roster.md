# Held-out evaluation roster

**Rule:** every call listed here must be one you did NOT use while writing or
tuning the rubric. These are quarantined for out-of-sample measurement only.
Do not read these transcripts during prompt iteration.

The builder script parses any line containing `TICKER YEAR Qn`, so prose is
ignored. Run:

```bash
python scripts/build_heldout_set.py --roster eval/heldout_roster.md \
    --out eval/heldout_to_label.csv --max-per-call 13
```

Then label the CSV's `ground_truth` column by hand, convert to JSONL, and run
the eval against it.

## Sample/prototype milestones

These are calibration/regression milestones, not real held-out results.

- sample_25: the original prototype eval scale
- sample_50: a larger regression set for prompt/rubric changes

## Real held-out dataset milestones

- real_50: small held-out smoke test
- real_200: first credible real held-out estimate with enough rows to inspect false positives
- real_300: broader sector/quarter coverage
- real_500: model-risk-ready candidate set with double-label subset

## Milestone real_200 - 200 examples (~16 calls, ~13 pairs each)

- JPM 2023 Q1
- BAC 2023 Q3
- KO 2023 Q2
- PG 2023 Q4
- PFE 2023 Q2
- MRK 2023 Q3
- CAT 2023 Q1
- HON 2023 Q4
- XOM 2023 Q2
- HD 2023 Q3
- VZ 2023 Q1
- NKE 2023 Q4

## Alternates

- WFC 2023 Q2
- C 2023 Q4
- PEP 2023 Q3
- CL 2023 Q1
- ABBV 2023 Q2
- LLY 2023 Q3
- GE 2023 Q2
- CVX 2023 Q4
- LOW 2023 Q1
- TGT 2023 Q3

## Milestone real_300 - add ~12 more calls

- JPM 2022 Q3
- KO 2022 Q4
- PFE 2022 Q2
- CAT 2022 Q1
- XOM 2022 Q3
- HD 2022 Q4
- TGT 2022 Q2
- LLY 2022 Q3
- WFC 2024 Q1
- PEP 2024 Q2
- GE 2024 Q1
- CVX 2024 Q2

## Milestone real_500 - add ~16 more calls

- NEE 2023 Q2
- DUK 2023 Q3
- T 2023 Q4
- TMUS 2023 Q1
- TXN 2023 Q2
- QCOM 2023 Q3
- DE 2023 Q4
- UPS 2023 Q2
- MMM 2023 Q1
- BA 2023 Q3
- F 2023 Q2
- GM 2023 Q4
- SBUX 2023 Q1
- MCD 2023 Q2
- DIS 2023 Q4
- INTC 2023 Q3

## Inter-rater subset

After ~150 labeled examples, pull ~40 into a kappa file and have a second
person label them blind. Then run:

```bash
python scripts/compute_kappa.py eval/kappa_subset.csv
```

Report binary kappa alongside the model's binary F1.
