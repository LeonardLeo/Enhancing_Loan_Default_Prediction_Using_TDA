# Experiment 24 — Are we recycling the same customers?

## In one sentence

If each snapshot draws `t` people and we repeat that `l` times, the average number of times a person is drawn is `R = (t × l) / n1`. The statistical checklist wants **R ≈ 1 or less**. Experiment 3 used `l = 500`, which makes R tens to hundreds.

## Who this is for

Imagine drawing 90 names from a hat of 300, then putting them back, 500 times. Most names appear over and over. The 500 barcode rows are **not** 500 independent stories — they are remixes of the same 300 people.

## Datasets (all six)

Each dataset folder has its own staged script:

`5_Experiments/24_Sampling_Ratio_Audit/<Dataset>/run_sampling_ratio_audit.py`

The file in this directory only launches those six scripts. Debug inside the dataset folder.

| Dataset | Balanced n1 | Landmark percents | Historical R (l=500) | Revised l that makes R≈1 |
|---------|-------------|-------------------|----------------------|--------------------------|
| DCCCD | 6,630 | L5 / L15 | 25 / 75 | 21 / 7 |
| Statlog German Credit | 300 | L30 / L60 | 150 / 300 | 4 / 2 |
| PKDD Czech Financial | 76 | L10 / L20 | 46 / 99 | 11 / 6 |
| Polish Bankruptcy 3-year | 495 | L10 / L20 | 49 / 100 | 11 / 5 |
| Taiwan Bankruptcy | 220 | L10 / L20 | 50 / 100 | 10 / 5 |
| South German Credit | 300 | L10 / L20 | 50 / 100 | 10 / 5 |

None of the historical `l = 500` rows pass the “R near 1” check. That is the finding. Landmark *percents* differ by dataset because `n1` differs — `docs/Design_Decisions.md` — but the reuse verdict does not: every grid over-samples at `l = 500`.

## What we do (in order)

1. Load the processed table (same file Experiment 3 starts from).
2. Count defaults vs non-defaults.
3. Set `n1 = n2 =` minority count (Experiment 3 undersamples to this size).
4. For each landmark percent, `t = floor(n1 × L / 100)`.
5. Score two snapshot budgets: historical `l = 500`, and `l = ceil(n1 / t)`.
6. Write a CSV anyone can open in Excel.

This experiment does **not** need barcodes and does **not** retrain models.

## What we found

Every table in this project over-samples the same people when we keep 500 snapshots. Statlog L60 is the most extreme (R = 300: each of 300 people appears in about 300 snapshots). Even DCCCD L5, with thousands of defaulters, still has R ≈ 25.

The revised `l` values are tiny (2–21). That is the trade-off Experiment 28 studies: honest sample size vs a large training matrix.

## How to read the CSV

| Column | Meaning |
|--------|---------|
| `t` | people in one snapshot |
| `l` | how many snapshots |
| `naive_tl_over_n1` | reuse score R |
| `l_rule` | `historical_l500` vs `revised_ceil_n1_over_t` |
| `suggested_naive_near_or_below_1` | True only if R ≤ 1 |

Results: `6_Results/24_Sampling_Ratio_Audit/{Folder}/sampling_ratio_audit.csv`.
