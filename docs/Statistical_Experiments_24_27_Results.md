# Statistical Experiments 24–27 — Actual Results (Plain English)

Open the live tables canvas beside chat, or use the CSVs under `6_Results/`.

---

## Experiment 24 — Are we over-reusing the same customers?

**What we measured:** For each class we take many random snapshots. Each snapshot has `t` people. We took `l = 500` snapshots. The **reuse score** `(t × l) / n₁` is roughly how many times a typical person appears across snapshots. Target: about **1 or less**. Ours are much higher.

| Dataset | Setting | People per class | Points per snapshot (t) | Snapshots (l) | Reuse score | OK? |
|---------|---------|------------------|-------------------------|---------------|-------------|-----|
| Credit Card | L5 (5%) | 6,630 | 331 | 500 | **25.0** | No |
| Credit Card | L15 (15%) | 6,630 | 994 | 500 | **75.0** | No |
| German Credit | L30 (30%) | 300 | 90 | 500 | **150.0** | No |
| German Credit | L60 (60%) | 180 | 180 | 500 | **300.0** | No |

**Suggested snapshots if we keep `t` the same:**

| Dataset | Setting | Current l | Suggested l |
|---------|---------|-----------|-------------|
| Credit Card | L5 | 500 | **20** |
| Credit Card | L15 | 500 | **7** |
| German Credit | L30 | 500 | **3** |
| German Credit | L60 | 500 | **2** |

Files: `6_Results/24_Sampling_Ratio_Audit/sampling_ratio_audit.csv`, `suggested_l_values.csv`

---

## Experiment 25 — Average barcode feature values across snapshots

**What we measured:** Each snapshot → 24 numbers. Across 1,000 snapshots we stored mean and variance of every feature.

**Highlights:**

| Dataset / setting | Feature | Mean | Variance |
|-------------------|---------|------|----------|
| Credit Card L5 | Mean Death (H₀) | 0.193 | 0.00013 |
| Credit Card L5 | Mean Persistence (H₁) | 0.023 | 0.000003 |
| Credit Card L15 | Mean Death (H₀) | 0.144 | 0.00007 |
| Credit Card L15 | Mean Persistence (H₁) | 0.018 | 0.000001 |
| German Credit L30 | Mean Death (H₀) | 1.121 | 0.00135 |
| German Credit L30 | Mean Persistence (H₁) | 0.114 | 0.00015 |
| German Credit L60 | Mean Death (H₀) | 0.996 | 0.00035 |
| German Credit L60 | Mean Persistence (H₁) | 0.110 | 0.00005 |

Full 96-row table: `6_Results/25_Snapshot_Mean_Variance/snapshot_mean_variance.csv`

---

## Experiment 26 — How “high-dimensional” is the data?

**What we measured:** Intrinsic dimension with Two-NN (main number). Roughly: how many degrees of freedom the cloud needs. Concern from the team: if this is ~7 we may be in trouble. It is **not**.

| Dataset | Two-NN (all features) | Two-NN (after PCA) | PCA components used in TDA | Variance kept |
|---------|----------------------:|-------------------:|---------------------------:|--------------:|
| Credit Card | 3.95 | **2.81** | 7 | 94.0% |
| German Credit | 5.34 | **4.06** | 15 | 89.3% |

File: `6_Results/26_Intrinsic_Dimension_Estimation/intrinsic_dimension_estimates.csv`

---

## Experiment 27 — Do default and non-default snapshots differ?

**What we measured:** Permutation test comparing default vs non-default snapshot clouds.  
**p = 0.005** means: under 200 random label shuffles, the real split looked more separated than almost all shuffles → classes look different.

| Dataset | Test | Observed score | Mean if labels shuffled | p-value | Verdict |
|---------|------|---------------:|------------------------:|--------:|---------|
| Credit Card L5 | F₂,₂ | 0.0109 | 0.0131 | 0.005 | Differ |
| Credit Card L5 | F₁,₁ | 0.2200 | 0.2888 | 0.005 | Differ |
| Credit Card L15 | F₂,₂ | 0.0069 | 0.0089 | 0.005 | Differ |
| Credit Card L15 | F₁,₁ | 0.1611 | 0.2364 | 0.005 | Differ |
| German Credit L30 | F₂,₂ | 0.1015 | 0.1139 | 0.005 | Differ |
| German Credit L30 | F₁,₁ | 1.0151 | 1.0662 | 0.005 | Differ |
| German Credit L60 | F₂,₂ | 0.0454 | 0.0560 | 0.005 | Differ |
| German Credit L60 | F₁,₁ | 0.6310 | 0.6995 | 0.005 | Differ |

File: `6_Results/27_Null_Hypothesis_Algorithm2/algorithm2_permutation_results.csv`

---

## Bottom line

1. **500 snapshots is too many** — reuse scores are 25×–300×; suggested `l` is 2–20.  
2. **Barcode features are stable** — means logged; variances small.  
3. **Intrinsic dimension (Two-NN) is below 7** on both datasets.  
4. **Default vs non-default snapshots differ** on every test we ran (p = 0.005).
