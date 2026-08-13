# Experiment 27 — Do default and non-default snapshots look like the same process?

## In one sentence

Robinson & Turner Algorithm 2 (arXiv:1310.7467) asks: if we shuffle the class labels, is the gap we see between default and non-default barcode rows still surprising?

## Who this is for

A classifier can get 70% accuracy on remixes of the same 300 people (Experiment 24) without the two classes being genuinely different processes. This test is a **sanity check**, not a proof that TDA will generalise to new customers.

**Limitation we always cite:** we apply the joint-loss F_{p,q} to 24-dimensional barcode **vectors**, not to bottleneck / Wasserstein distances between full persistence diagrams. Treat p-values as a tractable proxy.

## Datasets (all six)

Scripts live in each dataset folder, including **Statlog** and **DCCCD**:

`5_Experiments/27_Null_Hypothesis_Algorithm2/<Dataset>/run_algorithm2_nhst.py`

**Prerequisite:** Experiment 3 `data_L*.csv`.

## What we do (in order)

1. Load the Exp 3 matrix.
2. Split rows by label (1 = default / bankrupt, 0 = the other class).
3. Cap each group at 100 rows so 200 permutations stay affordable.
4. Run Algorithm 2 for (p, q) in {(2,2), (1,1), (2,1)}.
5. Write one CSV row per (file, p, q).

A **tiny p-value** means “these two clouds are probably not the same process”. It does **not** by itself mean a classifier will work on new people.

## What we found

### DCCCD (`data_L5.csv`, `data_L15.csv`)

All six tests gave p = 0.005 (the floor with 200 permutations). Default vs non-default barcode vectors are **not** exchangeable on this table.

### Statlog (`data_L30.csv`, `data_L60.csv`)

Same pattern: p = 0.005 on every (p, q) pair. The two classes’ barcode rows differ more than label-shuffling can explain.

### PKDD Czech Financial

- **L10:** p-values 0.17 / 0.23 / 0.26 — we **cannot** reject “same process” at 5%. Small landmarks on a 76-person class look noisy.
- **L20:** p-values 0.01 / 0.005 / 0.01 — larger snapshots **do** separate the classes under this proxy.

### South German Credit

- **L10:** mixed — F_{2,2} p = 0.10 (not significant at 5%), F_{1,1} p = 0.035. Small snapshots are only weakly different.
- **L20:** p = 0.005 on every pair. Larger snapshots separate the classes under this proxy, matching Experiment 3 (logistic F1 0.76 on L20 vs 0.65 on L10).

### Taiwanese Bankruptcy

- **L10:** p = 0.84 / 0.54 / 0.85 — we **cannot** reject “same process”. Matches Exp 3 (best accuracy ~0.53).
- **L20:** p = 0.13 / 0.19 / 0.20 — still not significant at 5%. SVM F1 ~0.63 is a weak lift on remixes of 220 firms (Exp 24 R = 100), not a class-cloud separation this test can see.

### Polish Bankruptcy (3-year)

p = 0.005 on every pair at L10 and L20. The two classes’ barcode rows differ more than label-shuffling can explain. That is consistent with Exp 3 XGBoost F1 0.79 / 0.92 — and must be quoted next to reuse R = 49 / 100.

## Where the files live

`6_Results/27_Null_Hypothesis_Algorithm2/{Folder}/algorithm2_permutation_results.csv`
