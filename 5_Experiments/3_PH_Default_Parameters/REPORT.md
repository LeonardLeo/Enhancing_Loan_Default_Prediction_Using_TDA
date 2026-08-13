# Experiment 3 — Turn each class into topological fingerprints, then classify

## In one sentence

We describe defaulters and non-defaulters by many small random subsets (landmarks), summarise the shape of each subset as 24 numbers, and ask whether those numbers predict default.

## Who this is for

You do not need to know algebraic topology. Think of a **barcode snapshot** as: pick a handful of people from one class, draw the “shape” of that handful, then squash the drawing into a row of numbers. We do that 500 times per class and train ordinary classifiers on the resulting table.

## Datasets

| Dataset | What it is | Rows (processed) | Landmark percents | PCA in this experiment |
|---------|------------|------------------|-------------------|------------------------|
| Default of Credit Card Clients (DCCCD) | Taiwan credit-card default next month | ~30,000 | L5 / L15 | 7 components |
| Statlog German Credit | UCI German loan applications | 1,000 | L30 / L60 | 15 components |
| PKDD'99 Czech Financial | Berka bank loans + pre-loan activity | 682 | L10 / L20 | 10 components |
| Polish Bankruptcy (3-year) | Firm financial ratios, 3-year horizon | 10,503 | L10 / L20 | 10 components |
| Taiwanese Bankruptcy | Company financials | 6,819 | L10 / L20 | 10 components |
| South German Credit | Updated German credit coding | 1,000 | L10 / L20 | 10 components |

## Why these knobs differ (read this before quoting L10/L20 or “90% variance”)

Full write-up: `docs/Design_Decisions.md`.

**Landmarks.** DCCCD uses L5/L15 because `n1 = 6,630`, so 5% is already `t = 331` people. Statlog uses L30/L60 because `n1 = 300`, so small percents would give clouds too tiny for H1. The four new tables share **L10/L20** so they stay comparable to *each other*. That is not a copy of either paper grid:

- Copying DCCCD’s 5% onto PKDD (`n1 = 76`) gives `t = 3` — persistent homology dies.
- Copying Statlog’s 30% onto Polish (`n1 = 495`) gives huge snapshots and worse reuse, and would confound “new dataset” with “different L”.
- South German has the same `n1 = 300` as Statlog but stays on L10/L20 because it is a **coding-sensitivity** table; mixing in 30/60 would confound coding with landmark size.
- L10 is the smallest shared percent that still gives PKDD `t = 7`. L20 is the 2× companion (same doubling as Statlog 30→60).

**PCA.** The *target* is ~90% variance (`DatasetConfig.pca_variance = 0.90`). DCCCD’s 7 components already keep ~94%; Statlog needs 15 to sit near 89%. The new tables share **10 components** so Ripser spaces are the same size; 10 was the rank that put Taiwan nearest 90% (~88%). Polish (~83%) and South German (~78%) miss the target. PKDD’s dummy-expanded PH table keeps only **~46.5%** — ten axes are not enough there. We document the miss rather than silently re-ranking (that would invalidate every downstream `data_L*.csv`). Experiment 13 matches variance instead of component count.

Only this experiment (and Exp 4 / 6 / 23 / 28) **builds** landmarks. Experiments 7, 8, 10, 11, 15, 17, 19, 21, 25, 27 **read** the `data_L*.csv` files this folder writes. They must not regenerate 500 Ripser jobs.

## What we do (in order)

1. Load the processed table (same starting file as Experiment 1).
2. Encode / fill missing values.
3. MinMax-scale **all rows**, then PCA. This is the historical protocol — it is slightly leaky because the scaler/PCA see the whole table. Experiments 23 and 28 split first.
4. Undersample the majority class so both classes have size `n1` (the minority count).
5. Draw `l = 500` random subsets per class, each of size `t = floor(n1 * L / 100)`.
6. Ripser computes H0 (connected pieces) and H1 (loops). Twelve barcode statistics per homology dimension give **24 numbers** per snapshot.
7. Stack 500 default + 500 non-default rows into Statlog-style files:
   - `1_Data/TDA_Datasets/{Folder}/3_PH_Default_Parameters/data_L{percent}.csv`
8. Train logistic regression, random forest, SVM, k-NN, and XGBoost with library defaults on an 80/20 split of those rows.

## Artefacts other experiments reuse

```
1_Data/Landmark_Sets/{Folder}/3_PH_Default_Parameters/
1_Data/Barcode_Statistics/{Folder}/3_PH_Default_Parameters/barcode_stats_*.csv
1_Data/TDA_Datasets/{Folder}/3_PH_Default_Parameters/data_L*.csv
6_Results/3_PH_Default_Parameters/{Folder}/model_results.pkl
```

## What we found (plain language)

### South German Credit (re-run, Statlog-style `data_L10.csv` / `data_L20.csv`)

PCA(10) kept **77.9%** of variance — weaker than Statlog’s 15-component setup. On the barcode table:

- **L10** (smaller snapshots): logistic regression was best, accuracy **0.65**, F1 **0.65**. The other models sat around 0.58–0.61. That is barely better than a coin flip on a balanced 200-row test set.
- **L20** (larger snapshots): logistic regression accuracy **0.76**, F1 **0.76**. SVM and XGBoost reached **0.73**. Bigger landmarks carried a clearer class signal.

### PKDD'99 Czech Financial (re-run)

PCA(10) on the dummy-expanded table kept only **~46.5%** of variance. That is a poor default: ten components are not enough for this encoding. Experiment 26, which estimates dimension on numeric columns only, kept **~90%** with the same 10 PCs — so the gap is the dummy/district columns, not “Czech loans are 10-dimensional”. Downstream k-NN on barcodes was weak (L10 accuracy **0.46**, L20 **0.59**). Treat PKDD Exp 3 metrics as “shape of a heavily compressed table”, not as a fair TDA vs tabular comparison until PCA rank is raised.

### Taiwanese Bankruptcy (re-run, Statlog-style `data_L10.csv` / `data_L20.csv`)

PCA(10) kept **88.0%** of variance. Each L10 snapshot has 22 points; each L20 snapshot has 44. Minority class after balancing is 220 firms, so 500 snapshots reuse the same bankrupt firms many times (see Experiment 24).

On the barcode table, L10 models sat around chance (best accuracy **0.53**, SVM F1 **0.57**). L20 was better: SVM accuracy **0.61**, F1 **0.63**; logistic F1 **0.62**. Larger snapshots again carried more signal, but the lift is modest.

### Polish Bankruptcy (re-run, Statlog-style `data_L10.csv` / `data_L20.csv`)

PCA(10) kept **82.6%** of variance. After balancing, `n1 = 495`; L10 snapshots have 49 points and L20 have 99.

Barcode classifiers were **stronger here than on PKDD or Taiwan**: L10 XGBoost accuracy **0.79**, F1 **0.79**; L20 XGBoost accuracy **0.92**, F1 **0.92**. That looks impressive until Experiment 24: historical `l = 500` gives R = 49 (L10) and R = 100 (L20). The 1,000 barcode rows are remixes of the same 495 firms. Quote the F1, then quote the reuse score. Experiment 28 is the protocol that stops this.

### DCCCD and Statlog

These already used the Statlog-style `data_L5.csv` / `data_L15.csv` and `data_L30.csv` / `data_L60.csv` layout. Do not rewrite them to L10/L20.

## Honest caveats

- 500 snapshots of size `t` from `n1` people **reuse** customers (Experiment 24). Historical `l = 500` is not an independent sample.
- PCA is fit on the full table (Experiments 23 / 28 split first).
- Accuracy on a **balanced barcode table** is not the same as accuracy on the original imbalanced customers.

## How to read a result without being a TDA expert

- **Accuracy** — share of snapshot-rows labelled correctly. Fine here because we forced a 50/50 table.
- **Precision** — of those we flagged as default, how many really were.
- **Recall** — of every default snapshot, how many we caught.
- **F1** — balance of precision and recall. Quote this number.
- **H0 / H1** — H0 is “how the points clump”; H1 is “whether they form loops”. Each contributes 12 summary numbers.

## How to re-run

```
python 5_Experiments/3_PH_Default_Parameters/<Folder>/<script>_PH.py
```

Set `PYTHONIOENCODING=utf-8` on Windows consoles.
