# 3_Snapshot_Count_Across_Cloud_Sizes

Dated 13/08/2026. English wording throughout. The symbol mapping used in the methods literature is in `docs/Notation.md`.

## What this folder is

**Number of snapshots on the x-axis; one curve per surviving points-per-snapshot value (families of cloud size).** Item 4 of the study.

This is **not** a duplicate of item 1. Item 1 holds cloud size at the dataset-aware default and draws one curve per classifier. Item 4 draws one curve per cloud size. Item 2 is the other 1-D slice (points per snapshot on x, always 180 snapshots).

**Item 3 is not a third grid.** It is the sample-size study made of items 1, 2, and 4. All three consume the same Ripser cache.

## Design

- **x-axis:** number of snapshots `{15, 30, 45, 60, 90, 120, 180}`
- **Families:** one curve per surviving points-per-snapshot value in `{15, 30, 45, 60, 90, 120, 180, 240, 330}` (dropped when the value is at least the protocol's binding class count; no silent clipping)
- Headline metric: **F1** (imbalanced tables, especially with no undersampling). Accuracy is always plotted as well.
- One customer split (`random_state=0`). Ten snapshot-draw repeats. Nested prefixes 15 ⊂ 30 ⊂ 45 ⊂ 60 ⊂ 90 ⊂ 120 ⊂ 180 from a shuffled pool of 180 training snapshots. Fifteen test snapshots drawn independently and held fixed across the snapshot-count sweep.
- 95% CI = mean ± 1.96 × SE across the 10 repeats (percentile interval also stored). This is snapshot-sampling uncertainty, not customer-split uncertainty. This study does not also run five customer splits on the full grid.
- Classifiers: SVM, KNN, XGBoost, Logistic Regression, Random Forest with Exp 1 TDA default hyperparameters. SVM and Logistic are thicker; KNN, XGBoost, and Random Forest use full-saturation Okabe–Ito colours (not muted). Combined overlays are mean trends only (no error bars); companion panels use one CI ribbon per (model, points-per-snapshot) cell.
- PCA ranks: same as `DatasetConfig` / `docs/Design_Decisions.md` (historical Exp 3 ranks).
- Early-split arms: split customers first, PCA on train only. Late-split arms: full-table PCA, then snapshot-level train/test.

CSV: `6_Results/Snapshot_Sample_Size/3_Snapshot_Count_Across_Cloud_Sizes/all_summary.csv` — full (`points_per_snapshot` × `n_snapshots`) family.

## Where to read the method

To see how barcodes were built, open `0_Shared_Pools`. The numbered experiment scripts only select which rows go on which figure.

Method (Default of Credit Card Client):

`../0_Shared_Pools/Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_shared_pools.py`

This folder’s `*_sample_size.py` files keep every surviving (points per snapshot × number of snapshots) cell so the figure can draw one curve per cloud size. 

## How to run

```
.\tda_env\Scripts\python.exe the dataset script in 5_Experiments/Snapshot_Sample_Size/1_Snapshot_Count_Sweep/<Dataset>/
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/3_Snapshot_Count_Across_Cloud_Sizes/run.py --protocol Early_Split_TDA --datasets statlog_german
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/3_Snapshot_Count_Across_Cloud_Sizes/visualize_results.py
```

Figures: `6_Results/Snapshot_Sample_Size/3_Snapshot_Count_Across_Cloud_Sizes/Visualizations/`

Each graph has a methodology note underneath (what, why, nested prefixes, 10 repeats, F1 vs accuracy, dataset-aware grid). Combined titles say “Number of snapshots on the x-axis; one curve per cloud size”.
