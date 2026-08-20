# 2_Points_Per_Snapshot_Sweep

Dated 13/08/2026. English wording throughout. The symbol mapping used in the methods literature is in `docs/Notation.md`.

## What this folder is

**Points per snapshot on the x-axis; always 60 snapshots.** Item 2 of the study.

This is **not** item 1 (`1_Snapshot_Count_Sweep`), which instead holds points per snapshot at the dataset-aware default and moves the number of snapshots. The two CSVs share the default-cloud × 60-snapshot cell and nothing else.

**Item 3 is not a third grid.** It is the sample-size study made of items 1, 2, and 4. All three consume the same Ripser cache.

## Design

- **Moves:** points per snapshot `{15, 30, 45, 60}` where they fit the protocol's binding class pool
- **Held fixed:** number of snapshots = **60**. In `all_summary.csv`, `n_snapshots` is always 60.
- Candidate points per snapshot are **dropped** when the value is at least the protocol's binding class count (no silent clipping). Early-split PKDD keeps only the documented clip of 14; early-split Taiwan keeps 15 and 30; early-split South German / Statlog keep 15, 30, 45.
- Headline metric: **F1** (imbalanced tables, especially with no undersampling). Accuracy is always plotted as well.
- One customer split (`random_state=0`). Ten snapshot-draw repeats. Nested prefixes 15 subset 30 subset 45 subset 60 from a shuffled pool of 60 training snapshots. Fifteen test snapshots drawn independently and held fixed across the snapshot-count sweep.
- 95% CI = mean ± 1.96 × SE across the 10 repeats (percentile interval also stored). This is snapshot-sampling uncertainty, not customer-split uncertainty. This study does not also run five customer splits on the full grid.
- Classifiers: SVM, KNN, XGBoost, Logistic Regression, Random Forest with Exp 1 TDA default hyperparameters. SVM and Logistic are thicker; KNN, XGBoost, and Random Forest use full-saturation Okabe–Ito colours (not muted). Combined overlay = mean trend across 10 repeats (no error bars); companion `*_ci_panels.png` use one CI ribbon per model.
- PCA ranks: same as `DatasetConfig` / `docs/Design_Decisions.md` (historical Exp 3 ranks).
- Early-split arms: split customers first, PCA on train only. Late-split arms: full-table PCA, then snapshot-level train/test.

CSV: `6_Results/Snapshot_Sample_Size/2_Points_Per_Snapshot_Sweep/all_summary.csv` — `points_per_snapshot` moves; `n_snapshots` is always 60.

## Where to read the method

Open the dataset script in this folder — for Default of Credit Card Client:

`Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_sample_size.py`

That file shows the protocol arm, customer split, PCA fit, how points-per-snapshot values are chosen (and which are dropped), the 60-snapshot draw, nested prefixes 15 ⊂ 30 ⊂ 45 ⊂ 60, the five classifiers, the 10 repeats, and how the CI is built. The shared-pool builder is `../0_Shared_Pools/<Dataset>/<stem>_shared_pools.py`. `run.py` is an optional convenience launcher.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/run_shared.py
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/2_Points_Per_Snapshot_Sweep/run.py --protocol Early_Split_TDA --datasets pkdd_czech
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/2_Points_Per_Snapshot_Sweep/visualize_results.py
```

Figures: `6_Results/Snapshot_Sample_Size/2_Points_Per_Snapshot_Sweep/Visualizations/`

Each graph has a methodology note underneath (what, why, nested prefixes, 10 repeats, F1 vs accuracy, dataset-aware grid). Combined titles say “Points per snapshot on the x-axis; always 60 snapshots”.
