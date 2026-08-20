# Snapshot sample size (13/08/2026)

A top-level bucket — **not** inside Archives and **not** nested inside the four TDA protocol folders. The same compute grid is repeated on all four protocol arms.

This study uses English names: **points per snapshot** and **number of snapshots**. In the snapshot-size literature these quantities are sometimes written t and l; that mapping is recorded once in `docs/Notation.md`.

I report F1 as the headline metric because several tables are class-imbalanced; accuracy is shown as well.

## Item 1 and item 2 are different x-factors

They share one Ripser cache. They are **not** the same sweep plotted twice.

| | Item 1 `1_Snapshot_Count_Sweep/` | Item 2 `2_Points_Per_Snapshot_Sweep/` |
|--|--|--|
| **x-axis** | Number of snapshots `{15, 30, 45, 60}` | Points per snapshot `{15, 30, 45, 60}` where they fit |
| **Held fixed** | Points per snapshot = dataset-aware default (`is_default_points_per_snapshot` is always 1) | Number of snapshots = **60** |
| **Figure title** | “Number of snapshots on the x-axis; each cloud has *N* points” | “Points per snapshot on the x-axis; always 60 snapshots” |
| **CSV** | `n_snapshots` moves; `points_per_snapshot` is one value per dataset×protocol | `points_per_snapshot` moves; `n_snapshots` is always 60 |

Folder `3_Snapshot_Count_Across_Cloud_Sizes/` is item 4: number of snapshots on x, **one curve per cloud size**. Do not treat it as a duplicate of item 1.

DCCCD Early Split example (logistic, mean accuracy across 10 repeats): item 1 holds clouds at **60 points** and moves snapshot count — 0.800 at 15 snapshots, 0.830 at 60 snapshots. Item 2 holds **60 snapshots** and moves cloud size — 0.647 at 15 points per snapshot, 0.830 at 60 points. The shared cell (60 points × 60 snapshots) matches; the rest of each grid does not.

## Where to read the method

To see how barcodes were built, open `0_Shared_Pools`. The numbered experiment scripts only select which rows go on which figure.

| What to read | File |
|--------------|------|
| **Method** — split, PCA, undersample, Ripser, train | the dataset script in each numbered Snapshot_Sample_Size folder |
| Item 1 slice — number of snapshots; default cloud size | `1_Snapshot_Count_Sweep/<Dataset>/<stem>_sample_size.py` |
| Item 2 slice — points per snapshot; always 60 snapshots | `2_Points_Per_Snapshot_Sweep/<Dataset>/<stem>_sample_size.py` |
| Item 4 slice — every surviving cell | `3_Snapshot_Count_Across_Cloud_Sizes/<Dataset>/<stem>_sample_size.py` |

Example for the Default of Credit Card Client table:

- **Method:** `0_Shared_Pools/Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_shared_pools.py`
- Item 2 slice: `2_Points_Per_Snapshot_Sweep/Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_sample_size.py`

`run.py` in those folders is an optional convenience launcher. Heavy Ripser / IO helpers live in `utils.py`.

## Item 3 is not a third grid

This bucket implements items **1, 2, and 4**. **Item 3 is the sample-size study made of those three** — one compute grid, three figure families. There is no independent third sweep.

| Item | Folder | What it plots |
|------|--------|----------------|
| 1 | `1_Snapshot_Count_Sweep/` | **Number of snapshots on the x-axis**; each cloud has the dataset-aware default number of points (largest surviving 15/30/45/60 value). F1 headline, accuracy secondary. No cloud-size families. Combined overlay = mean trend across 10 repeats (no error bars); companion `*_ci_panels.png` hold the 95% ribbons. |
| 2 | `2_Points_Per_Snapshot_Sweep/` | **Points per snapshot on the x-axis**; always 60 snapshots. Dataset-aware grid (see below). Same combined / CI-panel split as item 1. |
| 3 | — | Absorbed: this whole bucket. Figure notes say so. |
| 4 | `3_Snapshot_Count_Across_Cloud_Sizes/` | **Number of snapshots on the x-axis**; **one curve per surviving points-per-snapshot value**. Accuracy and F1. SVM and Logistic focus overlays; the other three classifiers as small multiples at full saturation. Combined overlays are mean trends only; companion panels use one CI ribbon per series. |

## Dataset-aware grid

Candidates: **15, 30, 45, 60**.

**Drop** any value that cannot be drawn without replacement from the protocol’s binding class pool after that arm’s split and optional undersample. Binding count:

- Early-split arms: `min(train minority, train majority, test minority, test majority)` because train and test snapshots come from different customer pools.
- Late-split arms: the smaller class on the full (possibly undersampled) PCA-transformed pool; train/test is a snapshot-level hold-out.

Never silent-clip. If every candidate would be dropped, a single clipped value of (class count − 1) is added and flagged in `dataset_aware_grid.csv` and on the figure note.

**Item 1 default points per snapshot:** the **largest surviving candidate**. This study does **not** substitute historical Exp 3 cloud sizes (331 on DCCCD, 90 on Statlog, 7 on PKDD). Those sit off this grid; item 4 already varies cloud size inside 15/30/45/60.

**Why a universal 15/30/45/60 cloud-size grid is not used:** PKDD’s class pool is tens of people; DCCCD’s is thousands. A step that is a small cloud on DCCCD can be the entire PKDD class. Item 2 / 4 footnotes say this.

Exact surviving values are written by `run_shared.py --stage design` to `6_Results/Snapshot_Sample_Size/shared/dataset_aware_grid.csv`.

Computed binding counts (`random_state=0`) and surviving points-per-snapshot values:

| Dataset | Late balanced / No undersampling (full class pool) | Early split (train and test pools; test is the bottleneck) |
|---------|-----------------------------------------------------|------------------------------------------------------------|
| PKDD | binding 76 → **15, 30, 45, 60** (default 60) | binding 15 → **14 only** (documented clip; every 15/30/45/60 candidate was ≥ 15) |
| South German | binding 300 → **15, 30, 45, 60** (default 60) | binding 60 → **15, 30, 45** (60 dropped; default 45) |
| Statlog | binding 300 → **15, 30, 45, 60** (default 60) | binding 60 → **15, 30, 45** (60 dropped; default 45) |
| Taiwan | binding 220 → **15, 30, 45, 60** (default 60) | binding 44 → **15, 30** (45 and 60 dropped; default 30) |
| Polish | binding 495 → **15, 30, 45, 60** (default 60) | binding 99 → **15, 30, 45, 60** (default 60) |
| DCCCD | binding 6630 → **15, 30, 45, 60** (default 60) | binding 1326 → **15, 30, 45, 60** (default 60) |

That table is why item 2’s figure footnote says a universal 15/30/45/60 cloud-size grid was not used.

## Confidence intervals

- One fixed customer train/test split (`random_state=0`). The CI does **not** mix “new customers” with “new snapshots”.
- Repeat **snapshot draws** 10 times: draw a pool of 60 training snapshots per class, shuffle, then train on nested prefixes 15 ⊂ 30 ⊂ 45 ⊂ 60.
- Hold out **15 test snapshots** drawn independently and kept fixed across the snapshot-count sweep.
- 95% CI = mean ± 1.96 × SE across the 10 repeats. A 2.5–97.5 percentile interval is also stored.
- Combined overlay plots are the **mean trend** across those 10 repeats (five models, no error bars). Companion `*_ci_panels.png` draw the same interval as a **ribbon**, one series per panel. There are no stacked translucent fill bands on a shared overlay.
- This study does **not** also run five customer splits on the full grid (cost explosion). That limitation is intentional.

## Compute

For each `(dataset, protocol, points_per_snapshot, repeat)` generate the 60 training snapshots **once**, Ripser **once** per snapshot, then reuse those barcodes for every snapshot-count value. `skip_existing` is per-snapshot. Smaller datasets first: PKDD, South German, Statlog, Taiwan, Polish, DCCCD last.

PCA ranks: same as `DatasetConfig` / `docs/Design_Decisions.md` (historical Exp 3 ranks).

Classifiers: `svm`, `knn`, `xgb`, `logistic`, `random_forest` — Exp 1 TDA default hyperparameters. Figures use a locked colourblind-safe mapping (Okabe–Ito: SVM blue, Logistic vermillion, KNN bluish green, XGBoost dark gold, Random Forest reddish purple). SVM and Logistic are thicker with filled circle markers; KNN / XGBoost / Random Forest stay at full saturation (square / triangle / diamond). Combined plots show mean trends only; companion panels show one 95% ribbon per series.

## Protocols

The dataset scripts loop these four arms:

- `Historical_Late_Split_Balanced_TDA` — full-table PCA, undersample, snapshot-level hold-out
- `Early_Split_TDA` — customers split first, PCA on train only, undersample inside each split
- `No_Undersampling` — full-table PCA, no majority downsample, snapshot-level hold-out
- `Early_Split_TDA_And_No_Undersampling` — customers split first, PCA on train only, no undersample

## Artefacts

```
5_Experiments/Snapshot_Sample_Size/
  utils.py
  build_shared_pools.py
  run_shared.py
  0_Shared_Pools/{Dataset}/
  1_Snapshot_Count_Sweep/   2_Points_Per_Snapshot_Sweep/   3_Snapshot_Count_Across_Cloud_Sizes/

1_Data/{Landmark_Sets,Barcode_Statistics,TDA_Datasets}/Snapshot_Sample_Size/{Protocol}/0_Shared_Pools/{Dataset}/
6_Results/Snapshot_Sample_Size/shared/{Protocol}/{Dataset}/
6_Results/Snapshot_Sample_Size/{Experiment}/{Protocol}/{Dataset}/
6_Results/Snapshot_Sample_Size/{Experiment}/Visualizations/
```

`0_Shared_Pools` is shared because items 1, 2, and 4 are views of one grid (item 3 is absorbed).

CSV one-liners:

- `1_Snapshot_Count_Sweep/all_summary.csv` — item 1: `n_snapshots` ∈ {15,30,45,60}; `points_per_snapshot` fixed at the default (`is_default_points_per_snapshot` always 1).
- `2_Points_Per_Snapshot_Sweep/all_summary.csv` — item 2: `points_per_snapshot` varies; `n_snapshots` always 60.
- `3_Snapshot_Count_Across_Cloud_Sizes/all_summary.csv` — item 4: full (`points_per_snapshot` × `n_snapshots`) family.

## How to run

```powershell
.\tda_env\Scripts\python.exe the dataset script in 5_Experiments/Snapshot_Sample_Size/1_Snapshot_Count_Sweep/<Dataset>/ --stage design
.\tda_env\Scripts\python.exe the dataset script in 5_Experiments/Snapshot_Sample_Size/1_Snapshot_Count_Sweep/<Dataset>/
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/1_Snapshot_Count_Sweep/visualize_results.py
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/2_Points_Per_Snapshot_Sweep/visualize_results.py
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/3_Snapshot_Count_Across_Cloud_Sizes/visualize_results.py
```

Resume-safe queue (does not interleave the existing Ripser/consumer shims):

```powershell
.\tda_env\Scripts\python.exe 6_Results\Run_Queue\_snapshot_sample_size_queue.py
```
