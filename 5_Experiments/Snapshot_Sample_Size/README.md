# Snapshot sample size (13/08/2026)

A top-level bucket — **not** inside Archives and **not** nested inside the four TDA protocol folders. The same compute grid is repeated on all four protocol arms.

This study uses English names: **points per snapshot** and **number of snapshots**. In the snapshot-size literature these quantities are sometimes written t and l; that mapping is recorded once in `docs/Notation.md`.

I report F1 as the headline metric because several tables are class-imbalanced; accuracy is shown as well.

## Item 1 and item 2 are different x-factors

They share one Ripser cache. They are **not** the same sweep plotted twice.

| | Item 1 `1_Snapshot_Count_Sweep/` | Item 2 `2_Points_Per_Snapshot_Sweep/` |
|--|--|--|
| **x-axis** | Number of snapshots `{15, 30, 45, 60, 90, 120, 180}` | Points per snapshot `{15, 30, 45, 60, 90, 120, 180, 240, 330}` where they fit |
| **Held fixed** | Points per snapshot = dataset-aware default (`is_default_points_per_snapshot` is always 1) | Number of snapshots = **180** |
| **Figure title** | “Number of snapshots on the x-axis; each cloud has *N* points” | “Points per snapshot on the x-axis; always 180 snapshots” |
| **CSV** | `n_snapshots` moves; `points_per_snapshot` is one value per dataset×protocol | `points_per_snapshot` moves; `n_snapshots` is always 180 |

Folder `3_Snapshot_Count_Across_Cloud_Sizes/` is item 4: number of snapshots on x, **one curve per cloud size**. Do not treat it as a duplicate of item 1.

DCCCD Early Split example (logistic, mean accuracy across 10 repeats on the previous 15–60 grid): item 1 held clouds at **60 points** and moved snapshot count — 0.800 at 15 snapshots, 0.830 at 60 snapshots. Item 2 held **60 snapshots** and moved cloud size — 0.647 at 15 points per snapshot, 0.830 at 60 points. The shared cell (60 points × 60 snapshots) matched. After the grid extension those two slices still share one cell, now at the dataset-aware default cloud × **180** snapshots.

## Where to read the method

To see how barcodes were built, open `0_Shared_Pools`. The numbered experiment scripts only select which rows go on which figure.

| What to read | File |
|--------------|------|
| **Method** — split, PCA, undersample, Ripser, train | the dataset script in each numbered Snapshot_Sample_Size folder |
| Item 1 slice — number of snapshots; default cloud size | `1_Snapshot_Count_Sweep/<Dataset>/<stem>_sample_size.py` |
| Item 2 slice — points per snapshot; always 180 snapshots | `2_Points_Per_Snapshot_Sweep/<Dataset>/<stem>_sample_size.py` |
| Item 4 slice — every surviving cell | `3_Snapshot_Count_Across_Cloud_Sizes/<Dataset>/<stem>_sample_size.py` |

Example for the Default of Credit Card Client table:

- **Method:** `0_Shared_Pools/Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_shared_pools.py`
- Item 2 slice: `2_Points_Per_Snapshot_Sweep/Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_sample_size.py`

`run.py` in those folders is an optional convenience launcher. Heavy Ripser / IO helpers live in `utils.py`.

## Item 3 is not a third grid

This bucket implements items **1, 2, and 4**. **Item 3 is the sample-size study made of those three** — one compute grid, three figure families. There is no independent third sweep.

| Item | Folder | What it plots |
|------|--------|----------------|
| 1 | `1_Snapshot_Count_Sweep/` | **Number of snapshots on the x-axis**; each cloud has the dataset-aware default number of points (largest surviving 15/30/45/60/90/120/180/240/330 value). F1 headline, accuracy secondary. No cloud-size families. Combined overlay = mean trend across 10 repeats (no error bars); companion `*_ci_panels.png` hold the 95% ribbons. |
| 2 | `2_Points_Per_Snapshot_Sweep/` | **Points per snapshot on the x-axis**; always 180 snapshots. Dataset-aware grid (see below). Same combined / CI-panel split as item 1. |
| 3 | — | Absorbed: this whole bucket. Figure notes say so. |
| 4 | `3_Snapshot_Count_Across_Cloud_Sizes/` | **Number of snapshots on the x-axis**; **one curve per surviving points-per-snapshot value**. Accuracy and F1. SVM and Logistic focus overlays; the other three classifiers as small multiples at full saturation. Combined overlays are mean trends only; companion panels use one CI ribbon per series. |

## Dataset-aware grid

Candidates: **15, 30, 45, 60, 90, 120, 180, 240, 330**. Locked in `utils.CANDIDATE_POINTS_PER_SNAPSHOT`. The extra steps after 60 move the large tables toward historical Exp 3 cloud sizes (Statlog 90/180, DCCCD L5 = 331). **330 is the DCCCD L5-scale ceiling.** DCCCD L15 (994 points) and the historical 500 snapshots stay off this study so the curves stay readable and reuse does not return to the original design.

**Drop** any value that cannot be drawn without replacement from the protocol’s binding class pool after that arm’s split and optional undersample. Binding count:

- Early-split arms: `min(train minority, train majority, test minority, test majority)` because train and test snapshots come from different customer pools.
- Late-split arms: the smaller class on the full (possibly undersampled) PCA-transformed pool; train/test is a snapshot-level hold-out.

Never silent-clip. If every candidate would be dropped, a single clipped value of (class count − 1) is added and flagged in `dataset_aware_grid.csv` and on the figure note.

**Item 1 default points per snapshot:** the **largest surviving candidate**. On DCCCD that is 330 (historical L5 was 331). On late-split Statlog that is 240 (330 is dropped against a class pool of 300). On early-split Statlog the test pool is the bottleneck, so 45 survives.

**Why a universal cloud-size grid is not used as-is:** Statlog’s class pool is hundreds of people; DCCCD’s is thousands. A step that is a small cloud on DCCCD can be most of Statlog’s class. Candidates that cannot be drawn are dropped. Item 2 / 4 footnotes say this.

Exact surviving values are written by `write_master_design_table()` to `6_Results/Snapshot_Sample_Size/shared/dataset_aware_grid.csv`.

Computed binding counts (`random_state=0`) and surviving points-per-snapshot values:

| Dataset | Late balanced / No undersampling (full class pool) | Early split (train and test pools; test is the bottleneck) |
|---------|-----------------------------------------------------|------------------------------------------------------------|
| Statlog | binding 300 → **15 … 240** (330 dropped; default 240) | binding 60 → **15, 30, 45** (60+ dropped; default 45) |
| DCCCD | binding 6630 → **15 … 330** (default 330) | binding 1326 → **15 … 330** (default 330) |

That table is why item 2’s figure footnote says a universal cloud-size grid is not used unchanged.

## Confidence intervals

- One fixed customer train/test split (`random_state=0`). The CI does **not** mix “new customers” with “new snapshots”.
- Repeat **snapshot draws** 10 times: draw a pool of 180 training snapshots per class, shuffle, then train on nested prefixes 15 ⊂ 30 ⊂ 45 ⊂ 60 ⊂ 90 ⊂ 120 ⊂ 180.
- Hold out **15 test snapshots** drawn independently and kept fixed across the snapshot-count sweep. That hold-out is the scoring set. If test size moved with training size, F1 would mix “more training barcodes” with “a different test set,” and the curve would not isolate the x-axis factor. Reuse is still scored on the **training** pool (points × training snapshots / minority count); test size does not enter that ratio.
- Reuse flags: `reuse_flags.csv` plus orange shading on F1/accuracy plots, companion `*_reuse_*.png` curves, and (item 4) a reuse heatmap. Reuse > 1 is marked on both axes — the snapshot counts that push reuse over 1, and the cloud sizes that do the same.
- 95% CI = mean ± 1.96 × SE across the 10 repeats. A 2.5–97.5 percentile interval is also stored.
- Combined overlay plots are the **mean trend** across those 10 repeats (five models, no error bars). Companion `*_ci_panels.png` draw the same interval as a **ribbon**, one series per panel. There are no stacked translucent fill bands on a shared overlay.
- This study does **not** also run five customer splits on the full grid (cost explosion). That limitation is intentional.

## Compute

For each `(dataset, protocol, points_per_snapshot, repeat)` generate the 180 training snapshots **once**, Ripser **once** per snapshot, then reuse those barcodes for every snapshot-count value. Existing 60-snapshot caches are extended (first 60 barcodes stay; nested 15 ⊂ … ⊂ 60 stay). `skip_existing` is per-snapshot. Statlog first, DCCCD last.

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

- `1_Snapshot_Count_Sweep/all_summary.csv` — item 1: `n_snapshots` ∈ {15,30,45,60,90,120,180}; `points_per_snapshot` fixed at the default (`is_default_points_per_snapshot` always 1).
- `2_Points_Per_Snapshot_Sweep/all_summary.csv` — item 2: `points_per_snapshot` varies; `n_snapshots` always 180.
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
