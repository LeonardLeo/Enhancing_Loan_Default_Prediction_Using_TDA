# Snapshot sample size (13/08/2026)

A top-level bucket — **not** inside Archives and **not** nested inside the four TDA protocol folders. The same grid is repeated on all four protocol arms.

This study uses English names: **points per snapshot** and **number of snapshots**. In the snapshot-size literature these quantities are sometimes written t and l; that mapping is recorded once in `docs/Notation.md`.

I report F1 as the headline metric because several tables are class-imbalanced; accuracy is shown as well.

## Where to read the method

Open these files — they show the pipeline in order:

| What to read | File |
|--------------|------|
| Shared-pool builder (this dataset) | `0_Shared_Pools/<Dataset>/<stem>_shared_pools.py` |
| Item 1 — number of snapshots, points per snapshot fixed | `1_Snapshot_Count_Sweep/<Dataset>/<stem>_sample_size.py` |
| Item 2 — points per snapshot, number of snapshots fixed at 60 | `2_Points_Per_Snapshot_Sweep/<Dataset>/<stem>_sample_size.py` |
| Item 4 — families of cloud size | `3_Snapshot_Count_Across_Cloud_Sizes/<Dataset>/<stem>_sample_size.py` |

Example for the Default of Credit Card Client table:

- `0_Shared_Pools/Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_shared_pools.py`
- `1_Snapshot_Count_Sweep/Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_sample_size.py`

`run.py` in those folders is an optional convenience launcher. Heavy Ripser / IO helpers live in `sample_size_lib.py`. `run_shared.py` / `build_shared_pools.py` run the same grid from the bucket root.

## Item 3 is not a third grid

This bucket implements items **1, 2, and 4**. **Item 3 is the sample-size study made of those three** — one compute grid, three figure families. There is no independent third sweep.

| Item | Folder | What it plots |
|------|--------|----------------|
| 1 | `1_Snapshot_Count_Sweep/` | Number of snapshots on x; points per snapshot **fixed** at the dataset-aware default (largest surviving 15/30/45/60 value). F1 headline, accuracy secondary. No cloud-size families. |
| 2 | `2_Points_Per_Snapshot_Sweep/` | Points per snapshot on x; number of snapshots **fixed at 60**. Dataset-aware grid (see below). |
| 3 | — | Absorbed: this whole bucket. Figure notes say so. |
| 4 | `3_Snapshot_Count_Across_Cloud_Sizes/` | Number of snapshots on x; **one curve per surviving points-per-snapshot value**. Accuracy and F1. SVM and Logistic focus panels; the other three classifiers as small multiples. 95% CI bands. |

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
- Caption: this CI is **snapshot-sampling uncertainty**, not customer-split uncertainty.
- This study does **not** also run five customer splits on the full grid (cost explosion). That limitation is intentional.

## Compute

For each `(dataset, protocol, points_per_snapshot, repeat)` generate the 60 training snapshots **once**, Ripser **once** per snapshot, then reuse those barcodes for every snapshot-count value. `skip_existing` is per-snapshot. Smaller datasets first: PKDD, South German, Statlog, Taiwan, Polish, DCCCD last.

PCA ranks: same as `DatasetConfig` / `docs/Design_Decisions.md` (historical Exp 3 ranks).

Classifiers: `svm`, `knn`, `xgb`, `logistic`, `random_forest` — Exp 1 TDA default hyperparameters. SVM and Logistic are highlighted (thicker, saturated); KNN, XGBoost, and Random Forest are visible but muted.

## Protocols

The dataset scripts loop these four arms:

- `Historical_Late_Split_Balanced_TDA` — full-table PCA, undersample, snapshot-level hold-out
- `Early_Split_TDA` — customers split first, PCA on train only, undersample inside each split
- `No_Undersampling` — full-table PCA, no majority downsample, snapshot-level hold-out
- `Early_Split_TDA_And_No_Undersampling` — customers split first, PCA on train only, no undersample

## Artefacts

```
5_Experiments/Snapshot_Sample_Size/
  sample_size_lib.py
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

## How to run

```powershell
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/run_shared.py --stage design
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/run_shared.py
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/1_Snapshot_Count_Sweep/visualize_results.py
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/2_Points_Per_Snapshot_Sweep/visualize_results.py
.\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/3_Snapshot_Count_Across_Cloud_Sizes/visualize_results.py
```

Resume-safe queue (does not interleave the existing Ripser/consumer shims):

```powershell
.\tda_env\Scripts\python.exe 6_Results\Run_Queue\_snapshot_sample_size_queue.py
```
