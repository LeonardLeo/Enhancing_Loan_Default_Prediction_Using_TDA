# Pipeline Issues and Data Leakage Analysis

This note summarises methodological risks in the current codebase and how new experiments address them.

## 1. Data leakage (historical TDA pipeline)

### 1.1 PCA fit on the full dataset
**Where:** Experiments 3, 4, 6, 11–19, etc.  
**Issue:** `MinMaxScaler` and `PCA` are fit on **all** rows of `processed_data.xlsx` before landmark generation. Hold-out information can influence the principal axes used for every snapshot.  
**Mitigation:** Experiment **23** fits scaler + PCA on the **train** split only, then transforms train and test.

### 1.2 Landmarks drawn from the full (balanced) pool
**Where:** `generate_landmark_sets` called on class-balanced data built from the full table.  
**Issue:** Snapshots used to build the barcode matrix can contain points that later appear (indirectly) in both train and test barcode rows after the late 80/20 split on `data_L*.csv`.  
**Mitigation:** Experiment **23** (Protocol B) generates landmarks **independently** for train and test after the early split.

### 1.3 Late split only on barcode rows
**Where:** `train_dataset_tda` / `train_models_on_dataset` split `data_L*.csv` 80/20.  
**Issue:** The topological feature distribution was estimated from a pool that mixed future train and test customers.  
**Mitigation:** Experiment **23** trains only on train barcodes and evaluates only on test barcodes (`train_dataset_tda_presplit`).

### 1.4 Cross-validation on barcode matrices
**Where:** `perform_cross_validation_tda` and `*_CV.py` scripts.  
**Issue:** CV reuses models and barcode features that were already produced under the leaking pipeline. CV therefore estimates optimism of a **leaky** feature pipeline, not a fully nested clean protocol.  
**Status:** Documented in `docs/CV_Results.md`. A fully nested CV (split → PCA → landmarks → barcodes → model) is computationally heavy and is out of scope for Exp 23; it remains a future item.

### 1.5 Baseline ML (Exp 1–2)
**Where:** Feature selection (`SelectFpr`) and ADASYN are applied after the train/test split on tabular data (good), but some scaling/feature choices should be double-checked for test contamination in each script.  
**Status:** Generally safer than historical TDA; keep as the non-TDA reference.

## 2. Statistical validity issues (Zaniar checklist)

### 2.1 Sampling ratios too large
With `l = 500` snapshots and landmark fractions L5/L15 (DCCCD) or L30/L60 (SGCD), the naive proportions `(t·l)/n₁` and `(t·2l)/n` are typically **≫ 1**.  
**Experiment 24** computes exact `n`, `t`, `l`, `t/n`, `t/n₁`, `(t·l)/n₁`, and suggests `l` values that bring `(t·l)/n₁ ≈ 1`.

### 2.2 Snapshot mean / variance not recorded for theory
The Frontiers survey §6.3.1 and Chazal et al. (arXiv:1406.1901) emphasise the empirical average landscape `\barλ` and snapshot variance.  
**Experiment 25** records mean/variance of barcode-statistic columns across snapshots and stores a **vector proxy** for `\barλ`. Full persistence-landscape `\barλ` remains optional (costly).

### 2.3 Intrinsic dimension `b` unknown
PCA component counts (7 / 15) are **not** estimates of intrinsic dimension. If `b ≈ 7`, subsample-size theory may be problematic.  
**Experiment 26** estimates `b` via Two-NN and Levina–Bickel on raw-scaled and PCA spaces.

### 2.4 No formal two-sample test on diagrams
**Experiment 27** implements Robinson & Turner Algorithm 2 (arXiv:1310.7467) with loss `F_{p,q}` on barcode-statistic vectors (proxy for diagram distances). Cite the paper; note the proxy when publishing.

## 3. Other pipeline / engineering issues

| Issue | Detail |
|-------|--------|
| Exp 20 | Deep learning placeholder; excluded from scope |
| Exp 5 Statlog Mapper | Empty stubs; excluded |
| Exp 22 Statlog | Possible wrong data path (DCCCD copy-paste) |
| Exp 11 | No `CV_results.pkl` in current results tree |
| Relative paths | Scripts must be run from their experiment directory (or with repo root on `PYTHONPATH`) |
| Landmark cost | 500 files × 2 classes × 2 splits (Exp 23) is expensive; plan runtime before full recompute |
| Metric naming | Some CV pickles use `mean_accracy` (typo) vs `mean_accuracy` |

## 4. Recommended reading order for the team

1. This document (leakage + stats gaps)  
2. `docs/CV_Results.md` (existing K-fold numbers)  
3. `docs/Exploratory_Experiments_Team_Report.md`  
4. Run Exp **24 → 26 → 25 → 27** (lightweight / medium)  
5. Schedule Exp **23** (heavy landmark regeneration)
