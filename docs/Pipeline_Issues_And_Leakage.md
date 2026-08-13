# Pipeline Issues and Data Leakage Analysis

Methodological risks in the codebase, what mitigates them, and what remains out of scope.

**Current layout:** every dataset uses mirrored folders  
`5_Experiments/{N}_{Name}/{Folder}/` ↔ `6_Results/{N}_{Name}/{Folder}/` ↔ `1_Data/.../{Folder}/`  
for all six datasets (`Default_Of_Credit_Card_Client_Data`, `Statlog_German_Credit_Data`, `PKDD_Czech_Financial`, `Polish_Bankruptcy_3Year`, `Taiwan_Bankruptcy`, `South_German_Credit`).

Shared helpers live in **`utils.py`** (same as Statlog / DCCCD). Registry raw→processed ingestion lives in **`1_Data/ingest_registry_datasets.py`**.

---

## 1. Data leakage (historical TDA pipeline)

### 1.1 PCA fit on the full dataset
**Where:** Legacy Experiments 3, 4, 6, 11–19 (DCCCD / Statlog scripts that call `generate_landmark_sets` on the full table).  
**Issue:** `MinMaxScaler` and `PCA` are fit on **all** rows of `processed_data.xlsx` / `processed_data.csv` before landmark generation. Hold-out information can influence the principal axes used for every snapshot.  
**Mitigation:**
- Experiment **23** fits scaler + PCA on the **train** split only (`stratified_early_split` + `fit_scaler_pca_on_train` in `utils.py`).
- Experiment **28** follows the revised early-split protocol for all six datasets.

### 1.2 Landmarks drawn from the full (balanced) pool
**Where:** `generate_landmark_sets` on class-balanced data built from the full table.  
**Issue:** Snapshots can mix future train and test customers before the late 80/20 split on barcode rows.  
**Mitigation:** Exp **23** / Exp **28** generate landmarks **independently** for train and test after the early split.

### 1.3 Late split only on barcode rows
**Where:** `train_dataset_tda` / `train_models_on_dataset` split `data_L*.csv` 80/20.  
**Issue:** Topological features were estimated from a mixed pool.  
**Mitigation:** Exp **23** / Exp **28** train only on train barcodes and evaluate only on test barcodes.

### 1.4 Cross-validation on barcode matrices
**Where:** `perform_cross_validation_tda` and legacy `*_CV.py` scripts.  
**Issue:** CV reuses barcode features from the leaking pipeline, so it estimates optimism of a **leaky** feature pipeline.  
**Status:** Documented in `docs/CV_Results.md`. Fully nested CV (split → PCA → landmarks → barcodes → model) remains future work.

### 1.5 Baseline ML (Exp 1–2)
**Where:** Feature selection (`SelectFpr`) and ADASYN after the train/test split on tabular data.  
**Status:** Generally safer than historical TDA; keep as the non-TDA reference. New-dataset Exp 1–2 scripts follow the same Statlog / DCCCD structure via `utils.py`.

### 1.6 Historical full-data PH (Exp 3+)
**Where:** Exp 3-style scripts that scale/PCA/balance on the full table before landmarks (including new-dataset ports that mirror Statlog).  
**Issue:** Same leakage class as legacy Statlog / DCCCD Exp 3.  
**Status:** Intentional comparability with the historical pipeline; publishable path is Exp **23** / Exp **28**.

---

## 2. Statistical validity issues (Zaniar checklist)

### 2.1 Sampling ratios too large
With `l = 500` and landmark fractions L5/L15 (DCCCD) or L30/L60 (SGCD), naive proportions `(t·l)/n₁` are typically **≫ 1**.  
**Experiment 24** audits `n`, `t`, `l` and suggests revised `l ≈ ceil(n₁/t)`. Registry results live under `6_Results/24_Sampling_Ratio_Audit/{Folder}/`.

### 2.2 Snapshot mean / variance
**Experiment 25** records mean/variance of barcode-statistic columns (vector proxy for `\barλ`). Artefacts: `6_Results/25_Snapshot_Mean_Variance/`.

### 2.3 Intrinsic dimension `b`
PCA component counts (7 / 15 / variance-driven) are **not** estimates of intrinsic dimension.  
**Experiment 26** estimates `b` via Two-NN and Levina–Bickel. Artefacts: `6_Results/26_Intrinsic_Dimension_Estimation/`.

### 2.4 Two-sample test on diagrams
**Experiment 27** implements Robinson & Turner Algorithm 2 with `F_{p,q}` on barcode-statistic vectors (proxy). Cite the paper; note the proxy when publishing.

### 2.5 Revised snapshot protocol
**Experiment 28** replaces percentage landmarks + `l=500` with fixed absolute `t`, default train/test `l = 60/15`, no undersampling, and reuse/overlap reporting for all six datasets. See `docs/Revised_Snapshot_Protocol_Deep_Report.md`.

---

## 3. Other pipeline / engineering issues

| Issue | Detail | Status |
|-------|--------|--------|
| Exp 20 folder name | Folder says “Deep Learning”; Statlog script is PH with `generate_landmark_sets_v2`, not TF/PyTorch | Documented; new-dataset ports mirror Statlog PH script |
| Exp 5 Statlog Mapper | `Full_Feature_Set` / `Balanced_Dataset` stubs | Explicit `NotImplementedError` placeholders; Feature_Selection arm exists |
| Exp 22 Statlog | Previously loaded DCCCD path | **Fixed** — uses Statlog `processed_data.xlsx`, `Class`, PCA(15) |
| Exp 9 Statlog | `perform_pca_analysis(..., target_column="label")` on data whose target is `Class` | Known mismatch in legacy Statlog script |
| Exp 11 | No `CV_results.pkl` in current results tree for some arms | Known gap |
| Metric naming | CV scripts wrote `mean_accracy` | **Fixed** in source → `mean_accuracy`; viz helpers still accept the old key when reading pickles |
| Relative paths | Legacy scripts expect cwd = experiment/dataset folder | Run from that folder; new scripts also insert repo root on `sys.path` |
| Landmark cost | 500 files × 2 classes is expensive | Prefer Exp 24 revised `l` / Exp 28 grids for new work |
| TDA path layout | New-dataset Exp 3+ now write Statlog-style `data_L10.csv` / `data_L20.csv` under `1_Data/TDA_Datasets/{Folder}/3_PH_Default_Parameters/` | Re-run Exp 3 after the rewrite; older `clean/L10/revised/` artefacts are from the retired parallel pipeline |

---

## 4. Recommended reading order

1. This document  
2. `docs/CV_Results.md`  
3. `docs/Exploratory_Experiments_Team_Report.md`  
4. `docs/Experiment_23_Results.md`  
5. `docs/Statistical_Experiments_24_27_Results.md`  
6. `docs/Revised_Snapshot_Protocol_Deep_Report.md` (Exp 28, all six datasets)
