# Pipeline Issues and Data Leakage Analysis

Methodological risks in the codebase, what mitigates them, and what remains out of scope.

**Current layout:** every dataset uses mirrored folders  
`5_Experiments/{Bucket}/{Experiment}/{Folder}/` ↔ `6_Results/{Bucket}/{Experiment}/{Folder}/` ↔ `1_Data/.../{ProtocolBucket}/{Experiment}/{Folder}/`  
for all six datasets (`Default_Of_Credit_Card_Client_Data`, `Statlog_German_Credit_Data`, `PKDD_Czech_Financial`, `Polish_Bankruptcy_3Year`, `Taiwan_Bankruptcy`, `South_German_Credit`). Paper LaTeX tables live in `6_Results/Paper_Tables/`.

Shared helpers live in **`utils.py`** (same as Statlog / DCCCD). Registry raw→processed ingestion lives in **`1_Data/ingest_registry_datasets.py`**.

---

## 1. Data leakage (historical TDA pipeline)

### 1.1 PCA fit on the full dataset
**Where:** Historical Late Split Balanced TDA experiments 1–5 and archived TDA scripts that call `generate_landmark_sets` on the full table.  
**Issue:** `MinMaxScaler` and `PCA` are fit on **all** rows of `processed_data.xlsx` / `processed_data.csv` before landmark generation. Hold-out information can influence the principal axes used for every snapshot.  
**Mitigation:**
- **Early split and undersample, using both H0 and H1** (Protocol B; historical Exp 23) fits scaler + PCA on the **train** split only (`stratified_early_split` + `fit_scaler_pca_on_train` in `utils.py`). Live path: `5_Experiments/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/`.
- **Arm experiment 9** (historical Exp 28) follows the revised protocol for all six datasets in every TDA arm.

### 1.2 Landmarks drawn from the full (balanced) pool
**Where:** `generate_landmark_sets` on class-balanced data built from the full table.  
**Issue:** Snapshots can mix future train and test customers before the late 80/20 split on barcode rows.  
**Mitigation:** Early Split TDA Exp 1 / arm Exp 9 generate landmarks **independently** for train and test after the early split.

### 1.3 Late split only on barcode rows
**Where:** `train_dataset_tda` / `train_models_on_dataset` split `data_L*.csv` 80/20.  
**Issue:** Topological features were estimated from a mixed pool.  
**Mitigation:** Early Split TDA Exp 1 / arm Exp 9 train only on train barcodes and evaluate only on test barcodes.

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
**Status:** Intentional comparability with the historical pipeline; publishable path is Early Split TDA Exp 1 / arm Exp 9.

---

## 2. Statistical validity issues (Zaniar checklist)

### 2.1 Sampling ratios too large
With 500 snapshots and snapshot-size percents 5%/15% (DCCCD) or 30%/60% (SGCD), naive reuse ratios (points per snapshot × number of snapshots) / minority class count are typically **≫ 1**.  
**Arm experiment 6** (historical Exp 24) audits class counts, points per snapshot, and number of snapshots, and suggests a revised snapshot count ≈ ceil(minority class count / points per snapshot). Registry results live under `6_Results/Late_Split_And_Undersample_H0_And_H1/6_Sampling_Ratio_Audit/{Folder}/`.

### 2.2 Snapshot mean / variance
**Arm experiment 7** (historical Exp 25) records mean/variance of barcode-statistic columns (vector proxy for `\barλ`). Artefacts: `6_Results/Archives/Four_Arm_Nested_Experiments/Historical_Late_Split_Balanced_TDA/7_Snapshot_Mean_Variance/`.

### 2.3 Intrinsic dimension
PCA component counts (7 / 15 / variance-driven) are **not** estimates of intrinsic dimension.  
**Statistics experiment 1** (historical Exp 26) estimates it via Two-NN and Levina–Bickel. Artefacts: `6_Results/Statistics/1_Intrinsic_Dimension_Estimation/`. There is no live `6_Results/26_Intrinsic_…` folder.

### 2.4 Two-sample test on diagrams
**Arm experiment 8** (historical Exp 27) implements Robinson & Turner Algorithm 2 with `F_{p,q}` on barcode-statistic vectors (proxy). Cite the paper; note the proxy when publishing.

### 2.5 Revised snapshot protocol
**Arm experiment 9** (historical Exp 28) replaces percentage landmarks + 500 snapshots with fixed points per snapshot, default 60 training / 15 test snapshots, and reuse/overlap reporting. It lives in the four H0-and-H1 process folders; the original (early split, no undersample, using both H0 and H1) is `5_Experiments/Early_Split_No_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/`. See `docs/Revised_Snapshot_Protocol_Deep_Report.md`. English names: `docs/Notation.md`.

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
| Landmark cost | 500 files × 2 classes is expensive | Prefer arm Exp 6 revised snapshot counts / arm Exp 9 grids for new work |
| TDA path layout | Protocol artefacts live at `1_Data/{TDA_Datasets\|Landmark_Sets\|Barcode_Statistics}/{ProtocolBucket}/{ExperimentName}/{Dataset}/` | Historical Exp 1 `data_L*.csv` reused; older `clean/L10/` trees are not the Design_Decisions PCA-rank protocol |

---

## 4. Recommended reading order

1. This document  
2. `docs/CV_Results.md`  
3. `docs/Exploratory_Experiments_Team_Report.md`  
4. `docs/Experiment_23_Results.md` (Early Split TDA Exp 1 / Protocol B hold-out numbers)  
5. `docs/Statistical_Experiments_24_27_Results.md`  
6. `docs/Revised_Snapshot_Protocol_Deep_Report.md` (arm Exp 9, all six datasets)  
7. `docs/Repository_Layout.md`
