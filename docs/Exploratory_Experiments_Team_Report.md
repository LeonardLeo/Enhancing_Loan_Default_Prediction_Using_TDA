# Exploratory Experiments — Team Report

**Project:** Enhancing Loan Default Prediction Using Topological Data Analysis  
**Audience:** Research team  
**Scope:** Exploratory experiments that are **not** in the main paper tables, but that shaped what is considered sensible for this project.  
**Excluded from this report (out of project scope):** Experiment 20 (deep learning placeholder); Experiment 5 Statlog `Full_Feature_Set` / `Balanced_Dataset` Mapper placeholders (`NotImplementedError`).  
**Layout note:** exploratory arms for the four registry datasets also live under the same mirrored `5_Experiments/{N}/{Folder}/` folders as DCCCD/Statlog.

---

## Purpose of this report

The paper-facing experiments (1–4, 6, 11–14, 19) answer predictive-performance questions. The exploratory set answers a different question: **what topological and geometric structure is present, and which modelling choices are defensible?**  

They form the basis for:

- choosing Mapper / visualisation parameters,
- deciding whether barcode features are redundant or geometrically separated,
- stress-testing PCA dimension and KNN neighbourhood size,
- motivating the statistical follow-ups (Experiments 24–27) and the early-split protocol (Experiment 23).

---

## Experiment map

| Exp | Folder | Role in the research narrative | Status |
|-----|--------|--------------------------------|--------|
| 5 | `5_Mapper` | Shape of **original** features via Kepler Mapper | DCCCD + registry scripts; Statlog Feature_Selection exists; other Statlog arms are placeholders |
| 7 | `7_EDA_Barcode_Statistics` | EDA of barcode matrices from Exp 3/6 | Complete artefacts in results (incl. registry folders) |
| 8 | `8_Dimensionality_Reduction_On_Barcode_Statistics` | PCA view of barcode space | Complete |
| 9 | `9_Dimensionality_Reduction_On_Original_Dataset` | PCA view of original processed data | Complete |
| 10 | `10_Covariance_Matrix_And_Distances` | Centroid / distance geometry of barcode clouds | Complete |
| 15 | `15_Working_With_K_in_KNN` | Sensitivity of KNN to `k` on barcodes | Complete |
| 16 | `16_Variance_Retained_..._Default_...` | PCA component sweep (DCCCD + registry) | Complete |
| 17 | `17_Distribution_For_Each_Class` | PCA / t-SNE / UMAP class separability | Complete |
| 18 | `18_Variance_Retained_..._Statlog_...` | PCA component sweep (SGCD + registry) | Complete |
| 21 | `21_Visualizing_Data_Shape_..._Using_TDA` | Mapper on **barcode** statistics | Complete (HTML outputs) |
| 22 | `22_Visualizing_Persistence_Diagrams` | Persistence diagrams per class | Complete — Statlog path fixed |

---

## Findings by experiment

### Experiment 5 — Mapper on original data (DCCCD)

**Goal.** Explore whether default vs non-default customers form connected structure under filter functions (PCA lens) and cover parameters (resolution, overlap, clustering).

**Method.** Kepler Mapper grids over resolution / overlap / KMeans `k`, producing interactive HTML graphs under `5_Experiments/Archives/5_Mapper/...` and mirrored results.

**Why it matters.** Supports the claim that topology is a *plausible* lens on credit data before committing to PH + ML. It does **not** produce a production classifier; it informs intuition and figure design.

**Caveat.** Large HTML volume; Statlog variants were not finished and are out of scope here.

---

### Experiment 7 — EDA on barcode statistics

**Goal.** Characterise the barcode-statistic feature space (H₀+H₁ and H₀-only) after Exp 3 / Exp 6.

**Method.** `eda(..., graphs=True)` on per-class and full barcode CSVs; stored as pickles under `6_Results/Archives/7_EDA_Barcode_Statistics/`.

**Why it matters.** Shows which barcode columns are skewed, sparse, or class-associated before we trust ML metrics. Feeds Experiment 11 (correlation dropping) conceptually.

---

### Experiments 8 & 9 — PCA on barcodes vs original data

**Goal.** Compare low-dimensional geometry of barcode features (Exp 8) with original processed features (Exp 9).

**Method.** `perform_pca_analysis` → scree plots, 2D scatters, loadings CSVs.

**Why it matters.** If classes separate more cleanly in barcode PCA than in original PCA (or vice versa), that guides interpretation of TDA gains. Also justifies later matched-variance experiments (13, 16, 18).

---

### Experiment 10 — Covariance and centroid distances

**Goal.** Probe whether barcode clouds for the two classes are geometrically distinguishable via mean / farthest / random centroids.

**Method.** Distance matrices and centroid-based summaries saved under `6_Results/Archives/10_Covariance_Matrix_And_Distances/` (L5/L15 and L30/L60).

**Why it matters.** Complements black-box ML scores with a geometric story: if centroids are far in barcode space, topological summaries carry class signal.

---

### Experiment 15 — K in KNN

**Goal.** Elbow / sensitivity analysis for KNN on barcode features (`k = 1…20`).

**Method.** `train_multiple_knn_datasets` + elbow curve plots.

**Why it matters.** Guards against over-interpreting a single default `k`. Informs whether neighbourhood methods are stable in barcode space.

---

### Experiments 16 & 18 — PCA component sweeps

**Goal.** Vary the number of PCA components used before PH and retrain, for DCCCD (16) and SGCD (18).

**Method.** `run_experiments_with_pca_components` over a grid (e.g. 2…19) with metric-vs-components plots.

**Why it matters.** PCA-7 / PCA-15 were modelling choices, not theorems. These sweeps show sensitivity of downstream accuracy to that choice and motivate Experiment 26 (estimate intrinsic dimension `b` properly).

---

### Experiment 17 — Class distribution visualisations

**Goal.** 2D/3D PCA, t-SNE, UMAP of barcode features with optional animations.

**Method.** `visualize_class_separability`.

**Why it matters.** Qualitative evidence for separability (or overlap) that numbers alone miss. Useful for slides and thesis figures.

---

### Experiment 21 — Mapper on barcode statistics

**Goal.** Apply Mapper to the **TDA feature space** rather than original attributes.

**Method.** `build_mapper_viz` with PCA/UMAP lenses on `data_L*` matrices.

**Why it matters.** Closes the loop: if Mapper structure appears in barcode space, our PH summaries preserve organised geometry—not only tabular signal.

---

### Experiment 22 — Persistence diagrams

**Goal.** Plot persistence diagrams for default vs non-default point clouds (Ripser + persim).

**Method.** Direct diagram visualisation (not barcode aggregation).

**Why it matters.** Pedagogical and diagnostic: confirms that PH is computing non-trivial structure. Statlog `viz.py` loads `Statlog_German_Credit_Data/processed_data.xlsx` with PCA(15).

---

## How these feed the next experiments

| Exploratory insight | Follow-up |
|---------------------|-----------|
| Full-data PCA/landmarks may leak | **Exp 23** early 80/20 + independent train/test snapshots |
| `l=500` and large `t` look aggressive | **Exp 24** sampling-ratio audit |
| Need mean/variance of snapshots for theory | **Exp 25** snapshot statistics / `\barλ` proxy |
| PCA dims ≠ intrinsic dimension | **Exp 26** Two-NN & Levina–Bickel |
| Need formal two-sample evidence | **Exp 27** Algorithm 2 / `F_{p,q}` (Robinson & Turner) |

---

## Recommended team actions

1. Read `docs/Pipeline_Issues_And_Leakage.md` alongside this report.  
2. Review CV numbers in `docs/CV_Results.md` (paper experiments).  
3. Review `docs/Experiment_23_Results.md` and `docs/Statistical_Experiments_24_27_Results.md`.  
4. For the fixed-`t` redesign across all six datasets, read `docs/Revised_Snapshot_Protocol_Deep_Report.md` (Exp 28).  
5. Cite Robinson & Turner (arXiv:1310.7467) and Chazal et al. (arXiv:1406.1901) in the write-up; reference Frontiers TDA survey §6.3.1 for snapshot statistics.

---

## References (for the statistical thread)

- Robinson, A. & Turner, K. *Hypothesis Testing for Topological Data Analysis*. arXiv:1310.7467 (Algorithm 2, loss `F_{p,q}`).  
- Chazal, F. et al. *Subsampling Methods for Persistent Homology*. arXiv:1406.1901 (`\barλ`, bias/variance of subsample landscapes).  
- Frontiers survey: *An Introduction to Topological Data Analysis* — Section 6.3.1 (snapshot mean/variance; predictive use of estimators).  
- Facco et al. — Two-NN intrinsic dimension.  
- Levina & Bickel — MLE intrinsic dimension.
