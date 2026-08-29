# Design decisions — snapshots, PCA, and intrinsic dimension

This note answers three questions that a “snapshot size 5% / 15% or 30% / 60% of the class” sentence does **not** answer. English names are used throughout. Compact symbols from the snapshot-size methods literature are recorded once in `docs/Notation.md`.

1. Why Default of Credit Card Client uses 5%/15% while Statlog uses 30%/60%.
2. Why PCA rank differs (7 vs 15), and what the ~90% variance rule actually did.
3. Should intrinsic dimension be measured before PCA, after PCA, or both?

Canonical knobs live in `utils.DatasetConfig` (`pca_variance`, `landmark_percentages`, `notes["pca_n_components_exp3"]`, `notes["landmark_reason"]`).

Live code for the historical protocol is `5_Experiments/Late_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/`. Intrinsic dimension is `5_Experiments/Statistics/1_Intrinsic_Dimension_Estimation/`. The revised protocol is experiment 9 in every H0-and-H1 process folder.

The live tables are **Statlog German Credit** and **Default of Credit Card Client**.

---

## 1. Snapshot size as a percent of the class — why the percents differ

A snapshot-size percent is **not** a universal constant. It is a relative size:

```text
points per snapshot = floor( minority class count × percent / 100 )
```

Minority class count is taken after Historical arm experiment 1 undersamples the majority to match it. The same percent produces wildly different points per snapshot on different tables.

| Dataset | Why these percents | Minority class count | Points per snapshot |
|---------|--------------------|---------------------:|--------------------:|
| **Default of Credit Card Client — L5 / L15** | Original paper. The table is huge, so 5% is already a large cloud. | 6,630 | 331 / 994 |
| **Statlog — L30 / L60** | Original paper. The table is tiny, so large percents are required to get a usable cloud. | 300 | 90 / 180 |

Five percent only works on Default of Credit Card Client because 5% of 6,630 is still 331 people. Copying 5% onto Statlog (minority class count = 300) would give 15 points per snapshot — tight for H1.

Statlog *needs* 30%/60% because 5–10% of 300 people is 15–30 points. Default of Credit Card Client does **not** have that problem. Using Statlog percents on it would confound “dataset” with “much larger snapshots + worse reuse”.

Arm experiment 6 (sampling-ratio audit) then shows that historical 500 snapshots still over-reuse both tables. Arm experiment 9 is where points per snapshot and number of snapshots are chosen from theory instead of from percents.

---

## 2. PCA — ~90% variance vs dataset-specific rank

### The rule

`DatasetConfig.pca_variance = 0.90`. The design target is: keep enough principal components that the reduced cloud still holds about 90% of feature variance. That is a compression rule, not a claim about intrinsic dimension (Statistics experiment 1).

### Why the two tables differ from each other

| Dataset | Historical Exp 1 rank | Variance kept | Why this rank |
|---------|----------------------:|--------------:|---------------|
| Default of Credit Card Client | **7** | ~94% | Original paper. Already above 90% at 7. |
| Statlog | **15** | ~89% | Original paper. 15 is what it takes on that table to sit near 90%. |

Matching **component count** across those two would *not* match the experiment: 7 axes on Statlog would starve it; 15 axes on Default of Credit Card Client would keep noise. Archived Experiment 13 exists specifically to match **variance**, not rank. Archived Experiments 16 and 18 sweep rank and ask whether F1 actually cares.

Default of Credit Card Client already overshoots 90% at **6** components (Historical Exp 1 kept 7 → 94%). Statlog needs **16** to cross 90% (Historical Exp 1 kept 15 → 89.3%).

### How to talk about this in the paper

- **Methods:** “PCA rank is table-specific so each cloud sits near 90% variance. Default of Credit Card Client uses 7 axes; Statlog uses 15.”
- **Do not write:** “Both tables used the same number of principal components.”
- **Sensitivity:** archived Exp 13 (match variance), archived Exp 16 / 18 (sweep rank), Statistics Exp 1 (`n_components_for_90pct` column).

---

## 3. Intrinsic dimension — before PCA *and* after PCA

PCA rank is an **ambient** dimension chosen for the pipeline. Intrinsic dimension is an **estimated** number of degrees of freedom. They are not interchangeable.

Both views are always reported:

| Estimate | What it measures | What it is for |
|----------|------------------|----------------|
| **Before PCA** (MinMax-scaled encoded table) | Geometry of the credit features themselves. | “How many knobs is this dataset turning?” Independent of the PCA choice. |
| **After PCA** (the Historical Exp 1 box: 7 / 15 axes) | Geometry of the space **Ripser actually samples**. | Snapshot-size theory (points per snapshot vs intrinsic dimension). Snapshots are drawn in PCA coordinates, so this is the intrinsic dimension that theory should use. |

### What each one would hide if used alone

- **Only after PCA.** Intrinsic dimension cannot exceed the number of components kept. A drop after PCA does **not** mean the original table had that lower dimension; it means the *compressed* cloud does. It would also be impossible to see whether PCA flattened a genuinely high-d table or an already-low-d one.
- **Only before PCA.** Snapshots would be sized for a space Ripser never sees. If PCA has already collapsed the cloud, theory that uses raw-space intrinsic dimension will overstate how large points per snapshot needs to be.

### Headline estimator

Two-NN (Facco et al.), implemented two ways so a reviewer can check them:

- Hand-coded MLE in `utils.estimate_intrinsic_dimension_two_nn` (transparent formula).
- `skdim.id.TwoNN` from scikit-dimension (Bac et al., [arXiv:2109.02596](https://arxiv.org/abs/2109.02596)).

Secondary: Levina–Bickel (hand-coded and `skdim.id.MLE`), `MiND_ML`, `lPCA`. DANCo runs in arm experiment 9 on modest samples. dadapy is **not** added: it is the same Two-NN estimator under another dependency.

The “is intrinsic dimension ≈ 7?” alarm from the snapshot-size discussion is evaluated on **after-PCA Two-NN**.

Statistics experiment 1 writes both views, plus `n_components_for_90pct` (how many PCs would actually hit the 90% rule on that encoding). Artefacts: `6_Results/Statistics/1_Intrinsic_Dimension_Estimation/`.
