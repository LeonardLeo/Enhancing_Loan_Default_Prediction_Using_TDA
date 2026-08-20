# Design decisions — snapshots, PCA, and intrinsic dimension

This note answers three questions that a “snapshot size 10% / 20% of the class” sentence does **not** answer. English names are used throughout. Compact symbols from the snapshot-size methods literature are recorded once in `docs/Notation.md`.

1. Why 10% and 20% on the four new tables, instead of Default of Credit Card Client’s 5%/15% or Statlog’s 30%/60%?
2. Why PCA rank differs (7 vs 15 vs 10), and what the ~90% variance rule actually did?
3. Should intrinsic dimension be measured before PCA, after PCA, or both?

Canonical knobs live in `utils.DatasetConfig` (`pca_variance`, `landmark_percentages`, `notes["pca_n_components_exp3"]`, `notes["landmark_reason"]`).

Live code for the historical protocol is `5_Experiments/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/`. Intrinsic dimension is `5_Experiments/Statistics/1_Intrinsic_Dimension_Estimation/`. The revised protocol is arm experiment 9 in every TDA bucket.

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
| **PKDD, Polish, Taiwan, South German — L10 / L20** | Shared *new-table* grid. Explained below. | 76 / 495 / 220 / 300 | see next table |

### Why the new tables cannot copy Default of Credit Card Client

Copying 5% onto PKDD (minority class count = 76) gives floor(76 × 0.05) = 3 points per snapshot. Persistent homology on three points is not a credit-shape experiment; H1 is empty by construction. Five percent only works on Default of Credit Card Client because 5% of 6,630 is still 331 people.

### Why the new tables cannot copy Statlog

Copying 30% onto Polish (minority class count = 495) gives 148 points per snapshot and, with historical 500 snapshots, reuse ratio = (148 × 500) / 495 ≈ 150. Statlog *needs* 30% because 5–10% of 300 people is 15–30 points — tight for H1. Polish and Taiwan do **not** have that problem. Using Statlog percents on them would confound “new dataset” with “much larger snapshots + worse reuse”.

South German has the same minority class count (300) as Statlog. It still stays on L10/L20. That table exists as a **coding-sensitivity** check on German credit. Mixing in Statlog’s 30/60 would confound “updated coding” with “different snapshot size”.

### Why L10 and L20, specifically

Three constraints, in order:

1. **PKDD is the bottleneck.** Minority class count = 76. L10 gives 7 points per snapshot. That is the smallest shared percent that still produces a non-trivial Vietoris–Rips complex. Anything below that (L5) kills persistent homology on the smallest new table.
2. **The four new tables must be comparable to each other.** One shared relative size, not four ad-hoc percents. Otherwise a Polish vs Taiwan F1 gap could be “different percent” rather than “different geometry”.
3. **L20 is the 2× companion.** Statlog already uses a doubling (30 → 60). Default of Credit Card Client uses a tripling (5 → 15). The doubling is kept so “small vs large snapshot” is a clean factor on every new table.

Worked points per snapshot on the new grid:

| Dataset | Minority class count | L10 | L20 |
|---------|---------------------:|----:|----:|
| PKDD | 76 | 7 | 15 |
| Taiwan | 220 | 22 | 44 |
| South German | 300 | 30 | 60 |
| Polish | 495 | 49 | 99 |

L10/L20 is therefore a **compromise grid for comparability**, not a claim that 10% is universally optimal. Arm experiment 6 (sampling-ratio audit) then shows that historical 500 snapshots still over-reuse every table. Arm experiment 9 is where points per snapshot and number of snapshots are chosen from theory instead of from percents.

---

## 2. PCA — ~90% variance vs a shared component count

### The rule

`DatasetConfig.pca_variance = 0.90`. The design target for **new** tables is: keep enough principal components that the reduced cloud still holds about 90% of feature variance. That is a compression rule, not a claim about intrinsic dimension (Statistics experiment 1).

### Why the original two tables differ from each other

| Dataset | Historical Exp 1 rank | Variance kept | Why this rank |
|---------|----------------------:|--------------:|---------------|
| Default of Credit Card Client | **7** | ~94% | Original paper. Already above 90% at 7. |
| Statlog | **15** | ~89% | Original paper. 15 is what it takes on that table to sit near 90%. |

Matching **component count** across those two would *not* match the experiment: 7 axes on Statlog would starve it; 15 axes on Default of Credit Card Client would keep noise. Archived Experiment 13 exists specifically to match **variance**, not rank. Archived Experiments 16 and 18 sweep rank and ask whether F1 actually cares.

### Why the four new tables use 10 components, not “whichever n hits 90%”

The 90% rule is the *target*. The *implementation* in Historical arm experiment 1 is a **shared ambient dimension of 10**, so Ripser on PKDD, Polish, Taiwan, and South German runs in the same-sized box. Ten was the count that put **Taiwan nearest 90%** (~88%). The others were allowed to miss, and the miss is documented rather than silently re-ranking after barcodes already exist (re-ranking would invalidate every `data_L*.csv` consumer).

| Dataset | Historical Exp 1 rank | Variance kept | Hits ~90%? | PCs actually needed for 90% (Statistics Exp 1, this encoding) |
|---------|----------------------:|--------------:|------------|------------------------------------------------------:|
| Taiwan | 10 | ~88% | Near | **11** |
| Polish | 10 | ~83% | Short | **17** |
| South German | 10 | ~78–79% | Short | **14** |
| PKDD (Historical Exp 1 dummy-expanded PH table) | 10 | **~46.5%** | Miss | (different matrix than Statistics Exp 1) |
| PKDD (Statistics Exp 1 numeric-after-encoding) | 10 | ~89.8% | Near | **11** |

Default of Credit Card Client already overshoots 90% at **6** components (Historical Exp 1 kept 7 → 94%). Statlog needs **16** to cross 90% (Historical Exp 1 kept 15 → 89.3%).

### How to talk about this in the paper

- **Methods:** “New tables share 10 PCA axes so the Ripser spaces are comparable. The design target is ~90% variance; 10 was chosen from Taiwan. Tables that miss 90% are reported as such.”
- **Do not write:** “90% variance was kept on every new dataset.” That sentence is false for PKDD, Polish, and South German.
- **Sensitivity:** archived Exp 13 (match variance), archived Exp 16 / 18 (sweep rank), Statistics Exp 1 (`n_components_for_90pct` column).

---

## 3. Intrinsic dimension — before PCA *and* after PCA

PCA rank is an **ambient** dimension chosen for the pipeline. Intrinsic dimension is an **estimated** number of degrees of freedom. They are not interchangeable.

Both views are always reported:

| Estimate | What it measures | What it is for |
|----------|------------------|----------------|
| **Before PCA** (MinMax-scaled encoded table) | Geometry of the credit / bankruptcy features themselves. | “How many knobs is this dataset turning?” Independent of the PCA choice. |
| **After PCA** (the Historical Exp 1 box: 7 / 10 / 15 axes) | Geometry of the space **Ripser actually samples**. | Snapshot-size theory (points per snapshot vs intrinsic dimension). Snapshots are drawn in PCA coordinates, so this is the intrinsic dimension that theory should use. |

### What each one would hide if used alone

- **Only after PCA.** Intrinsic dimension cannot exceed the number of components kept. A drop from 6.9 → 4.9 after 10 PCs does **not** mean the original table was 4.9-dimensional; it means the *compressed* cloud is. It would also be impossible to see whether PCA flattened a genuinely high-d table or an already-low-d one.
- **Only before PCA.** Snapshots would be sized for a space Ripser never sees. If PCA has already collapsed the cloud, theory that uses raw-space intrinsic dimension will overstate how large points per snapshot needs to be.

### Headline estimator

Two-NN (Facco et al.), implemented two ways so a reviewer can check them:

- Hand-coded MLE in `utils.estimate_intrinsic_dimension_two_nn` (transparent formula).
- `skdim.id.TwoNN` from scikit-dimension (Bac et al., [arXiv:2109.02596](https://arxiv.org/abs/2109.02596)).

Secondary: Levina–Bickel (hand-coded and `skdim.id.MLE`), `MiND_ML`, `lPCA`. DANCo runs in arm experiment 9 on modest samples. dadapy is **not** added: it is the same Two-NN estimator under another dependency.

The “is intrinsic dimension ≈ 7?” alarm from the snapshot-size discussion is evaluated on **after-PCA Two-NN**. None of the six tables sit at 7 there.

Statistics experiment 1 writes both views, plus `n_components_for_90pct` (how many PCs would actually hit the 90% rule on that encoding). Artefacts: `6_Results/Statistics/1_Intrinsic_Dimension_Estimation/`.
