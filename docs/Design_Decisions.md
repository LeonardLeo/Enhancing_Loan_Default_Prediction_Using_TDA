# Design decisions — landmarks, PCA, and intrinsic dimension

This note answers three questions that a “we used L10/L20” sentence does **not** answer:

1. Why L10 and L20 on the four new tables, instead of DCCCD’s L5/L15 or Statlog’s L30/L60?
2. Why PCA rank differs (7 vs 15 vs 10), and what the ~90% variance rule actually did?
3. Should intrinsic dimension be measured before PCA, after PCA, or both?

Canonical knobs live in `utils.DatasetConfig` (`pca_variance`, `landmark_percentages`, `notes["pca_n_components_exp3"]`, `notes["landmark_reason"]`).

---

## 1. Landmark percents — why they differ

A landmark percent `L` is **not** a universal constant. It is a relative size:

```text
t = floor( n1 * L / 100 )
```

`n1` is the minority-class count after Experiment 3 undersamples the majority to match it. The same `L` produces wildly different snapshot sizes `t` on different tables.

| Dataset | Why these percents | `n1` | Resulting `t` |
|---------|--------------------|-----:|--------------:|
| **DCCCD — L5 / L15** | Original paper. The table is huge, so 5% is already a large cloud. | 6,630 | 331 / 994 |
| **Statlog — L30 / L60** | Original paper. The table is tiny, so large percents are required to get a usable cloud. | 300 | 90 / 180 |
| **PKDD, Polish, Taiwan, South German — L10 / L20** | Shared *new-table* grid. Explained below. | 76 / 495 / 220 / 300 | see next table |

### Why the new tables cannot copy DCCCD

Copying L5 onto PKDD (`n1 = 76`) gives `t = floor(76 * 0.05) = 3`. Persistent homology on three points is not a credit-shape experiment; H1 is empty by construction. DCCCD’s 5% only works because 5% of 6,630 is still 331 people.

### Why the new tables cannot copy Statlog

Copying L30 onto Polish (`n1 = 495`) gives `t = 148` and, with historical `l = 500`, reuse `R = (t * l) / n1 ≈ 150`. Statlog *needs* 30% because 5–10% of 300 people is 15–30 points — tight for H1. Polish and Taiwan do **not** have that problem. Using Statlog percents on them would confound “new dataset” with “much larger snapshots + worse reuse”.

South German has the same `n1 = 300` as Statlog. We still keep it on L10/L20. That table exists as a **coding-sensitivity** check on German credit. Mixing in Statlog’s 30/60 would confound “updated coding” with “different landmark size”.

### Why L10 and L20, specifically

Three constraints, in order:

1. **PKDD is the bottleneck.** `n1 = 76`. L10 gives `t = 7`. That is the smallest shared percent that still produces a non-trivial Vietoris–Rips complex. Anything below that (L5) kills PH on the smallest new table.
2. **The four new tables must be comparable to each other.** One shared relative size, not four ad-hoc percents. Otherwise a Polish vs Taiwan F1 gap could be “different L” rather than “different geometry”.
3. **L20 is the 2× companion.** Statlog already uses a doubling (30 → 60). DCCCD uses a tripling (5 → 15). We kept the doubling so “small vs large snapshot” is a clean factor on every new table.

Worked `t` on the new grid:

| Dataset | `n1` | L10 `t` | L20 `t` |
|---------|-----:|--------:|--------:|
| PKDD | 76 | 7 | 15 |
| Taiwan | 220 | 22 | 44 |
| South German | 300 | 30 | 60 |
| Polish | 495 | 49 | 99 |

L10/L20 is therefore a **compromise grid for comparability**, not a claim that 10% is universally optimal. Experiment 24 then shows that historical `l = 500` still over-reuses every table; Experiment 28 is where `t` and `l` are chosen from theory instead of from percents.

---

## 2. PCA — ~90% variance vs a shared component count

### The rule

`DatasetConfig.pca_variance = 0.90`. The design target for **new** tables is: keep enough principal components that the reduced cloud still holds about 90% of feature variance. That is a compression rule, not a claim about intrinsic dimension (Experiment 26).

### Why the original two tables differ from each other

| Dataset | Exp 3 rank | Variance kept | Why this rank |
|---------|-----------:|--------------:|---------------|
| DCCCD | **7** | ~94% | Original paper. Already above 90% at 7. |
| Statlog | **15** | ~89% | Original paper. 15 is what it takes on that table to sit near 90%. |

Matching **component count** across those two would *not* match the experiment: 7 axes on Statlog would starve it; 15 axes on DCCCD would keep noise. Experiment 13 exists specifically to match **variance**, not rank. Experiments 16 and 18 sweep rank and ask whether F1 actually cares.

### Why the four new tables use 10 components, not “whichever n hits 90%”

The 90% rule is the *target*. The *implementation* in Experiment 3 is a **shared ambient dimension of 10**, so Ripser on PKDD, Polish, Taiwan, and South German runs in the same-sized box. Ten was the count that put **Taiwan nearest 90%** (~88%). The others were allowed to miss, and we document the miss rather than silently re-ranking after barcodes already exist (re-ranking would invalidate every `data_L*.csv` consumer).

| Dataset | Exp 3 rank | Variance kept in Exp 3 | Hits ~90%? | PCs actually needed for 90% (Exp 26, this encoding) |
|---------|-----------:|-----------------------:|------------|------------------------------------------------------:|
| Taiwan | 10 | ~88% | Near | **11** |
| Polish | 10 | ~83% | Short | **17** |
| South German | 10 | ~78–79% | Short | **14** |
| PKDD (Exp 3 dummy-expanded PH table) | 10 | **~46.5%** | Miss | (different matrix than Exp 26) |
| PKDD (Exp 26 numeric-after-encoding) | 10 | ~89.8% | Near | **11** |

DCCCD already overshoots 90% at **6** components (Exp 3 kept 7 → 94%). Statlog needs **16** to cross 90% (Exp 3 kept 15 → 89.3%).

### How to talk about this in the paper

- **Methods:** “New tables share 10 PCA axes so the Ripser spaces are comparable. The design target is ~90% variance; 10 was chosen from Taiwan. Tables that miss 90% are reported as such.”
- **Do not write:** “We kept 90% variance on every new dataset.” That sentence is false for PKDD, Polish, and South German.
- **Sensitivity:** Exp 13 (match variance), Exp 16 / 18 (sweep rank), Exp 26 (`n_components_for_90pct` column).

---

## 3. Intrinsic dimension — before PCA *and* after PCA

PCA rank is an **ambient** dimension we chose. Intrinsic dimension `b` is an **estimated** number of degrees of freedom. They are not interchangeable.

We always report both:

| Estimate | What it measures | What it is for |
|----------|------------------|----------------|
| **Before PCA** (MinMax-scaled encoded table) | Geometry of the credit / bankruptcy features themselves. | “How many knobs is this dataset turning?” Independent of our PCA choice. |
| **After PCA** (the Exp 3 box: 7 / 10 / 15 axes) | Geometry of the space **Ripser actually samples**. | Snapshot-size theory (`t` vs `b`). Landmarks are drawn in PCA coordinates, so this is the `b` that theory should use. |

### What each one would hide if used alone

- **Only after PCA.** `b` cannot exceed the number of components we kept. A drop from 6.9 → 4.9 after 10 PCs does **not** mean the original table was 4.9-dimensional; it means the *compressed* cloud is. We would also be unable to see whether PCA flattened a genuinely high-d table or an already-low-d one.
- **Only before PCA.** We would be sizing snapshots for a space Ripser never sees. If PCA has already collapsed the cloud, theory that uses raw-space `b` will overstate how large `t` needs to be.

### Headline estimator

Two-NN (Facco et al.), implemented two ways so a reviewer can check them:

- Hand-coded MLE in `utils.estimate_intrinsic_dimension_two_nn` (transparent formula).
- `skdim.id.TwoNN` from scikit-dimension (Bac et al., [arXiv:2109.02596](https://arxiv.org/abs/2109.02596)).

Secondary: Levina–Bickel (hand-coded and `skdim.id.MLE`), `MiND_ML`, `lPCA`. DANCo runs in Experiment 28 on modest samples. We do **not** add dadapy: it is the same Two-NN estimator under another dependency.

The “is `b ≈ 7`?” alarm from the snapshot-size discussion is evaluated on **after-PCA Two-NN**. None of the six tables sit at 7 there.

Experiment 26 writes both views, plus `n_components_for_90pct` (how many PCs would actually hit the 90% rule on that encoding).
