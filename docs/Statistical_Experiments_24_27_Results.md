# Statistical Experiments 24–27 — Actual Results (with Calculations)

Open the live tables canvas beside chat, or use the CSVs under `6_Results/`.

Scripts live in `5_Experiments/24_*` … `27_*`; formulas are implemented in `utils.py` (`compute_sampling_ratio_audit`, `summarize_snapshot_statistics`, `estimate_intrinsic_dimension_two_nn`, `permutation_test_algorithm2`).

Notation used below:

| Symbol | Meaning |
|--------|---------|
| `n1`, `n2` | Balanced class sizes (after undersampling, `n1 = n2`) |
| `t` | Points per snapshot |
| `l` | Number of snapshots per class |
| `R` | Reuse score |
| `b` | Intrinsic dimension |
| `L%` | Landmark percent (e.g. L5 = 5%) |

---

## Shared pipeline context (what “snapshots” are)

Before Experiments 24–27, Experiment 3 built TDA features like this:

1. Balance classes by **undersampling** to the minority count → each class has size `n1 = n2`.
2. For a landmark percent `L%`, each snapshot draws

```text
t = floor( n1 * L / 100 )
```

points from one class.

3. Repeat that `l = 500` times **per class** → 500 default + 500 non-default = **1,000** barcode rows.
4. On each snapshot, Ripser gives persistence diagrams H0 and H1; each diagram is turned into **12 numbers** → **24 features** per snapshot (`g1_0`…`g12_0`, `g1_1`…`g12_1`).

Those 1,000 × 24 matrices are the inputs to Experiments 25 and 27.

| Code | Meaning |
|------|---------|
| `g2_0` | Mean Death (Dim 0 / H0) |
| `g3_1` | Mean Persistence (Dim 1 / H1) |
| … | Full map in `utils.COLUMN_DESCRIPTIONS` |

---

## Experiment 24 — Are we over-reusing the same customers?

### Question

With `l = 500` snapshots of size `t`, how many times does a typical customer get drawn, on average?

### Inputs (from processed Excel + Exp 3 practice)

| Symbol | Meaning | How we get it |
|--------|---------|---------------|
| `n_pos`, `n_neg` | Raw class counts | Count labels in `processed_data.xlsx` |
| `n1 = n2` | Balanced class size | `min(n_pos, n_neg)` |
| `t` | Points per snapshot | `t = floor(n1 * L / 100)` |
| `l` | Snapshots per class | Fixed at **500** in Exp 3 |

### Formulas (as coded in `compute_sampling_ratio_audit`)

```text
Reuse / “naive tl/n1” score:

    R  =  (t * l) / n1
```

Target from the statistical checklist: `R ≈ 1` or less.

Suggested snapshot count keeping `t` fixed:

```text
l_star  =  round( n1 / t )

⇒  R_star ≈ 1
```

Also computed (same CSV): `t/n`, `t/n1`, `(2 * t * l) / n`, and pass/fail flags `t/n1 < 0.20` and `R ≤ 1`.

### Worked numbers

**Credit Card — L5**

```text
n1 = 6630
t  = floor(6630 * 0.05) = 331
R  = (331 * 500) / 6630 = 24.96 ≈ 25.0
l_star = round(6630 / 331) = 20
```

**Credit Card — L15**

```text
t  = floor(6630 * 0.15) = 994
R  = (994 * 500) / 6630 = 74.96 ≈ 75.0
l_star = round(6630 / 994) = 7
```

**German Credit — L30**

```text
n1 = 300
t  = floor(300 * 0.30) = 90
R  = (90 * 500) / 300 = 150
l_star = round(300 / 90) = 3
```

**German Credit — L60**

```text
Same n1 = 300  (not 180 — 180 is t, not class size)
t  = floor(300 * 0.60) = 180
R  = (180 * 500) / 300 = 300
l_star = round(300 / 180) = 2
```

### Results table

| Dataset | Setting | People per class `n1` | Points per snapshot `t` | Snapshots `l` | Reuse `R = (t*l)/n1` | OK? (`R ≤ 1`) |
|---------|---------|----------------------:|------------------------:|--------------:|---------------------:|---------------|
| Credit Card | L5 (5%) | 6,630 | 331 | 500 | **25.0** | No |
| Credit Card | L15 (15%) | 6,630 | 994 | 500 | **75.0** | No |
| German Credit | L30 (30%) | 300 | 90 | 500 | **150.0** | No |
| German Credit | L60 (60%) | 300 | 180 | 500 | **300.0** | No |

**Suggested snapshots if we keep `t` the same:**

| Dataset | Setting | Current `l` | Suggested `l_star` |
|---------|---------|------------:|-------------------:|
| Credit Card | L5 | 500 | **20** |
| Credit Card | L15 | 500 | **7** |
| German Credit | L30 | 500 | **3** |
| German Credit | L60 | 500 | **2** |

Files: `6_Results/24_Sampling_Ratio_Audit/sampling_ratio_audit.csv`, `suggested_l_values.csv`  
Script: `5_Experiments/24_Sampling_Ratio_Audit/run_sampling_ratio_audit.py`

---

## Experiment 25 — Average barcode feature values across snapshots

### Question

Across the 1,000 snapshots already produced by Exp 3, what are the mean and sample variance of each barcode statistic? (Stability check; also a feature-space proxy for the landscape average.)

### Inputs

Existing matrices:

- `1_Data/TDA_Datasets/.../3_PH_Default_Parameters/data_L5.csv` (and L15 / L30 / L60)
- Shape: **1,000 rows × 24 features + `label`** (500 per class)

### How each row was originally built (reminder)

For one snapshot’s diagram in dimension `d`, twelve stats include:

```text
g1 = mean(birth)
g2 = mean(death)
g3 = mean(death - birth)          ← persistence
g4 = mean(y_max - death)
...
```

(12 stats for H0, 12 for H1; see `compute_barcode_statistics`.)

### Formulas (as coded in `summarize_snapshot_statistics`)

For each feature column `x` over all `N = 1000` snapshots:

```text
mean:

    x_bar  =  (1/N) * sum_{i=1..N} x_i

sample variance (pandas var(ddof=1)):

    s^2  =  (1/(N-1)) * sum_{i=1..N} (x_i - x_bar)^2
```

The vector `(x_bar_1, …, x_bar_24)` is stored as `lambda_bar_proxy` (barcode-statistic proxy for the landscape average — **not** a full persistence landscape).

### Worked example — Credit Card L5, Mean Death H0 (`g2_0`)

From `snapshot_mean_variance.csv`:

```text
x_bar ≈ 0.193
s^2   ≈ 0.00013
```

Same file: Mean Persistence H1 (`g3_1`) → mean `0.023027`, variance `3.28e-6`.

### Highlights

| Dataset / setting | Feature (code) | Mean | Variance |
|-------------------|----------------|------|----------|
| Credit Card L5 | Mean Death H0 (`g2_0`) | 0.193 | 0.00013 |
| Credit Card L5 | Mean Persistence H1 (`g3_1`) | 0.023 | 0.000003 |
| Credit Card L15 | Mean Death H0 (`g2_0`) | 0.144 | 0.00007 |
| Credit Card L15 | Mean Persistence H1 (`g3_1`) | 0.018 | 0.000001 |
| German Credit L30 | Mean Death H0 (`g2_0`) | 1.121 | 0.00135 |
| German Credit L30 | Mean Persistence H1 (`g3_1`) | 0.114 | 0.00015 |
| German Credit L60 | Mean Death H0 (`g2_0`) | 0.996 | 0.00035 |
| German Credit L60 | Mean Persistence H1 (`g3_1`) | 0.110 | 0.00005 |

Full 96-row table (4 files × 24 features): `6_Results/25_Snapshot_Mean_Variance/snapshot_mean_variance.csv`  
Script: `5_Experiments/25_Snapshot_Mean_Variance/run_snapshot_mean_variance.py`

---

## Experiment 26 — How “high-dimensional” is the data?

### Question

What is the **intrinsic dimension** `b` of the point cloud (degrees of freedom), especially after the same PCA used in the TDA pipeline? Team concern: if `b ≈ 7`, subsample-size theory may be shaky. We estimate `b` rather than assuming PCA rank equals intrinsic dimension.

### Inputs / preprocessing (matches Exp 26 script)

1. Load processed tabular features; drop target / index columns.
2. `MinMaxScaler` → `X_scaled`.
3. Estimate `b` on `X_scaled` (“raw”).
4. Fit `PCA(n_components=7)` (Credit Card) or `15` (German); transform → `X_PCA`.
5. Estimate `b` again on `X_PCA`.
6. Credit Card is subsampled to at most **5,000** rows (`random_state=42`) for cost; German uses all rows.

```text
Variance retained = sum of explained_variance_ratio of that PCA fit
```

### Two-NN estimator (main number; Facco et al.)

For each point `i`, let `r_i1` and `r_i2` be Euclidean distances to the 1st and 2nd nearest neighbours. Define:

```text
mu_i  =  r_i2 / r_i1     (keep only mu_i > 1)
```

MLE:

```text
b_hat_TwoNN  =  1 / mean( log(mu_i) )
             =  1 / (average of log mu)
```

(`estimate_intrinsic_dimension_two_nn` in `utils.py`.)

### Levina–Bickel (also logged; secondary)

With `k = 10` neighbours, local MLE at point `i`:

```text
b_hat_i  =  (1/(k-1)) * sum_{j=1..k-1}  log( r_i,k / r_i,j )

b_hat_LB =  mean over i of b_hat_i
```

### Results (Two-NN is the headline)

| Dataset | Two-NN (scaled features) | Two-NN (after PCA) | PCA comps in TDA | Variance kept |
|---------|-------------------------:|-------------------:|-----------------:|--------------:|
| Credit Card | 3.95 | **2.81** | 7 | 94.0% |
| German Credit | 5.34 | **4.06** | 15 | 89.3% |

Neither PCA-space Two-NN is near 7 → the “`b ≈ 7` trouble” flag does **not** fire for these estimates.

Also in CSV: Levina–Bickel raw / PCA (numerically much smaller here; treat Two-NN as the primary report number).

File: `6_Results/26_Intrinsic_Dimension_Estimation/intrinsic_dimension_estimates.csv`  
Script: `5_Experiments/26_Intrinsic_Dimension_Estimation/run_intrinsic_dimension.py`

---

## Experiment 27 — Do default and non-default snapshots differ?

### Question

Do the barcode-statistic clouds for default vs non-default look like they come from the **same** distribution? Formal two-sample test via Robinson & Turner Algorithm 2 (arXiv:1310.7467), applied to **24-D barcode vectors** (proxy for full diagram distances).

### Inputs / setup

1. Load the same Exp 3 CSVs as Exp 25.
2. Split rows: `label == 1` vs `label == 0` (feature columns only).
3. Cap each group at **100** snapshots (`MAX_PER_GROUP`, seed 42) → `n1 = n2 = 100`.
4. Run `B = 200` permutations for each `(p, q)` in `{(2,2), (1,1), (2,1)}`.

### Loss `F_p,q` (vector proxy)

Within a group `G` of size `n`, using Minkowski distance `d_p`:

```text
L(G) = (1 / (2 * n * (n-1))) * sum_{i ≠ j}  d_p(x_i, x_j)^q

F_p,q(G1, G2) = L(G1) + L(G2)
```

(`joint_loss_fpq_feature_vectors`.)  
Smaller `F` ⇒ points are **more tightly clustered within each label group**.

### Permutation p-value (Algorithm 2)

1. Compute `F_obs = F_p,q(G1, G2)`.
2. Pool the `n1 + n2` vectors; randomly re-split into groups of the same sizes; recompute `F` for each of `B-1` shuffles (observed counted as one of `B`).
3. Count how often a shuffle is at least as extreme (here: `F ≤ F_obs`, because **smaller** `F` is more evidence of within-class structure):

```text
p  =  Z / B

Z  =  number of perms with F ≤ F_obs
      (including the observed itself)
```

With `B = 200`, the smallest possible p-value is `1/200 = 0.005`. Every reported cell hit that floor → observed `F` was the most extreme (smallest) among the 200.

### Worked sketch — Credit Card L5, `F_2,2`

```text
F_obs              = 0.0109
Mean under shuffles = 0.0131   (null clouds looser / more mixed)
p                  = 0.005     → reject “same process”; classes differ
```

### Results table

| Dataset | Test | Observed `F` | Mean if labels shuffled | p-value | Verdict |
|---------|------|-------------:|------------------------:|--------:|---------|
| Credit Card L5 | F_2,2 | 0.0109 | 0.0131 | 0.005 | Differ |
| Credit Card L5 | F_1,1 | 0.2200 | 0.2888 | 0.005 | Differ |
| Credit Card L15 | F_2,2 | 0.0069 | 0.0089 | 0.005 | Differ |
| Credit Card L15 | F_1,1 | 0.1611 | 0.2364 | 0.005 | Differ |
| German Credit L30 | F_2,2 | 0.1015 | 0.1139 | 0.005 | Differ |
| German Credit L30 | F_1,1 | 1.0151 | 1.0662 | 0.005 | Differ |
| German Credit L60 | F_2,2 | 0.0454 | 0.0560 | 0.005 | Differ |
| German Credit L60 | F_1,1 | 0.6310 | 0.6995 | 0.005 | Differ |

**Caveat for publication:** distances are on barcode-statistic **vectors**, not bottleneck/Wasserstein on raw persistence diagrams. Same Algorithm 2 machinery; different ground metric.

File: `6_Results/27_Null_Hypothesis_Algorithm2/algorithm2_permutation_results.csv`  
Script: `5_Experiments/27_Null_Hypothesis_Algorithm2/run_algorithm2_nhst.py`

---

## Bottom line

1. **500 snapshots is too many** — reuse `R = (t * l) / n1` is 25×–300×; suggested `l_star = round(n1 / t)` is 2–20.  
2. **Barcode features are stable** — column-wise means over 1,000 snapshots; sample variances are small.  
3. **Intrinsic dimension (Two-NN)** on the PCA space used for TDA is **2.81** (Credit Card) and **4.06** (German) — below 7.  
4. **Default vs non-default snapshots differ** on every `F_p,q` permutation test we ran (`p = 0.005 = 1/200`).

### Doc fix vs older summary

German Credit L60 **people per class** is `n1 = 300` (same as L30). The value **180** is `t` (60% of 300), not the class size.
