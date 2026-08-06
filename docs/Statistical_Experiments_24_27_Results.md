# Statistical Experiments 24–27 — Actual Results (with Calculations)

Open the live tables canvas beside chat, or use the CSVs under `6_Results/`.

Scripts live in `5_Experiments/24_*` … `27_*`; formulas are implemented in `utils.py` (`compute_sampling_ratio_audit`, `summarize_snapshot_statistics`, `estimate_intrinsic_dimension_two_nn`, `permutation_test_algorithm2`).

---

## Shared pipeline context (what “snapshots” are)

Before Experiments 24–27, Experiment 3 built TDA features like this:

1. Balance classes by **undersampling** to the minority count → each class has size \(n_1 = n_2\).
2. For a landmark percent \(L\%\), each snapshot draws \(t = \lfloor n_1 \cdot L/100 \rfloor\) points from one class.
3. Repeat that \(l = 500\) times **per class** → 500 default + 500 non-default = **1,000** barcode rows.
4. On each snapshot, Ripser gives persistence diagrams \(H_0\) and \(H_1\); each diagram is turned into **12 numbers** → **24 features** per snapshot (`g1_0`…`g12_0`, `g1_1`…`g12_1`).

Those 1,000 × 24 matrices are the inputs to Experiments 25 and 27.

| Code | Meaning |
|------|---------|
| `g2_0` | Mean Death (Dim 0 / \(H_0\)) |
| `g3_1` | Mean Persistence (Dim 1 / \(H_1\)) |
| … | Full map in `utils.COLUMN_DESCRIPTIONS` |

---

## Experiment 24 — Are we over-reusing the same customers?

### Question
With \(l = 500\) snapshots of size \(t\), how many times does a typical customer get drawn, on average?

### Inputs (from processed Excel + Exp 3 practice)

| Symbol | Meaning | How we get it |
|--------|---------|---------------|
| \(n_{\mathrm{pos}}, n_{\mathrm{neg}}\) | Raw class counts | Count labels in `processed_data.xlsx` |
| \(n_1 = n_2\) | Balanced class size | \(\min(n_{\mathrm{pos}}, n_{\mathrm{neg}})\) |
| \(t\) | Points per snapshot | \(t = \lfloor n_1 \cdot L/100 \rfloor\) |
| \(l\) | Snapshots per class | Fixed at **500** in Exp 3 |

### Formulas (as coded in `compute_sampling_ratio_audit`)

\[
R = \frac{t \cdot l}{n_1}
\qquad\text{(reuse / “naive \(tl/n_1\)” score)}
\]

Target from the statistical checklist: \(R \approx 1\) or less.  
Suggested snapshot count keeping \(t\) fixed:

\[
l^\star = \mathrm{round}\!\left(\frac{n_1}{t}\right)
\quad\Rightarrow\quad R^\star \approx 1
\]

Also computed (same CSV): \(t/n\), \(t/n_1\), \((2tl)/n\), and pass/fail flags \(t/n_1 < 0.20\) and \(R \le 1\).

### Worked numbers

**Credit Card — L5**

- Minority (balanced) size: \(n_1 = 6{,}630\)
- \(t = \lfloor 6630 \times 0.05 \rfloor = 331\)
- \(R = (331 \times 500) / 6630 = 24.96 \approx \mathbf{25.0}\)
- \(l^\star = \mathrm{round}(6630/331) = \mathbf{20}\)

**Credit Card — L15**

- \(t = \lfloor 6630 \times 0.15 \rfloor = 994\)
- \(R = (994 \times 500) / 6630 = 74.96 \approx \mathbf{75.0}\)
- \(l^\star = \mathrm{round}(6630/994) = \mathbf{7}\)

**German Credit — L30**

- \(n_1 = 300\) (minority after balancing)
- \(t = \lfloor 300 \times 0.30 \rfloor = 90\)
- \(R = (90 \times 500) / 300 = \mathbf{150}\)
- \(l^\star = \mathrm{round}(300/90) = \mathbf{3}\)

**German Credit — L60**

- Same \(n_1 = 300\) (not 180 — 180 is \(t\), not class size)
- \(t = \lfloor 300 \times 0.60 \rfloor = 180\)
- \(R = (180 \times 500) / 300 = \mathbf{300}\)
- \(l^\star = \mathrm{round}(300/180) = \mathbf{2}\)

### Results table

| Dataset | Setting | People per class \(n_1\) | Points per snapshot \(t\) | Snapshots \(l\) | Reuse \(R=(tl)/n_1\) | OK? (\(R\le 1\)) |
|---------|---------|--------------------------|---------------------------|-----------------|----------------------:|------------------|
| Credit Card | L5 (5%) | 6,630 | 331 | 500 | **25.0** | No |
| Credit Card | L15 (15%) | 6,630 | 994 | 500 | **75.0** | No |
| German Credit | L30 (30%) | 300 | 90 | 500 | **150.0** | No |
| German Credit | L60 (60%) | 300 | 180 | 500 | **300.0** | No |

**Suggested snapshots if we keep \(t\) the same:**

| Dataset | Setting | Current \(l\) | Suggested \(l^\star\) |
|---------|---------|---------------|----------------------:|
| Credit Card | L5 | 500 | **20** |
| Credit Card | L15 | 500 | **7** |
| German Credit | L30 | 500 | **3** |
| German Credit | L60 | 500 | **2** |

Files: `6_Results/24_Sampling_Ratio_Audit/sampling_ratio_audit.csv`, `suggested_l_values.csv`  
Script: `5_Experiments/24_Sampling_Ratio_Audit/run_sampling_ratio_audit.py`

---

## Experiment 25 — Average barcode feature values across snapshots

### Question
Across the 1,000 snapshots already produced by Exp 3, what are the mean and sample variance of each barcode statistic? (Stability check; also a feature-space proxy for the landscape average \(\bar\lambda\).)

### Inputs
Existing matrices:

- `1_Data/TDA_Datasets/.../3_PH_Default_Parameters/data_L5.csv` (and L15 / L30 / L60)
- Shape: **1,000 rows × 24 features + `label`** (500 per class)

### How each row was originally built (reminder)
For one snapshot’s diagram in dimension \(d\):

\[
\begin{align*}
g1 &= \mathrm{mean}(\mathrm{birth}), &
g2 &= \mathrm{mean}(\mathrm{death}), &
g3 &= \mathrm{mean}(\mathrm{death}-\mathrm{birth}), \\
g4 &= \mathrm{mean}(y_{\max}-\mathrm{death}), &\ldots&&
\end{align*}
\]

(12 stats for \(H_0\), 12 for \(H_1\); see `compute_barcode_statistics`.)

### Formulas (as coded in `summarize_snapshot_statistics`)

For each feature column \(x\) over all \(N = 1000\) snapshots:

\[
\bar x = \frac{1}{N}\sum_{i=1}^{N} x_i,
\qquad
s^2 = \frac{1}{N-1}\sum_{i=1}^{N}(x_i - \bar x)^2
\quad\text{(pandas ``var(ddof=1)``)}
\]

The vector \((\bar x_1,\ldots,\bar x_{24})\) is stored as `lambda_bar_proxy` (barcode-statistic proxy for \(\bar\lambda\), not a full persistence landscape).

### Worked example — Credit Card L5, Mean Death \(H_0\) (`g2_0`)

From `snapshot_mean_variance.csv`:

- \(\bar x = 0.193416\ldots \approx 0.193\)
- \(s^2 = 0.00012827\ldots \approx 0.00013\)

Same file: Mean Persistence \(H_1\) (`g3_1`) → mean \(0.023027\), variance \(3.28\times 10^{-6}\).

### Highlights

| Dataset / setting | Feature (code) | Mean | Variance |
|-------------------|----------------|------|----------|
| Credit Card L5 | Mean Death \(H_0\) (`g2_0`) | 0.193 | 0.00013 |
| Credit Card L5 | Mean Persistence \(H_1\) (`g3_1`) | 0.023 | 0.000003 |
| Credit Card L15 | Mean Death \(H_0\) (`g2_0`) | 0.144 | 0.00007 |
| Credit Card L15 | Mean Persistence \(H_1\) (`g3_1`) | 0.018 | 0.000001 |
| German Credit L30 | Mean Death \(H_0\) (`g2_0`) | 1.121 | 0.00135 |
| German Credit L30 | Mean Persistence \(H_1\) (`g3_1`) | 0.114 | 0.00015 |
| German Credit L60 | Mean Death \(H_0\) (`g2_0`) | 0.996 | 0.00035 |
| German Credit L60 | Mean Persistence \(H_1\) (`g3_1`) | 0.110 | 0.00005 |

Full 96-row table (4 files × 24 features): `6_Results/25_Snapshot_Mean_Variance/snapshot_mean_variance.csv`  
Script: `5_Experiments/25_Snapshot_Mean_Variance/run_snapshot_mean_variance.py`

---

## Experiment 26 — How “high-dimensional” is the data?

### Question
What is the **intrinsic dimension** \(b\) of the point cloud (degrees of freedom), especially after the same PCA used in the TDA pipeline? Team concern: if \(b \approx 7\), subsample-size theory may be shaky. We estimate \(b\) rather than assuming PCA rank equals intrinsic dimension.

### Inputs / preprocessing (matches Exp 26 script)

1. Load processed tabular features; drop target / index columns.
2. `MinMaxScaler` → \(X_{\mathrm{scaled}}\).
3. Estimate \(b\) on \(X_{\mathrm{scaled}}\) (“raw”).
4. Fit `PCA(n_components=7)` (Credit Card) or `15` (German); transform → \(X_{\mathrm{PCA}}\).
5. Estimate \(b\) again on \(X_{\mathrm{PCA}}\).
6. Credit Card is subsampled to at most **5,000** rows (`random_state=42`) for cost; German uses all rows.

Variance retained = \(\sum_j \mathrm{explained\_variance\_ratio}_j\) of that PCA fit.

### Two-NN estimator (main number; Facco et al.)

For each point \(i\), let \(r_{i,1}\) and \(r_{i,2}\) be Euclidean distances to the 1st and 2nd nearest neighbours. Define

\[
\mu_i = \frac{r_{i,2}}{r_{i,1}} \quad (\mu_i > 1).
\]

MLE:

\[
\hat b_{\mathrm{Two\text{-}NN}} = \frac{1}{\frac{1}{m}\sum_{i=1}^{m} \log \mu_i}
= \frac{1}{\overline{\log\mu}}
\]

(`estimate_intrinsic_dimension_two_nn` in `utils.py`.)

### Levina–Bickel (also logged; secondary)

With \(k=10\) neighbours, local MLE at point \(i\):

\[
\hat b_i = \frac{1}{k-1}\sum_{j=1}^{k-1}\log\frac{r_{i,k}}{r_{i,j}},
\qquad
\hat b_{\mathrm{LB}} = \mathrm{mean}_i(\hat b_i).
\]

### Results (Two-NN is the headline)

| Dataset | Two-NN (scaled features) | Two-NN (after PCA) | PCA comps in TDA | Variance kept |
|---------|-------------------------:|-------------------:|-----------------:|--------------:|
| Credit Card | 3.95 | **2.81** | 7 | 94.0% |
| German Credit | 5.34 | **4.06** | 15 | 89.3% |

Neither PCA-space Two-NN is near 7 → the “\(b \approx 7\) trouble” flag does **not** fire for these estimates.

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
3. Cap each group at **100** snapshots (`MAX_PER_GROUP`, seed 42) → \(n_1 = n_2 = 100\).
4. Run \(B = 200\) permutations for each \((p,q) \in \{(2,2),(1,1),(2,1)\}\).

### Loss \(F_{p,q}\) (vector proxy)

Within a group \(G\) of size \(n\), using Minkowski distance \(d_p\):

\[
L(G) = \frac{1}{2n(n-1)}\sum_{i \neq j} d_p(x_i,x_j)^{q}
\]

\[
F_{p,q}(G_1,G_2) = L(G_1) + L(G_2)
\]

(`joint_loss_fpq_feature_vectors`.)  
Smaller \(F\) ⇒ points are **more tightly clustered within each label group**.

### Permutation p-value (Algorithm 2)

1. Compute \(F_{\mathrm{obs}} = F_{p,q}(G_1,G_2)\).
2. Pool the \(n_1+n_2\) vectors; randomly re-split into groups of the same sizes; recompute \(F\) for each of \(B-1\) shuffles (observed counted as one of \(B\)).
3. Count how often a shuffle is at least as extreme (here: \(F \le F_{\mathrm{obs}}\), because **smaller** \(F\) is more evidence of within-class structure):

\[
p = \frac{Z}{B},
\qquad
Z = \#\{\text{perms with } F \le F_{\mathrm{obs}}\}
\quad\text{(including the observed itself)}.
\]

With \(B = 200\), the smallest possible p-value is \(1/200 = \mathbf{0.005}\). Every reported cell hit that floor → observed \(F\) was the most extreme (smallest) among the 200.

### Worked sketch — Credit Card L5, \(F_{2,2}\)

- Observed: \(F_{\mathrm{obs}} = 0.0109\)
- Mean under shuffles: \(0.0131\) (null clouds looser / more mixed)
- \(p = 0.005\) → reject “same process”; classes **differ**

### Results table

| Dataset | Test | Observed \(F\) | Mean if labels shuffled | p-value | Verdict |
|---------|------|---------------:|------------------------:|--------:|---------|
| Credit Card L5 | \(F_{2,2}\) | 0.0109 | 0.0131 | 0.005 | Differ |
| Credit Card L5 | \(F_{1,1}\) | 0.2200 | 0.2888 | 0.005 | Differ |
| Credit Card L15 | \(F_{2,2}\) | 0.0069 | 0.0089 | 0.005 | Differ |
| Credit Card L15 | \(F_{1,1}\) | 0.1611 | 0.2364 | 0.005 | Differ |
| German Credit L30 | \(F_{2,2}\) | 0.1015 | 0.1139 | 0.005 | Differ |
| German Credit L30 | \(F_{1,1}\) | 1.0151 | 1.0662 | 0.005 | Differ |
| German Credit L60 | \(F_{2,2}\) | 0.0454 | 0.0560 | 0.005 | Differ |
| German Credit L60 | \(F_{1,1}\) | 0.6310 | 0.6995 | 0.005 | Differ |

**Caveat for publication:** distances are on barcode-statistic **vectors**, not bottleneck/Wasserstein on raw persistence diagrams. Same Algorithm 2 machinery; different ground metric.

File: `6_Results/27_Null_Hypothesis_Algorithm2/algorithm2_permutation_results.csv`  
Script: `5_Experiments/27_Null_Hypothesis_Algorithm2/run_algorithm2_nhst.py`

---

## Bottom line

1. **500 snapshots is too many** — reuse \(R = (tl)/n_1\) is 25×–300×; suggested \(l^\star = \mathrm{round}(n_1/t)\) is 2–20.  
2. **Barcode features are stable** — column-wise means over 1,000 snapshots; sample variances are small.  
3. **Intrinsic dimension (Two-NN)** on the PCA space used for TDA is **2.81** (Credit Card) and **4.06** (German) — below 7.  
4. **Default vs non-default snapshots differ** on every \(F_{p,q}\) permutation test we ran (\(p = 0.005 = 1/200\)).

### Doc fix vs older summary
German Credit L60 **people per class** is \(n_1 = 300\) (same as L30). The value **180** is \(t\) (60% of 300), not the class size.
