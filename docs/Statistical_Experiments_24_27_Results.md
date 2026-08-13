# Statistical Experiments 24–27 — Actual Results (with Calculations)

Stage order (when we do what) is in `docs/Statistical_Approach_Flow.md`.
Why landmark percents and PCA rank differ is in `docs/Design_Decisions.md`.

Scripts live in `5_Experiments/24_*` … `27_*`. Formulas are in `utils.py`.

| Symbol | Meaning |
|--------|---------|
| `n1`, `n2` | Balanced class sizes (after undersampling, `n1 = n2`) |
| `t` | Points per snapshot |
| `l` | Number of snapshots per class |
| `R` | Reuse score `(t * l) / n1` |
| `b` | Intrinsic dimension |
| `L%` | Landmark percent (L10 = 10%) |

---

## Shared pipeline context (what “snapshots” are)

Experiment 3 built TDA features like this:

1. Balance classes by **undersampling** to the minority count → each class has size `n1 = n2`.
2. For a landmark percent `L%`, each snapshot draws `t = floor(n1 * L / 100)` points from one class.
3. Repeat that `l = 500` times **per class** → 500 + 500 = **1,000** barcode rows.
4. Ripser → H0 + H1 → **24 features** per snapshot (`g1_0`…`g12_0`, `g1_1`…`g12_1`).

Those 1,000 × 24 matrices are the inputs to Experiments 25 and 27. Experiments 24 and 26 do **not** need them.

Percents are dataset-specific on purpose: DCCCD L5/L15, Statlog L30/L60, new tables L10/L20. That is a size-of-`n1` decision, not a default we forgot to change.

---

## Experiment 24 — Are we over-reusing the same customers?

### Question

With `l = 500` snapshots of size `t`, how many times does a typical customer get drawn?

```text
R       = (t * l) / n1
l_star  = round(n1 / t)     ⇒  R_star ≈ 1
```

Target from the statistical checklist: `R ≈ 1` or less.

### Worked numbers (historical `l = 500`)

**Credit Card — L5**

```text
n1 = 6630
t  = floor(6630 * 0.05) = 331
R  = (331 * 500) / 6630 ≈ 25.0
l_star = round(6630 / 331) = 20
```

**PKDD — L10** (why L5 is impossible here)

```text
n1 = 76
t  = floor(76 * 0.10) = 7          # L5 would have been t = 3
R  = (7 * 500) / 76 ≈ 46.1
l_star = ceil(76 / 7) = 11
```

### Results table — all six datasets

| Dataset | Setting | `n1` | `t` | `l` | `R = (t*l)/n1` | OK? (`R ≤ 1`) |
|---------|---------|-----:|----:|----:|---------------:|---------------|
| Credit Card | L5 | 6,630 | 331 | 500 | **25.0** | No |
| Credit Card | L15 | 6,630 | 994 | 500 | **75.0** | No |
| Statlog German | L30 | 300 | 90 | 500 | **150.0** | No |
| Statlog German | L60 | 300 | 180 | 500 | **300.0** | No |
| PKDD Czech | L10 | 76 | 7 | 500 | **46.1** | No |
| PKDD Czech | L20 | 76 | 15 | 500 | **98.7** | No |
| Polish 3-year | L10 | 495 | 49 | 500 | **49.5** | No |
| Polish 3-year | L20 | 495 | 99 | 500 | **100.0** | No |
| Taiwan | L10 | 220 | 22 | 500 | **50.0** | No |
| Taiwan | L20 | 220 | 44 | 500 | **100.0** | No |
| South German | L10 | 300 | 30 | 500 | **50.0** | No |
| South German | L20 | 300 | 60 | 500 | **100.0** | No |

**Suggested snapshots if we keep `t` the same:**

| Dataset | Setting | Current `l` | Suggested `l_star` |
|---------|---------|------------:|-------------------:|
| Credit Card | L5 / L15 | 500 | **20 / 7** |
| Statlog | L30 / L60 | 500 | **3 / 2** |
| PKDD | L10 / L20 | 500 | **11 / 6** |
| Polish | L10 / L20 | 500 | **10 / 5** |
| Taiwan | L10 / L20 | 500 | **10 / 5** |
| South German | L10 / L20 | 500 | **10 / 5** |

None of the historical `l = 500` rows pass. Experiment 28 is the protocol that stops this.

Files: `6_Results/24_Sampling_Ratio_Audit/{Folder}/sampling_ratio_audit.csv`

---

## Experiment 25 — Average barcode feature values across snapshots

### Question

Across the 1,000 snapshots already produced by Exp 3, what are the mean and sample variance of each barcode statistic?

```text
x_bar  =  (1/N) * sum x_i
s^2    =  (1/(N-1)) * sum (x_i - x_bar)^2
```

The vector of 24 means is stored as `lambda_bar_proxy` — **not** a full persistence landscape.

### Highlights (Mean Death H0 = `g2_0`, Mean Persistence H1 = `g3_1`)

| Dataset / setting | `g2_0` mean | `g2_0` var | `g3_1` mean | `g3_1` var |
|-------------------|------------:|-----------:|------------:|-----------:|
| Credit Card L5 | 0.193 | 0.00013 | 0.023 | 3.3e-6 |
| Credit Card L15 | 0.144 | 0.00007 | 0.018 | 1e-6 |
| Statlog L30 | 1.121 | 0.00135 | 0.114 | 0.00015 |
| Statlog L60 | 0.996 | 0.00035 | 0.110 | 0.00005 |
| PKDD L10 | 1.657 | 0.0453 | 0.040 | 0.00554 |
| PKDD L20 | 1.384 | 0.0114 | 0.101 | 0.00339 |
| Polish L10 | 0.019 | 0.00016 | 0.00094 | 8.4e-7 |
| Polish L20 | 0.016 | 8.0e-5 | 0.00099 | 1.7e-7 |
| Taiwan L10 | 0.752 | 0.00341 | 0.070 | 0.00071 |
| Taiwan L20 | 0.648 | 0.00121 | 0.070 | 0.00022 |
| South German L10 | 0.973 | 0.00317 | 0.098 | 0.00059 |
| South German L20 | 0.850 | 0.00123 | 0.093 | 0.00021 |

PKDD variances are an order of magnitude larger than DCCCD — 7-point snapshots on a 76-person class wiggle. Polish H1 persistences are tiny (the cloud is very connected after PCA). Larger `L` generally **shrinks** variance: successive snapshots agree more, which is the stability we want, at the cost of even more reuse (Exp 24).

Files: `6_Results/25_Snapshot_Mean_Variance/{Folder}/snapshot_mean_variance.csv`

---

## Experiment 26 — How “high-dimensional” is the data?

### Question

What is the intrinsic dimension `b` **before** PCA (the credit table) and **after** the same PCA Exp 3 uses (the space Ripser samples)? We also record how many components would hit the ~90% variance target.

Hand-coded Two-NN (Facco) is the headline. scikit-dimension TwoNN / MLE / MiND_ML / lPCA are written to the same CSV so the hand-coded formula can be checked against the published package.

### Two-NN (headline)

```text
mu_i = r_i2 / r_i1
b_hat = 1 / mean(log mu_i)
```

### Results (Two-NN)

Numbers below are the Exp 26 re-run (hand-coded Two-NN + skdim suite). `n_components_for_90pct` is computed on the same scaled matrix.

| Dataset | Exp 3 PCA rank | Variance kept by that PCA | PCs to hit 90% | Two-NN **before** PCA | Two-NN **after** PCA |
|---------|---------------:|--------------------------:|---------------:|----------------------:|---------------------:|
| Credit Card | 7 | 94.0% | 6 | 3.95 | **2.81** |
| Statlog German | 15 | 89.3% | 16 | 5.34 | **4.06** |
| PKDD Czech | 10 | 89.8%* | 11 | 4.77 | **4.03** |
| Polish 3-year | 10 | 82.6% | 17 | 5.73 | **4.35** |
| Taiwan | 10 | 88.0% | 11 | 6.93 | **4.86** |
| South German | 10 | 79.1% | 14 | 6.37 | **4.19** |

\*Exp 26’s PKDD matrix is numeric-after-encoding. Exp 3’s dummy-heavy PH table kept only **~46.5%** with 10 components. Quote both; do not mix them.

The “PCs to hit 90%” column is why shared-10 is a *compromise*, not a 90% guarantee: Taiwan needed 11, Polish 17, South German 14. We did not re-rank Exp 3 after seeing this.

None of the after-PCA Two-NN values sit at 7. The “`b ≈ 7` trouble” flag does **not** fire.

**How to read before vs after:** before = geometry of the loan table; after = geometry Ripser sees. Snapshot-size theory should use the after column. Details: `docs/Design_Decisions.md` §3.

Levina–Bickel with k=10 came out much smaller than Two-NN on these tables. Treat Two-NN as the primary report number; LB / skdim MLE as a sensitivity check.

---

## Experiment 27 — Do default and non-default snapshots differ?

### Question

Robinson & Turner Algorithm 2 (arXiv:1310.7467) on **24-D barcode vectors** (proxy for full diagram distances). Cap 100 snapshots per class, `B = 200` permutations. Smallest possible p-value is `1/200 = 0.005`.

### Results

| Dataset | Setting | F_{2,2} p | F_{1,1} p | F_{2,1} p | Verdict |
|---------|---------|----------:|----------:|----------:|---------|
| Credit Card | L5 | 0.005 | 0.005 | 0.005 | Differ |
| Credit Card | L15 | 0.005 | 0.005 | 0.005 | Differ |
| Statlog | L30 | 0.005 | 0.005 | 0.005 | Differ |
| Statlog | L60 | 0.005 | 0.005 | 0.005 | Differ |
| PKDD | L10 | 0.17 | 0.23 | 0.26 | **Cannot reject** at 5% |
| PKDD | L20 | 0.01 | 0.005 | 0.01 | Differ |
| Polish | L10 | 0.005 | 0.005 | 0.005 | Differ |
| Polish | L20 | 0.005 | 0.005 | 0.005 | Differ |
| Taiwan | L10 | 0.84 | 0.54 | 0.85 | **Cannot reject** |
| Taiwan | L20 | 0.13 | 0.19 | 0.20 | **Cannot reject** at 5% |
| South German | L10 | 0.10 | 0.035 | 0.065 | Mixed (only F_{1,1} at 5%) |
| South German | L20 | 0.005 | 0.005 | 0.005 | Differ |

**How to read this with Exp 3 models:**

- Polish: classes differ under the proxy **and** XGBoost F1 is high (0.79 / 0.92) — but Exp 24 R = 49 / 100, so quote reuse next to F1.
- PKDD L10: Algorithm 2 agrees with the weak classifiers (k-NN acc 0.46). L20 starts to separate.
- Taiwan: models were near chance on L10 and only modest on L20; Algorithm 2 does **not** reject “same process”. That is a consistent story, not a contradiction.
- South German: L20 is where both the test and logistic F1 (0.76 vs 0.65) move.

**Caveat:** distances are on barcode-statistic **vectors**, not bottleneck/Wasserstein on raw diagrams.

Files: `6_Results/27_Null_Hypothesis_Algorithm2/{Folder}/algorithm2_permutation_results.csv`

---

## Bottom line

1. **500 snapshots is too many** on all six tables — `R` is 25×–300×; suggested `l_star` is 2–20.
2. **Barcode features are mostly stable**, except PKDD’s tiny snapshots (high variance). Larger `L` shrinks variance and raises reuse.
3. **Intrinsic dimension (Two-NN after PCA)** is 2.8–4.9 — below 7. Measure it **before and after** PCA; they answer different questions.
4. **Class clouds differ** on DCCCD, Statlog, Polish, and South German L20. They do **not** on PKDD L10 or Taiwan. Match that to the Exp 3 F1 numbers instead of averaging six datasets into one TDA-works claim.

### Doc fix vs older summary

Statlog L60 **people per class** is `n1 = 300` (same as L30). The value **180** is `t` (60% of 300), not the class size.
