# Four New Datasets — Deep Explanation Report

**Audience:** you, explaining the work to a supervisor or collaborator  
**Purpose:** not just *what* numbers appeared, but *what we did*, *how*, *why*, and *what the numbers mean*  
**Companion artefacts:** `6_Results/New_Datasets/*.csv`, canvas `four-new-datasets-results.canvas.tsx`, shorter table dump `New_Datasets_Final_Report.md`

Notation:

| Symbol | Meaning |
|--------|---------|
| `t` | points per snapshot |
| `l` | number of snapshots |
| `n_c` | size of one class pool |
| `R` | reuse score = `(t * l) / n_c` |
| `b` | intrinsic dimension |

---

## 1. What question were we answering?

We already had two working datasets:

1. **Default of Credit Card Clients (DCCCD)** — large Taiwan credit-card default table  
2. **Statlog German Credit (SGCD)** — classic UCI German credit data  

The team then pulled **four additional public credit / bankruptcy tables** so we could check whether the TDA pipeline behaves the same way outside those two originals:

| Key | Dataset | Rough size | Default / bad rate |
|-----|---------|------------|--------------------|
| `pkdd_czech` | PKDD’99 Czech financial (loan-level, pre-loan aggregates only) | ~682 loans | ~11% |
| `polish_bankruptcy` | Polish companies bankruptcy (3-year ARFF) | ~10.5k | ~4.7% |
| `taiwan_bankruptcy` | Taiwanese company bankruptcy | ~6.8k | ~3.2% |
| `south_german_credit` | South German Credit (sensitivity cousin of Statlog) | 1,000 | 30% |

**Scientific question:**  
If we build persistent-homology barcode features from random point-cloud snapshots of each class, do classifiers on those barcode rows beat (or even match) ordinary tabular baselines — and do the statistical red flags we found on DCCCD/SGCD (leakage, over-reuse of customers, intrinsic dimension) show up here too?

---

## 2. Two protocols — why both exist

### Protocol A — “Historical” (comparability, known leakage)

This mirrors the older paper pipeline:

1. Fit scaler / PCA (and related prep) in a way that can see the **full** table  
2. Build landmarks / barcodes  
3. Only then split for ML  

**Why keep it?** So numbers are comparable to earlier runs.  
**Why not trust it for claims?** Because train and test are not cleanly separated before topology is computed → **information leakage**.

### Protocol B — “Clean” (early split)

1. Stratified **80/20 split on the tabular rows first**  
2. Imputation, missing indicators, winsorization, encoding, scaling, PCA, any resampling: **fit on train only**  
3. Build train snapshots from train rows only; test snapshots from test rows only  
4. Train models on train barcodes; score on test barcodes  

**Why?** This is the scientifically defensible protocol. Experiment 23 on the original two datasets already showed that once you do this properly, “perfect” barcode accuracy often collapses toward chance — which is an important negative result, not a failure of the code.

---

## 3. Dataset-specific decisions (and why)

### PKDD Czech

- **Target:** accounts with loan status B/D = default; A/C = non-default  
- **Features:** only aggregates computable **before** the loan date (`transaction_date < loan_date`); undated standing orders excluded  
- **Why:** using post-loan behaviour to “predict” default would be cheating  

### Polish bankruptcy (3-year)

- Many missing cells in financial ratios  
- **Rule:** train-only **median imputation** + **missing-indicator** columns  
- **Why:** medians from the full data would leak; indicators let the model see “this field was missing,” which is often informative in financial statements  

### Taiwan bankruptcy

- Several features on a **~1e9** scale  
- Treated carefully in prep (scaling / winsorization path in the clean pipeline)  
- **Why:** unscaled billion-range columns dominate distances and PCA  

### South German Credit

- Treated as a **sensitivity** dataset relative to Statlog German  
- Label: bad → 1, good → 0  
- **Why:** same domain, different encoding / packaging; checks robustness of conclusions  

---

## 4. What is a “snapshot” in this project? (the TDA unit)

We do **not** feed one persistence diagram per customer into the classifier.

Instead, for each class (default / non-default):

1. Take the class’s feature points (after PCA) as a point cloud  
2. Draw a random subset of `t` points → one **landmark / snapshot**  
3. Repeat `l` times → `l` snapshots per class  
4. Run **Ripser** persistent homology on each snapshot  
5. Summarise each diagram with **12 barcode statistics × 2 homology dimensions (H0, H1) = 24 numbers**  
6. Each snapshot becomes **one training row** with those 24 numbers + a class label  

So the ML problem becomes:  
“Given the topology summary of a random cloud of defaulters vs non-defaulters, can we tell which class the cloud came from?”

That is a **different** question from “will this individual default?” — keep that distinction clear when presenting.

---

## 5. Two snapshot budgets we compared

### `historical500`

- `l = 500` snapshots per class (legacy habit from the original pipeline)  
- Landmark size still tied to a **percentage** of the (often balanced) class  

### `revised`

- `l` chosen so the **reuse score** is near or below 1  

### Reuse score (formula)

For a class with `n_c` customers, snapshot size `t`, snapshot count `l`:

```text
R_c  =  (t * l) / n_c
```

**Meaning:** if you glued all snapshot memberships together, how many times does a typical customer appear?

- `R_c ≈ 1`: each customer is used about once across the whole snapshot library → good  
- `R_c = 25` (seen on old DCCCD L5): each customer appears on the order of **25 times** → snapshots are highly dependent; effective sample size is much smaller than 500  

**Also checked:**

```text
t / n_c  <  0.20
```

so a single snapshot does not eat most of the class.

### What the audit found on the four new datasets

| Dataset | Variant | Typical reuse `R = (t * l) / n_c` | Verdict |
|---------|---------|----------------------------------|---------|
| All four | `historical500` | often ≫ 1 — e.g. South German ~**32**, PKDD test ~**12** | Over-reuse |
| All four | `revised` | train/test `R` brought near **≤ 1** | Meets the sampling checklist |

**How to say this in a meeting:**  
“500 was a historical convenience, not a statistically justified sample size. Once we enforce `R` near or below 1, we only keep on the order of **tens** of snapshots, not hundreds.”

---

## 6. Baseline results (tabular features — the honest yardstick)

Models: logistic, random forest, SVM, kNN, XGBoost.  
Primary metric for imbalance: **balanced accuracy** (average of recall on each class). We also track F1, ROC-AUC, average precision.

**Best baseline per dataset (by balanced accuracy):**

| Dataset | Protocol | Best model | Balanced acc | F1 | ROC-AUC |
|---------|----------|------------|-------------:|---:|--------:|
| Taiwan bankruptcy | historical | logistic | 0.87 | 0.32 | 0.93 |
| Taiwan bankruptcy | clean | logistic | 0.85 | 0.30 | 0.93 |
| Polish bankruptcy | clean | XGB | 0.81 | 0.44 | 0.91 |
| Polish bankruptcy | historical | XGB | 0.81 | 0.44 | 0.91 |
| South German | historical | SVM | 0.70 | 0.58 | 0.75 |
| South German | clean | SVM | 0.69 | 0.57 | 0.74 |
| PKDD Czech | clean / hist | XGB | 0.68 | 0.48 | ~0.81–0.82 |

**How to read this:**

- Taiwan/Polish look “easy” on ROC-AUC because negatives dominate; **F1 stays modest** — the model is not magically perfect at finding bankrupts.  
- Clean vs historical baselines are **close**, which is what we want: tabular baselines should not need leaky PCA.  
- These numbers are the bar TDA must clear **without** leakage tricks.

---

## 7. TDA barcode results — why many scores look “too good”

### Observed pattern

On `historical` protocol especially, many barcode runs hit **balanced accuracy ≈ 1.0**.  
On `clean` protocol, some revised runs still look extremely high (Taiwan revised clean mean bal. acc. ≈ 0.995), while South German clean sits near **0.47–0.49** (≈ chance on balanced snapshot labels).

### Why perfect scores are suspicious here

1. **Snapshot labels are assigned by construction**  
   Every snapshot drawn only from defaulters is labelled “default”. The classifier is not predicting an unknown individual’s label; it is separating two families of point-cloud summaries.

2. **If train and test clouds are built with leakage (Protocol A)**  
   PCA axes and/or shared pools make train and test barcode clouds artificially similar within class and separable across class.

3. **Tiny test barcode sets**  
   With revised `l` of order 5–11, a test matrix may have only ~10 rows. One lucky split + a flexible model → accuracy 1.0 without generalising.

4. **Class balance of snapshots ≠ class balance of customers**  
   We still emit the same number of snapshots per class, even when real defaults are 3–11%. That makes the barcode ML problem **class-balanced by construction**, unlike the tabular problem.

### Clean takeaway you can say out loud

> “High barcode accuracy under the historical protocol is **not** evidence that TDA predicts loan default for new customers. Under the clean protocol, results are mixed: some datasets still look separated at the cloud level, South German collapses toward chance, and tiny snapshot counts make variance huge. The sampling audit shows why we had to abandon `l = 500`.”

---

## 8. Statistical experiments on the new datasets (24–27 analogues)

### Intrinsic dimension `b` (Two-NN / Levina–Bickel)

**Why:** persistent homology and the email-style sample-size formula need a notion of how many degrees of freedom the cloud has. If `b` were enormous, topology estimates would need huge `t`.

**What we saw (Two-NN order of magnitude):**

| Dataset | Two-NN (approx.) |
|---------|------------------|
| PKDD | ~2.6 |
| Polish | ~3.9–4.4 |
| South German | ~7.9 |
| Taiwan | ~9.4 |

South German / Taiwan sit near or above the “~7 is worrying” informal threshold from the team chat; PKDD/Polish look milder.

### Mean / variance of barcode features

We logged global mean/variance of the 24 barcode stats across snapshots — a proxy for landscape stability (average barcode-feature vector across snapshots). Large variance ⇒ snapshots are noisy; tiny variance ⇒ summaries are stable (but stability ≠ predictive power for individuals).

### Robinson & Turner Algorithm 2 (permutation test on F_p,q)

**Idea:** are default-snapshot clouds and non-default-snapshot clouds exchangeable?

- Compute a joint loss `F_p,q` on barcode-statistic vectors (proxy for diagram distances)  
- Shuffle labels many times; see how often the shuffled loss is as small as the real one  
- Small p-value ⇒ classes look different in barcode space  

**Caveat to always mention:** we used **barcode-vector distances**, not true persistence-diagram metrics. It is a computational proxy.

Many settings gave `p ≈ 0.005` (minimum resolution with ~200 permutations). That supports “the two clouds differ,” **not** “we can deploy this as a credit score.”

---

## 9. Extended experiments (12 / 13 / 16 / 18 redesign; 20 excluded)

| Exp | Intent | What to remember |
|-----|--------|------------------|
| 12 | Match sample sizes / `t` | Isolates whether gains are just from snapshot size |
| 13 | Match PCA variance | Fairer comparison when component counts differ |
| 14 | 1:4 default:non-default snapshot mix | Stress test under imprint of real imbalance |
| 16 / 18 | PCA component sweeps | Sensitivity of tabular/TDA path to dimension |
| 19 | Linear regression on barcodes | Sanity / baseline linear probe |
| 2 | Tuned tabular baselines | Grid-searched reference performance |
| **20** | Deep learning | **Out of scope — excluded** |

When a redesign unit shows ROC-AUC ~1.0 on barcodes with handfuls of test rows, treat it as a **diagnostics flag**, not a paper headline.

---

## 10. Provenance / licence honesty

- Mirror checksums were recorded and largely matched  
- **Primary-source verification was not completed** for every host  
- Licence strings are inherited from the acquisition manifest  

Say this explicitly if asked about publication readiness.

---

## 11. How this connects to the *new* meeting protocol (Experiment 28)

The four-new-dataset suite still used:

- class balancing / percentage landmarks in places  
- `l = 500` vs a hastily revised small `l`  
- mixed clarity in the first markdown report  

The meeting then ordered a cleaner rule set (implemented as **Experiment 28**):

1. **No undersampling** — keep full class pools  
2. **Fixed absolute `t`** (same for train and test)  
3. Default **train `l = 60`**, **test `l = 15`**, plus a 3-point sweep in Zaniar’s ranges  
4. For **DCCCD** (the bigger dataset) non-split arm: `l` in `{60, 75, 90}`  
5. Separate **Concern A** (email formula) from **Concern B** (reuse):

```text
Concern A:   l ≈ (t / log(t))^(2/b)
Concern B:   R = (t * l) / n_c   (target R ≤ 1)
```

6. Measure **pairwise snapshot overlap**, reuse, **and** significance tests  

See: `docs/Revised_Snapshot_Protocol_Deep_Report.md` and `5_Experiments/28_Revised_Snapshot_Protocol/`.

---

## 12. One-page oral summary

1. We onboarded four public credit/bankruptcy sets with explicit prep rules.  
2. We ran leaky-historical and clean-early-split protocols.  
3. Tabular baselines are competent but not perfect (balanced acc roughly 0.68–0.87).  
4. `l = 500` massively reuses customers; revised `l` fixes reuse but shrinks the barcode sample.  
5. Barcode ML often looks unrealistically strong under leakage / tiny tests; South German clean is a useful counterexample near chance.  
6. Intrinsic dimensions vary (~2.6 to ~9.4); class clouds often differ by permutation test.  
7. The next methodology (Exp 28) removes undersampling, fixes `t`, and separates formula vs reuse math so we can defend every knob.

---

## File map

| Path | Contents |
|------|----------|
| `6_Results/New_Datasets/baseline_results.csv` | Tabular model metrics |
| `6_Results/New_Datasets/tda_results.csv` | Barcode model metrics |
| `6_Results/New_Datasets/sampling_ratio_audit.csv` | `t`, `l`, `R` audit |
| `6_Results/New_Datasets/statistical_results.csv` | ID + `F_p,q` tests |
| `6_Results/New_Datasets/experiment_coverage.csv` | What finished |
| `docs/new_datasets/New_Datasets_Final_Report.md` | Compact tables (less narrative) |
