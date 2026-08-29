# Revised snapshot protocol — arm experiment 9 (the four H0-and-H1 process folders)
## Deep Explanation Report (What / How / Why / Formulas / Calculations)

**Status:** design + ML complete for the canonical arm (early split, no undersample, using both H0 and H1) under `6_Results/Early_Split_No_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/`. The same Exp 9 engine is cloned into the other three TDA arms (`Late_Split_And_Undersample_H0_And_H1`, `Early_Split_And_Undersample_H0_And_H1`, `Late_Split_No_Undersample_H0_And_H1`) with that arm's split/undersample knobs.  
**Canonical code:** `5_Experiments/Early_Split_No_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/`  
**Package for intrinsic dimension:** [scikit-dimension](https://pypi.org/project/scikit-dimension/) (`TwoNN`, `MLE`, `lPCA`, optional `DANCo`).

English names are used throughout. Compact symbols from the methods literature are recorded once in `docs/Notation.md`.

| English name | Meaning |
|--------|---------|
| points per snapshot | fixed absolute count of customers in one Vietoris–Rips cloud |
| number of snapshots | how many snapshots are drawn per class |
| intrinsic dimension | Two-NN primary |
| class pool size | number of rows in one class pool |
| reuse ratio | (points per snapshot × number of snapshots) / class pool size |

---

## 0. How to use this document

1. **§1** — what changed vs the old pipeline  
2. **§2–3** — two *separate* mathematical concerns (formula vs reuse)  
3. **§4** — worked numbers for **DCCCD** (the bigger dataset)  
4. **§5** — overlap + significance tests  
5. **§6** — grids we ran (60/15 + Zaniar sweeps + DCCCD 60–90)  
6. **§7** — ML results  
7. **§8** — what each result means (and what it does not)

---

## 1. What the meeting changed (old → new)

| Old habit | New rule | Why |
|-----------|----------|-----|
| Undersample majority to minority before landmarks | **No class balancing** — keep full class pools | Undersampling throws away majority geometry |
| points per snapshot = percentage of (balanced) class | **Fixed absolute points per snapshot** (same for train and test) | Percentage couples snapshot size to class size/balancing |
| 500 snapshots | Default **60 training snapshots**, **15 test snapshots** | 500 caused extreme reuse (arm experiment 6) |
| — | Also sweep Zaniar ranges with **3 points each**: train `{60, 80, 100}`, test `{15, 22, 30}` | Sensitivity inside Zaniar’s quoted ranges |
| — | Non-split arm on **DCCCD only**: number of snapshots in `{60, 75, 90}` | Bigger-dataset full-data counts 60–90 |
| Landmark % as main story | Story is **points per snapshot, number of snapshots, intrinsic dimension**, reuse, overlap | Matches the statistical checklist |

**Protocol for split experiments:** early 80/20 on tabular rows → train-only median impute (+ missing indicators) → MinMaxScaler → PCA → fixed-points-per-snapshot clouds independently on train and test → Ripser barcode stats → ML on barcode rows.

---

## 2. Concern A — email formula (sample-complexity style)

### 2.1 Statement

```text
suggested snapshot count  ≈  (points_per_snapshot / log(points_per_snapshot)) ^ (2 / intrinsic_dimension)
```

Where:

- points per snapshot = absolute cloud size  
- intrinsic dimension = Two-NN estimate  
- `log` = natural log (`math.log` in code)  
- suggested snapshot count = the count from this rule **alone**

### 2.2 What this concern answers

> “For a cloud with a given intrinsic dimension, if each snapshot has a given points-per-snapshot value, how many snapshots does this scaling heuristic suggest?”

It does **not** answer whether the same customers are reused. That is Concern B.

### 2.3 How intrinsic dimension is estimated

Using **scikit-dimension**:

- `TwoNN` (Facco et al.) — primary  
- `MLE` (Levina–Bickel) — secondary  
- `lPCA` — secondary  
- `DANCo` — only on smaller samples (slow)

We estimate intrinsic dimension in the **train PCA space** used for TDA (after early split).

### 2.4 Worked calculation — DCCCD (bigger dataset)

From the design run:

- Two-NN on train PCA: intrinsic dimension = 3.093  
- Joint Concern-B choice: points per snapshot = 88, 60 training snapshots, 15 test snapshots

Step-by-step:

```text
Step 1:  log(points_per_snapshot) = ln(88) = 4.477

Step 2:  88 / 4.477 = 19.655

Step 3:  2 / 3.093 = 0.6466

Step 4:  suggested snapshot count = (19.655) ^ (0.6466) ≈ 6.86
```

**What this led to:**  
The email formula, by itself, suggests only about **7 snapshots** at 88 points per snapshot, intrinsic dimension ≈ 3.1.  
The meeting default is **60 / 15**, which is much larger than 6.86.  
We do **not** overwrite 60/15 with 7. We report the gap: Concern A says “~7”; the meeting default says “60/15”; Concern B checks whether 60/15 is sampling-feasible (it is, on DCCCD — next section).

---

## 3. Concern B — reuse / sampling constraints

### 3.1 Definitions

```text
class reuse ratio  =  (points per snapshot × number of snapshots) / class pool size

Single-snapshot footprint:

    points per snapshot / class pool size  <  0.20
```

**Target:** class reuse ratio near or below 1 on every class that contributes snapshots.

**Binding class:** the minority class (smaller class pool size), because it hits reuse = 1 first.

```text
Largest points per snapshot allowed at a fixed snapshot count (reuse ≤ 1):

    floor( class pool size / number of snapshots )

Largest snapshot count allowed at a fixed points per snapshot (reuse ≤ 1):

    floor( class pool size / points per snapshot )
```

### 3.2 What this concern answers

> “If training uses 60 snapshots and test uses 15, with the same points per snapshot, can that be done without recycling the same people over and over?”

Independent of the email formula.

### 3.3 Worked calculation — DCCCD early split

Class counts after 80/20:

| Split | Defaults | Non-defaults |
|-------|--------------:|------------------:|
| Train | 5304 | 18668 |
| Test | 1326 | 4667 |

At 88 points per snapshot, 60 training snapshots:

```text
R+(train) = (88 * 60) / 5304 = 0.995  ≤ 1
R-(train) = (88 * 60) / 18668 = 0.283  ≤ 1
```

At 88 points per snapshot, 15 test snapshots:

```text
R+(test) = (88 * 15) / 1326 = 0.995  ≤ 1
R-(test) = (88 * 15) / 4667 = 0.283  ≤ 1
```

Also:

```text
88 / 5304 = 0.017  <  0.20
```

**What this led to:**  
On DCCCD, meeting **60/15 is reuse-feasible** at 88 points per snapshot.  
So for the bigger dataset we can honour the meeting default under Concern B, while Concern A still only suggests about 7 snapshots.

### 3.4 Joint choice on smaller datasets

When the minority test pool is tiny (Statlog), strict reuse ≤ 1 cannot support both large points per snapshot and 60/15 snapshots.  
We then choose jointly:

1. Prefer reuse ≤ 1 on **both** train and test minorities  
2. Push `training_snapshot_count` toward 60 and `test_snapshot_count` toward 15  
3. Prefer larger points per snapshot among ties  
4. Only if needed, relax **test** reuse to ≤ 2 (documented)

**Effective defaults from the design run:**

| Dataset | intrinsic dimension (Two-NN) | Points per snapshot | train number of snapshots | test number of snapshots | suggested snapshot count | Notes |
|---------|-------------:|----:|----------:|---------:|------------:|-------|
| **DCCCD** | 3.09 | **88** | **60** | **15** | 6.86 | Full meeting default |
| Statlog German | 4.74 | 5 | 48 | 12 | 1.61 | Concern B reduced number of snapshots |

**Oral line:**  
“On DCCCD we can run the meeting’s 60/15 under reuse ≤ 1. On smaller sets, reuse math forces smaller number of snapshots or tiny points per snapshot; the formula alone wanted even fewer snapshots (~1–7).”

---

## 4. Concern A vs Concern B — keep them separate

```text
Concern A (formula)  →  “How many snapshots does (points per snapshot, intrinsic dimension) suggest?”
                        DCCCD: ~7 at 88 points per snapshot

Concern B (reuse)    →  “Is (points per snapshot, number of snapshots) sampling-legal on each class pool?”
                        DCCCD: 60/15 at 88 points per snapshot is legal (R ≈ 0.995)

Meeting default      →  Use 60/15 when Concern B allows it
Zaniar sweep         →  Also try {60,80,100} x {15,22,30} when reuse allows
ML metrics           →  Third concern: does barcode ML actually work?
```

Never say “the formula gave 60.” It did not.  
Never say “reuse requires 7.” It does not.  
Say: “formula ≈ 7; reuse allows up to ~60 at this points per snapshot; we run the meeting default and the sweep, and we report all three stories.”

---

## 5. Overlapping — all three layers

For each class pool we store the **index set** of every snapshot and compute:

### 5.1 Pairwise overlap

For two snapshots A and B, each of size points per snapshot:

```text
overlap_fraction = |A ∩ B| / points_per_snapshot

Jaccard          = |A ∩ B| / |A ∪ B|
```

Under independent sampling without replacement, expected overlap is about:

```text
E[ overlap_fraction ]  ≈  points_per_snapshot / class_pool_size
```

Example DCCCD train defaults: 88 / 5304 ≈ 0.017 → a typical pair shares ~1.7% of points.

### 5.2 Reuse ratio

```text
reuse ratio = (points_per_snapshot * n_snapshots) / class_pool_size
```

(global recycling across the whole library, not just pairs)

### 5.3 Formal significance tests

Null: snapshots behave like independent uniform draws of size points per snapshot.

1. **Monte Carlo test** on mean pairwise overlap: simulate null libraries;  
   `p` = fraction of null mean-overlaps ≥ observed (excess overlap).  
2. **Mann–Whitney U** (alternative: observed pair-overlaps stochastically greater than null).

**How to read:** large `p` → overlap looks like chance (good). Small `p` → systematic excess dependence.

JSON outputs: `6_Results/Early_Split_No_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/<dataset>/overlap_*.json`.

---

## 6. Experimental grid actually executed

### 6.1 Default arm

- Early-split, no undersampling  
- Effective `(training_snapshot_count, test_snapshot_count)` from §3.4 (60/15 on DCCCD)  
- points per snapshot sweep: 3 points under the chosen cap (DCCCD: 29, 58, 88)

### 6.2 Zaniar sweep (3×3)

At a reuse-feasible points per snapshot (for DCCCD: 44 points per snapshot, the largest points per snapshot that still allows 100 training snapshots and 30 test snapshots with reuse ≤ 1):

```text
(training_snapshot_count, test_snapshot_count) in {60, 80, 100} × {15, 22, 30}
```

**Skipped** when Concern B fails on train or test (logged in `reuse_skips.csv`).

At the default chosen 88 points per snapshot on DCCCD, the upper grid is illegal, e.g.:

```text
R = (88 * 80) / 5304 ≈ 1.33  >  1
```

### 6.3 DCCCD non-split arm (bigger dataset 60–90)

Full-data PCA (sensitivity arm — not the clean protocol), number of snapshots in `{60, 75, 90}`, same points per snapshot.  
Then stratified 80/20 on **barcode rows** for ML.

```text
max_snapshot_count(minority=6630, points_per_snapshot=88) = floor(6630 / 88) = 75
```

So **90 snapshots is skipped** under reuse ≤ 1.

---

## 7. ML results (completed)

Sources:

- Per dataset: `6_Results/Early_Split_No_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/<Folder>/ml_results.csv`  
- Aggregate: `6_Results/Early_Split_No_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/all_ml_results.csv`  

Numbers below use the **final design’s effective `(training_snapshot_count, test_snapshot_count)`**.

### 7.1 DCCCD — bigger dataset (headline)

**Default 60/15, no undersampling, fixed points per snapshot** (mean across 5 models):

| Points per snapshot | Mean balanced acc | Mean ROC-AUC | Best model (bal. acc.) |
|----:|------------------:|-------------:|------------------------|
| 29 | 0.787 | 0.864 | — |
| 58 | 0.853 | 0.951 | — |
| **88** | **0.920** | **0.982** | SVM / RF / logistic ≈ **0.933** |

**What this led to:** larger points per snapshot (still reuse-legal) improves cloud separation under clean early-split. Formula wanted about 7 snapshots; meeting 60/15 still yields strong barcode-row metrics on DCCCD.

**Zaniar 3×3 at 44 points per snapshot** (mean balanced accuracy across models):

| train number of snapshots \ test number of snapshots | 15 | 22 | 30 |
|---------------------:|---:|---:|---:|
| 60 | 0.873 | 0.841 | 0.777 |
| 80 | 0.833 | 0.823 | 0.777 |
| 100 | 0.827 | 0.823 | 0.770 |

**What this led to:** moving up Zaniar’s ranges does **not** help once reuse is controlled — larger number of snapshots / larger test libraries slightly weaken mean balanced accuracy vs the 60/15 corner at the same 44 points per snapshot.

**Non-split DCCCD number of snapshots in `{60, 75}`** (`90` skipped: reuse = 1.195 > 1):

| number of snapshots | Mean bal. acc | Mean ROC-AUC |
|----:|--------------:|-------------:|
| 60 | 0.808 | 0.946 |
| 75 | 0.747 | 0.851 |

Full-data PCA + later barcode split is **weaker** than the clean early-split 60/15 arm at 88 points per snapshot.

### 7.2 Overlap tests on DCCCD (train, 88 points per snapshot, 60 snapshots)

| Class | Mean pairwise overlap | Theory (points per snapshot / class size) | Reuse ratio | Monte Carlo `p` (excess) | Mann–Whitney `p` |
|-------|----------------------:|-------------:|----------:|-------------------------:|-----------------:|
| Default | 0.0177 | 0.0166 | 0.995 | **0.020** | **0.011** |
| Non-default | 0.0051 | 0.0047 | 0.283 | 0.444 | 0.567 |

**What this led to:** non-default snapshots look like independent draws. Default snapshots show a **mild** excess overlap vs the null (`p ≈ 0.02`) while still sitting at the reuse boundary reuse ≈ 1. Worth watching; not a disaster like reuse = 25 under 500 snapshots.

### 7.3 Other datasets (effective defaults; mean bal. acc. by points per snapshot)

| Dataset | Eff. number of snapshots train/test | points per snapshot values | Mean bal. acc. | Best bal. acc. |
|---------|--------------------:|-----------:|---------------:|---------------:|
| Statlog German | 48 / 12 | 3, 5 | 0.48 → 0.56 | logistic 0.708 at 5 points per snapshot |

Zaniar upper grid is **reuse-infeasible** on Statlog at any practical points per snapshot — Concern B correctly skipped those cells (`reuse_skips.csv`).

### 7.4 How to interpret barcode ML (critical)

- Each row is a **snapshot**, not a customer  
- Labels are assigned by which class pool was sampled  
- Success means “topological summaries of the two clouds differ enough to classify clouds”  
- It is **not** automatically a deployable individual default score  
- Small test libraries (Statlog test 12 snapshots) make single-run accuracy noisy — quote trends, not one lucky cell  

---

## 8. Per-class safe minima (how we pick conservatively)

For each class at the chosen points per snapshot:

```text
max_snapshot_count(c) = floor(class_pool_size / points_per_snapshot)
```

Conservative experimental number of snapshots must satisfy:

```text
number of snapshots  ≤  min over classes of max_snapshot_count(c)
```

On DCCCD train at 88 points per snapshot:

```text
floor(5304 / 88) = 60
```

We sit exactly on the minority bound for 60 training snapshots. That is intentional and tight.

---

## 9. File map

| Path | Role |
|------|------|
| `5_Experiments/Early_Split_No_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/utils.py` | Formulas, overlap tests, fixed-points-per-snapshot sampling |
| `5_Experiments/Early_Split_No_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/run_protocol.py` | Orchestrator |
| `6_Results/Early_Split_No_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/all_designs.json` | All design decisions |
| `.../<dataset>/worked_calculations.csv` | Step-by-step numeric audit |
| `.../<dataset>/concern_A_formula_rows.csv` | Formula table |
| `.../<dataset>/concern_B_reuse_rows.csv` | Reuse table |
| `.../<dataset>/overlap_*.json` | Pairwise + significance |
| `.../<dataset>/ml_results.csv` | Classifier metrics |
| `utils.py` | `formula_l_from_t_b`, `select_landmarks_fixed_t`, `reuse_ratio_tl_over_n` |

---

## 10. Thirty-second oral pitch

“Undersampling and percentage landmarks were dropped.  
Points per snapshot are fixed, set train/test snapshot counts to 60/15 by default, and sweep three points in Zaniar’s ranges.  
Intrinsic dimension from scikit-dimension feeds the email formula

```text
suggested snapshot count ≈ (points_per_snapshot / log(points_per_snapshot))^(2/intrinsic_dimension)
```

— on DCCCD that gives ≈7, which we report separately from reuse.  
Reuse `R = (t * l) / n` shows 60/15 is legal on DCCCD at 88 points per snapshot with reuse ≈ 1.  
Pairwise snapshot overlap is also measured and test it against an independence null.  
ML is then run on barcode rows under the clean early-split protocol — and it is interpreted as cloud separation, not individual scoring.”
