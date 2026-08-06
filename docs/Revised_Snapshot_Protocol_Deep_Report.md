# Experiment 28 — Revised Snapshot Protocol
## Deep Explanation Report (What / How / Why / Formulas / Calculations)

**Status:** design + ML complete for all six datasets under `6_Results/28_Revised_Snapshot_Protocol/`.  
**Code:** `5_Experiments/28_Revised_Snapshot_Protocol/`  
**Package for intrinsic dimension:** [scikit-dimension](https://pypi.org/project/scikit-dimension/) (`TwoNN`, `MLE`, `lPCA`, optional `DANCo`).

Notation used everywhere below:

| Symbol | Meaning |
|--------|---------|
| `t` | points per snapshot (fixed absolute count) |
| `l` | number of snapshots |
| `b` | intrinsic dimension (Two-NN primary) |
| `n_c` | number of rows in one class pool |
| `R` | reuse score |

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
| `t` = percentage of (balanced) class | **Fixed absolute `t`** (same for train and test) | Percentage couples snapshot size to class size/balancing |
| `l = 500` snapshots | Default **train `l = 60`**, **test `l = 15`** | 500 caused extreme reuse (Exp 24) |
| — | Also sweep Zaniar ranges with **3 points each**: train `{60, 80, 100}`, test `{15, 22, 30}` | Sensitivity inside Zaniar’s quoted ranges |
| — | Non-split arm on **DCCCD only**: `l` in `{60, 75, 90}` | Bigger-dataset full-data counts 60–90 |
| Landmark % as main story | Story is **`t`, `l`, `b`**, reuse, overlap | Matches the statistical checklist |

**Protocol for split experiments:** early 80/20 on tabular rows → train-only median impute (+ missing indicators) → MinMaxScaler → PCA → fixed-`t` snapshots independently on train and test → Ripser barcode stats → ML on barcode rows.

---

## 2. Concern A — email formula (sample-complexity style)

### 2.1 Statement

```text
l_formula  ≈  ( t / log(t) ) ^ (2 / b)
```

Where:

- `t` = points per snapshot  
- `b` = intrinsic dimension (Two-NN)  
- `log` = natural log (`math.log` in code)  
- `l_formula` = suggested snapshot count from this rule **alone**

### 2.2 What this concern answers

> “For a cloud with intrinsic dimension `b`, if each snapshot has `t` points, how many snapshots does this scaling heuristic suggest?”

It does **not** answer whether the same customers are reused. That is Concern B.

### 2.3 How we estimate `b`

Using **scikit-dimension**:

- `TwoNN` (Facco et al.) — primary  
- `MLE` (Levina–Bickel) — secondary  
- `lPCA` — secondary  
- `DANCo` — only on smaller samples (slow)

We estimate `b` in the **train PCA space** used for TDA (after early split).

### 2.4 Worked calculation — DCCCD (bigger dataset)

From the design run:

- Two-NN on train PCA: `b = 3.093`  
- Joint Concern-B choice: `t = 88`, train `l = 60`, test `l = 15`

Step-by-step:

```text
Step 1:  log(t) = ln(88) = 4.477

Step 2:  t / log(t) = 88 / 4.477 = 19.655

Step 3:  2 / b = 2 / 3.093 = 0.6466

Step 4:  l_formula = (19.655) ^ (0.6466) ≈ 6.86
```

**What this led to:**  
The email formula, by itself, suggests only about **7 snapshots** at `t = 88`, `b ≈ 3.1`.  
The meeting default is **60 / 15**, which is much larger than 6.86.  
We do **not** overwrite 60/15 with 7. We report the gap: Concern A says “~7”; the meeting default says “60/15”; Concern B checks whether 60/15 is sampling-feasible (it is, on DCCCD — next section).

---

## 3. Concern B — reuse / sampling constraints

### 3.1 Definitions

```text
Reuse score for class c:

    R_c  =  (t * l) / n_c

Single-snapshot footprint:

    t / n_c  <  0.20
```

**Target:** `R_c` near or below 1 on every class that contributes snapshots.

**Binding class:** the minority class (smaller `n_c`), because it hits `R = 1` first.

```text
Largest t allowed at a fixed l (reuse ≤ 1):

    t_max(n_c, l)  =  floor( n_c / l )

Largest l allowed at a fixed t (reuse ≤ 1):

    l_max(n_c, t)  =  floor( n_c / t )
```

### 3.2 What this concern answers

> “If I insist on train `l = 60` and test `l = 15` with the same `t`, can I do it without recycling the same people over and over?”

Independent of the email formula.

### 3.3 Worked calculation — DCCCD early split

Class counts after 80/20:

| Split | Defaults `n+` | Non-defaults `n-` |
|-------|--------------:|------------------:|
| Train | 5304 | 18668 |
| Test | 1326 | 4667 |

At `t = 88`, train `l = 60`:

```text
R+(train) = (88 * 60) / 5304 = 0.995  ≤ 1
R-(train) = (88 * 60) / 18668 = 0.283  ≤ 1
```

At `t = 88`, test `l = 15`:

```text
R+(test) = (88 * 15) / 1326 = 0.995  ≤ 1
R-(test) = (88 * 15) / 4667 = 0.283  ≤ 1
```

Also:

```text
t / n+(train) = 88 / 5304 = 0.017  <  0.20
```

**What this led to:**  
On DCCCD, meeting **60/15 is reuse-feasible** at `t = 88`.  
So for the bigger dataset we can honour the meeting default under Concern B, while Concern A still only suggests `l ≈ 7`.

### 3.4 Joint choice on smaller datasets

When minority test pools are tiny (PKDD, Taiwan, Statlog), strict `R ≤ 1` cannot support both large `t` and `l = 60/15`.  
We then choose jointly:

1. Prefer `R ≤ 1` on **both** train and test minorities  
2. Push `train_l` toward 60 and `test_l` toward 15  
3. Prefer larger `t` among ties  
4. Only if needed, relax **test** reuse to ≤ 2 (documented)

**Effective defaults from the design run:**

| Dataset | `b` (Two-NN) | `t` | train `l` | test `l` | `l_formula` | Notes |
|---------|-------------:|----:|----------:|---------:|------------:|-------|
| **DCCCD** | 3.09 | **88** | **60** | **15** | 6.86 | Full meeting default |
| Polish bankruptcy | 2.41 | 6 | 60 | 15 | 2.72 | Full meeting default; small `t` |
| Statlog German | 4.74 | 5 | 48 | 12 | 1.61 | Concern B reduced `l` |
| South German | 4.85 | 5 | 48 | 12 | 1.60 | Same pattern |
| Taiwan bankruptcy | 6.21 | 5 | 35 | 8 | 1.44 | Concern B reduced `l` |
| PKDD Czech | 5.87 | 5 | 12 | 3 | 1.47 | Smallest pools; heavily constrained |

**Oral line:**  
“On DCCCD we can run the meeting’s 60/15 under reuse ≤ 1. On smaller sets, reuse math forces smaller `l` or tiny `t`; the formula alone wanted even fewer snapshots (~1–7).”

---

## 4. Concern A vs Concern B — keep them separate

```text
Concern A (formula)  →  “How many snapshots does (t, b) suggest?”
                        DCCCD: ~7 at t = 88

Concern B (reuse)    →  “Is (t, l) sampling-legal on each class pool?”
                        DCCCD: 60/15 at t = 88 is legal (R ≈ 0.995)

Meeting default      →  Use 60/15 when Concern B allows it
Zaniar sweep         →  Also try {60,80,100} x {15,22,30} when reuse allows
ML metrics           →  Third concern: does barcode ML actually work?
```

Never say “the formula gave 60.” It did not.  
Never say “reuse requires 7.” It does not.  
Say: “formula ≈ 7; reuse allows up to ~60 at this `t`; we run the meeting default and the sweep, and we report all three stories.”

---

## 5. Overlapping — all three layers

For each class pool we store the **index set** of every snapshot and compute:

### 5.1 Pairwise overlap

For two snapshots A and B, each of size `t`:

```text
overlap_fraction = |A ∩ B| / t

Jaccard          = |A ∩ B| / |A ∪ B|
```

Under independent sampling without replacement, expected overlap is about:

```text
E[ overlap_fraction ]  ≈  t / n_c
```

Example DCCCD train defaults: `t / n+ = 88 / 5304 ≈ 0.017` → a typical pair shares ~1.7% of points.

### 5.2 Reuse ratio

```text
R = (t * l) / n_c
```

(global recycling across the whole library, not just pairs)

### 5.3 Formal significance tests

Null: snapshots behave like independent uniform draws of size `t`.

1. **Monte Carlo test** on mean pairwise overlap: simulate null libraries;  
   `p` = fraction of null mean-overlaps ≥ observed (excess overlap).  
2. **Mann–Whitney U** (alternative: observed pair-overlaps stochastically greater than null).

**How to read:** large `p` → overlap looks like chance (good). Small `p` → systematic excess dependence.

JSON outputs: `6_Results/28_Revised_Snapshot_Protocol/<dataset>/overlap_*.json`.

---

## 6. Experimental grid actually executed

### 6.1 Default arm

- Early-split, no undersampling  
- Effective `(train_l, test_l)` from §3.4 (60/15 on DCCCD & Polish)  
- `t` sweep: 3 points under the chosen cap (DCCCD: 29, 58, 88)

### 6.2 Zaniar sweep (3×3)

At a reuse-feasible `t` (for DCCCD: `t = 44`, the largest `t` that still allows train `l = 100` and test `l = 30` with `R ≤ 1`):

```text
(train_l, test_l) in {60, 80, 100} × {15, 22, 30}
```

**Skipped** when Concern B fails on train or test (logged in `reuse_skips.csv`).

At the default chosen `t = 88` on DCCCD, the upper grid is illegal, e.g.:

```text
R = (88 * 80) / 5304 ≈ 1.33  >  1
```

### 6.3 DCCCD non-split arm (bigger dataset 60–90)

Full-data PCA (sensitivity arm — not the clean protocol), `l` in `{60, 75, 90}`, same `t`.  
Then stratified 80/20 on **barcode rows** for ML.

```text
l_max(n_min=6630, t=88) = floor(6630 / 88) = 75
```

So **`l = 90` is skipped** under `R ≤ 1`.

---

## 7. ML results (completed)

Sources:

- Per dataset: `6_Results/28_Revised_Snapshot_Protocol/<Folder>/ml_results.csv`  
- Aggregate: `6_Results/28_Revised_Snapshot_Protocol/all_ml_results.csv`  

Numbers below use the **final design’s effective `(train_l, test_l)`**.

### 7.1 DCCCD — bigger dataset (headline)

**Default 60/15, no undersampling, fixed `t`** (mean across 5 models):

| `t` | Mean balanced acc | Mean ROC-AUC | Best model (bal. acc.) |
|----:|------------------:|-------------:|------------------------|
| 29 | 0.787 | 0.864 | — |
| 58 | 0.853 | 0.951 | — |
| **88** | **0.920** | **0.982** | SVM / RF / logistic ≈ **0.933** |

**What this led to:** larger `t` (still reuse-legal) improves cloud separation under clean early-split. Formula wanted `l ≈ 7`; meeting 60/15 still yields strong barcode-row metrics on DCCCD.

**Zaniar 3×3 at `t = 44`** (mean balanced accuracy across models):

| train `l` \ test `l` | 15 | 22 | 30 |
|---------------------:|---:|---:|---:|
| 60 | 0.873 | 0.841 | 0.777 |
| 80 | 0.833 | 0.823 | 0.777 |
| 100 | 0.827 | 0.823 | 0.770 |

**What this led to:** moving up Zaniar’s ranges does **not** help once reuse is controlled — larger `l` / larger test libraries slightly weaken mean balanced accuracy vs the 60/15 corner at the same `t = 44`.

**Non-split DCCCD `l` in `{60, 75}`** (`90` skipped: `R = 1.195 > 1`):

| `l` | Mean bal. acc | Mean ROC-AUC |
|----:|--------------:|-------------:|
| 60 | 0.808 | 0.946 |
| 75 | 0.747 | 0.851 |

Full-data PCA + later barcode split is **weaker** than the clean early-split 60/15 arm at `t = 88`.

### 7.2 Overlap tests on DCCCD (train, `t = 88`, `l = 60`)

| Class | Mean pairwise overlap | Theory `t/n` | Reuse `R` | Monte Carlo `p` (excess) | Mann–Whitney `p` |
|-------|----------------------:|-------------:|----------:|-------------------------:|-----------------:|
| Default | 0.0177 | 0.0166 | 0.995 | **0.020** | **0.011** |
| Non-default | 0.0051 | 0.0047 | 0.283 | 0.444 | 0.567 |

**What this led to:** non-default snapshots look like independent draws. Default snapshots show a **mild** excess overlap vs the null (`p ≈ 0.02`) while still sitting at the reuse boundary `R ≈ 1`. Worth watching; not a disaster like `R = 25` under `l = 500`.

### 7.3 Other datasets (effective defaults; mean bal. acc. by `t`)

| Dataset | Eff. `l` train/test | `t` values | Mean bal. acc. | Best bal. acc. |
|---------|--------------------:|-----------:|---------------:|---------------:|
| Polish bankruptcy | 60 / 15 | 3, 4, **6** | 0.61 → 0.70 → **0.90** | XGB **0.933** at `t = 6` |
| Statlog German | 48 / 12 | 3, 5 | 0.48 → 0.56 | logistic 0.708 at `t = 5` |
| South German | 48 / 12 | 3, 5 | 0.50 → 0.43 | logistic 0.625 at `t = 5` |
| Taiwan bankruptcy | 35 / 8 | 3, 5 | 0.54 → 0.51 | XGB 0.625 at `t = 3` |
| PKDD Czech | 12 / 3 | 3, 5 | 0.63 → 0.43 | logistic 0.667 at `t = 3` |

Polish Zaniar at `t = 3`: mean bal. acc. ≈ 0.62 (below its best default `t = 6` @ 60/15).  
Zaniar upper grid is **reuse-infeasible** on Statlog / South German / PKDD / Taiwan at any practical `t` — Concern B correctly skipped those cells (`reuse_skips.csv`).

### 7.4 How to interpret barcode ML (critical)

- Each row is a **snapshot**, not a customer  
- Labels are assigned by which class pool was sampled  
- Success means “topological summaries of the two clouds differ enough to classify clouds”  
- It is **not** automatically a deployable individual default score  
- Tiny test libraries (PKDD test `l = 3`) make single-run accuracy noisy — quote trends, not one lucky cell  

---

## 8. Per-class safe minima (how we pick conservatively)

For each class at the chosen `t`:

```text
l_max(c) = floor( n_c / t )
```

Conservative experimental `l` must satisfy:

```text
l  ≤  min over classes of l_max(c)
```

On DCCCD train at `t = 88`:

```text
floor(5304 / 88) = 60
```

We sit exactly on the minority bound for train `l = 60`. That is intentional and tight.

---

## 9. File map

| Path | Role |
|------|------|
| `5_Experiments/28_Revised_Snapshot_Protocol/protocol_lib.py` | Formulas, overlap tests, fixed-`t` snapshots |
| `5_Experiments/28_Revised_Snapshot_Protocol/run_protocol.py` | Orchestrator |
| `6_Results/28_Revised_Snapshot_Protocol/all_designs.json` | All design decisions |
| `.../<dataset>/worked_calculations.csv` | Step-by-step numeric audit |
| `.../<dataset>/concern_A_formula_rows.csv` | Formula table |
| `.../<dataset>/concern_B_reuse_rows.csv` | Reuse table |
| `.../<dataset>/overlap_*.json` | Pairwise + significance |
| `.../<dataset>/ml_results.csv` | Classifier metrics |
| `utils.py` | `formula_l_from_t_b`, `select_landmarks_fixed_t`, `reuse_ratio_tl_over_n` |

---

## 10. Thirty-second oral pitch

“We stopped undersampling and stopped using percentage landmarks.  
We fix `t`, set train/test snapshot counts to 60/15 by default, and sweep three points in Zaniar’s ranges.  
Intrinsic dimension from scikit-dimension feeds the email formula

```text
l ≈ (t / log(t))^(2/b)
```

— on DCCCD that gives ≈7, which we report separately from reuse.  
Reuse `R = (t * l) / n` shows 60/15 is legal on DCCCD at `t = 88` with `R ≈ 1`.  
We also measure pairwise snapshot overlap and test it against an independence null.  
ML is then run on barcode rows under the clean early-split protocol — and we interpret it as cloud separation, not individual scoring.”
