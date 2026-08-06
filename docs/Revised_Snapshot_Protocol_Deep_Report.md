# Experiment 28 — Revised Snapshot Protocol  
## Deep Explanation Report (What / How / Why / Formulas / Calculations)

**Status:** design stage complete for all six datasets; ML stage running (results appended under `6_Results/28_Revised_Snapshot_Protocol/`).  
**Code:** `5_Experiments/28_Revised_Snapshot_Protocol/`  
**Package for intrinsic dimension:** [`scikit-dimension`](https://pypi.org/project/scikit-dimension/) (`skdim.id.TwoNN`, `MLE`, `lPCA`, optional `DANCo`).

---

## 0. How to use this document

Read it top-to-bottom once. When presenting, you can jump:

1. **§1** — what changed vs the old pipeline  
2. **§2–3** — two *separate* mathematical concerns (formula vs reuse)  
3. **§4** — worked numbers for **DCCCD** (the “bigger dataset”)  
4. **§5** — overlap + significance tests  
5. **§6** — grids we actually ran (60/15 + Zaniar 3-point sweeps + DCCCD 60–90)  
6. **§7** — ML results (updated as runs finish)  
7. **§8** — what each result *means* (and what it does not)

---

## 1. What the meeting changed (old → new)

| Old habit | New rule | Why |
|-----------|----------|-----|
| Undersample majority to minority before landmarks | **No class balancing** — keep full class pools | Undersampling throws away majority geometry and changes the point cloud the topology sees |
| \(t\) = percentage of (balanced) class | **Fixed absolute \(t\)** (same for train and test) | Percentage \(t\) couples snapshot size to class size and balancing; absolute \(t\) is a controllable experimental knob |
| \(l = 500\) snapshots | Default **train \(l=60\)**, **test \(l=15\)** | 500 caused extreme reuse (Exp 24); meeting set 60/15 as the new default |
| — | Also sweep Zaniar ranges with **3 points each**: train \(\{60,80,100\}\), test \(\{15,22,30\}\) | Sensitivity inside the ranges Zaniar quoted |
| — | Non-split arm on **DCCCD only**: \(l \in \{60,75,90\}\) | “Bigger dataset” full-data snapshot counts 60–90 |
| Landmark % as main story | Story is **\(t\), \(l\), \(b\)**, reuse, overlap | Matches the statistical checklist |

**Protocol for split experiments:** early 80/20 on tabular rows → train-only median impute (+ missing indicators) → MinMaxScaler → PCA → fixed-\(t\) snapshots **independently** on train and test → Ripser barcode stats → ML on barcode rows.

---

## 2. Concern A — email formula (sample-complexity style)

### 2.1 Statement

\[
\boxed{l_{\text{formula}} \;\approx\; \left(\frac{t}{\log t}\right)^{2/b}}
\]

- \(t\): points per snapshot  
- \(b\): intrinsic dimension of the point cloud (we use **Two-NN** as primary \(b\))  
- \(\log\): natural log in our implementation (`math.log`)  
- \(l_{\text{formula}}\): suggested number of snapshots from this rule alone  

### 2.2 What this concern answers

> “For a cloud with intrinsic dimension \(b\), if each snapshot has \(t\) points, how many snapshots does this scaling heuristic suggest?”

It does **not** answer whether the same customers are reused. That is Concern B.

### 2.3 How we estimate \(b\)

Using **scikit-dimension**:

- `TwoNN` (Facco et al.) — primary  
- `MLE` (Levina–Bickel) — secondary  
- `lPCA` — secondary  
- `DANCo` — only on smaller samples (slow)

We estimate \(b\) in the **train PCA space** actually used for TDA (after early split).

### 2.4 Worked calculation — DCCCD (bigger dataset)

From the design run:

- Two-NN on train PCA: \(b = 3.093\)  
- Joint Concern-B choice (see §3): \(t = 88\), train \(l=60\), test \(l=15\)

Step-by-step:

\[
\ln t = \ln 88 = 4.477
\]

\[
\frac{t}{\ln t} = \frac{88}{4.477} = 19.655
\]

\[
\frac{2}{b} = \frac{2}{3.093} = 0.6466
\]

\[
l_{\text{formula}} = (19.655)^{0.6466} \approx 6.86
\]

**What this led to:**  
The email formula, *by itself*, suggests only about **7 snapshots** at \(t=88\), \(b\approx 3.1\).  
The meeting default is **60 / 15**, which is **much larger** than 6.86.  
We **do not** overwrite 60/15 with 7. We **report** the gap: Concern A says “~7”; the meeting default says “60/15”; Concern B checks whether 60/15 is sampling-feasible (it is, on DCCCD — see next section).

---

## 3. Concern B — reuse / sampling constraints

### 3.1 Definitions

For class pool size \(n_c\):

\[
R_c = \frac{t \cdot l}{n_c}
\qquad\text{(reuse score)}
\]

\[
\frac{t}{n_c} < 0.20
\qquad\text{(single-snapshot footprint)}
\]

**Target:** \(R_c \lesssim 1\) on every class that contributes snapshots.

**Binding class:** the minority class (smaller \(n_c\)), because it hits \(R=1\) first.

Maximum reusable \(t\) at a fixed \(l\):

\[
t_{\max}(n_c,l) = \left\lfloor \frac{n_c}{l} \right\rfloor
\]

Maximum reusable \(l\) at a fixed \(t\):

\[
l_{\max}(n_c,t) = \left\lfloor \frac{n_c}{t} \right\rfloor
\]

### 3.2 What this concern answers

> “If I insist on train \(l=60\) and test \(l=15\) with the same \(t\), can I do it without recycling the same people over and over?”

Independent of the email formula.

### 3.3 Worked calculation — DCCCD early split

Class counts after 80/20:

| Split | Defaults \(n_+\) | Non-defaults \(n_-\) |
|-------|-----------------:|---------------------:|
| Train | 5304 | 18668 |
| Test | 1326 | 4667 |

At \(t=88\), \(l_{\text{train}}=60\):

\[
R_+^{\text{train}} = \frac{88\times 60}{5304} = 0.995 \le 1
\]

\[
R_-^{\text{train}} = \frac{88\times 60}{18668} = 0.283 \le 1
\]

At \(t=88\), \(l_{\text{test}}=15\):

\[
R_+^{\text{test}} = \frac{88\times 15}{1326} = 0.995 \le 1
\]

\[
R_-^{\text{test}} = \frac{88\times 15}{4667} = 0.283 \le 1
\]

Also \(t/n_+^{\text{train}} = 88/5304 = 0.017 < 0.20\).

**What this led to:**  
On DCCCD, meeting **60/15 is reuse-feasible** at \(t=88\).  
So for the bigger dataset we can honour the meeting default under Concern B, while Concern A still only suggests \(l\approx 7\).

### 3.4 Joint choice on smaller datasets

When minority test pools are tiny (PKDD, Taiwan, Statlog), **strict** \(R\le 1\) cannot support both large \(t\) and \(l=60/15\).  
We then solve a joint programme:

1. Prefer \(R\le 1\) on **both** train and test minorities  
2. Push `train_l` toward 60 and `test_l` toward 15  
3. Prefer larger \(t\) among ties  
4. Only if needed, relax **test** reuse to \(\le 2\) (documented)

**Effective defaults from the design run:**

| Dataset | \(b\) (Two-NN) | \(t\) | train \(l\) | test \(l\) | \(l_{\text{formula}}\) | Notes |
|---------|---------------:|------:|------------:|-----------:|------------------------:|-------|
| **DCCCD** | 3.09 | **88** | **60** | **15** | 6.86 | Full meeting default |
| Polish bankruptcy | 2.41 | 6 | 60 | 15 | 2.72 | Full meeting default; small \(t\) |
| Statlog German | 4.74 | 5 | 48 | 12 | 1.61 | Concern B reduced \(l\) |
| South German | 4.85 | 5 | 48 | 12 | 1.60 | Same pattern |
| Taiwan bankruptcy | 6.21 | 5 | 35 | 8 | 1.44 | Concern B reduced \(l\) |
| PKDD Czech | 5.87 | 5 | 12 | 3 | 1.47 | Smallest pools; heavily constrained |

**Oral line:**  
“On DCCCD we can run the meeting’s 60/15 under reuse ≤ 1. On smaller sets, reuse math forces smaller \(l\) or tiny \(t\); the formula alone wanted even fewer snapshots (~1–7).”

---

## 4. Concern A vs Concern B — keep them separate in your head

```text
Concern A (formula)     →  “How many snapshots does (t,b) suggest?”
                           DCCCD: ~7 at t=88

Concern B (reuse)       →  “Is (t,l) sampling-legal on each class pool?”
                           DCCCD: 60/15 at t=88 is legal (R≈0.995)

Meeting default         →  Use 60/15 when Concern B allows it
Zaniar sweep            →  Also try {60,80,100} × {15,22,30} when reuse allows
ML metrics              →  Third concern: does barcode ML actually work?
```

Never say “the formula gave 60.” It did not.  
Never say “reuse requires 7.” It does not.  
Say: “formula ≈ 7; reuse allows up to ~60 at this \(t\); we run the meeting default and the sweep, and we report all three stories.”

---

## 5. Overlapping — all three layers

For each class pool we store the **index set** of every snapshot and compute:

### 5.1 Pairwise overlap

For snapshots \(A,B\) of size \(t\):

\[
\text{overlap fraction} = \frac{|A \cap B|}{t},
\qquad
\text{Jaccard} = \frac{|A \cap B|}{|A \cup B|}
\]

Under independent sampling without replacement, the expected overlap fraction is approximately

\[
\mathbb{E}\!\left[\frac{|A\cap B|}{t}\right] \approx \frac{t}{n_c}.
\]

Example DCCCD train defaults: \(t/n_+ = 88/5304 \approx 0.017\) → typical pair shares ~1.7% of points.

### 5.2 Reuse ratio

Already \(R = tl/n_c\) (global recycling, not just pairs).

### 5.3 Formal significance tests

Null: snapshots behave like independent uniform draws of size \(t\).

1. **Monte Carlo test** on mean pairwise overlap: simulate null libraries;  
   \(p =\) fraction of null mean-overlaps \(\ge\) observed (excess overlap).  
2. **Mann–Whitney U** (alternative: observed pair-overlaps stochastically greater than null).

**How to read:** large \(p\) ⇒ overlap looks like chance (good). Small \(p\) ⇒ systematic excess dependence.

JSON outputs: `6_Results/28_Revised_Snapshot_Protocol/<dataset>/overlap_*.json`.

---

## 6. Experimental grid actually executed

### 6.1 Default arm

- Early-split, no undersampling  
- Effective `(train_l, test_l)` from §3.4 (60/15 on DCCCD & Polish)  
- `t` sweep: 3 points under the chosen cap (DCCCD: 29, 58, 88)

### 6.2 Zaniar sweep (3×3)

At the chosen \(t\), all pairs

\[
(l_{\text{train}}, l_{\text{test}}) \in \{60,80,100\}\times\{15,22,30\}
\]

except the default already run.  
**Skipped** when Concern B fails on train or test (logged in `reuse_skips.csv`).

On DCCCD at \(t=88\), many upper-grid cells skip because  
\(R = 88\times 80 / 5304 \approx 1.33 > 1\).

### 6.3 DCCCD non-split arm (bigger dataset 60–90)

Full-data PCA (sensitivity arm — not the clean protocol), \(l \in \{60,75,90\}\), same \(t\).  
Then stratified 80/20 on **barcode rows** for ML.  
Reuse check: \(l_{\max}(6630,88)=75\) ⇒ **90 is skipped** under \(R\le 1\).

---

## 7. ML results (completed)

Sources:

- Per dataset: `6_Results/28_Revised_Snapshot_Protocol/<Folder>/ml_results.csv`  
- Aggregate: `6_Results/28_Revised_Snapshot_Protocol/all_ml_results.csv`  

Numbers below use the **final design’s effective `(train_l, test_l)`** (intermediate adaptation runs are left in the CSV for audit but ignored here).

### 7.1 DCCCD — bigger dataset (headline)

**Default 60/15, no undersampling, fixed \(t\)** (mean across 5 models):

| \(t\) | Mean balanced acc | Mean ROC-AUC | Best model (bal. acc.) |
|------:|------------------:|-------------:|------------------------|
| 29 | 0.787 | 0.864 | — |
| 58 | 0.853 | 0.951 | — |
| **88** | **0.920** | **0.982** | SVM / RF / logistic ≈ **0.933** |

**What this led to:** larger \(t\) (still reuse-legal) improves cloud separation under clean early-split. Formula wanted \(l\approx 7\); meeting 60/15 still yields strong barcode-row metrics on DCCCD.

**Zaniar 3×3 at \(t=44\)** (largest \(t\) that keeps \(R\le 1\) even at train \(l=100\), test \(l=30\)):

| train \(l\) \ test \(l\) | 15 | 22 | 30 |
|---:|---:|---:|---:|
| 60 | 0.873 | 0.841 | 0.777 |
| 80 | 0.833 | 0.823 | 0.777 |
| 100 | 0.827 | 0.823 | 0.770 |

(Values = mean balanced accuracy across models.)

**What this led to:** moving up Zaniar’s ranges does **not** help once reuse is controlled — if anything, larger \(l\) / larger test libraries slightly **weaken** mean balanced accuracy vs the 60/15 corner at the same \(t=44\). The meeting default is not only simpler; it is competitive.

**Non-split DCCCD \(l\in\{60,75\}\)** (90 skipped: \(R=1.195>1\)):

| \(l\) | Mean bal. acc | Mean ROC-AUC |
|------:|--------------:|-------------:|
| 60 | 0.808 | 0.946 |
| 75 | 0.747 | 0.851 |

Full-data PCA + later barcode split is **weaker** than the clean early-split 60/15 arm at \(t=88\).

### 7.2 Overlap tests on DCCCD (train, \(t=88\), \(l=60\))

| Class | Mean pairwise overlap | Theory \(t/n\) | Reuse \(R\) | Monte Carlo \(p\) (excess) | Mann–Whitney \(p\) |
|-------|----------------------:|---------------:|------------:|---------------------------:|-------------------:|
| Default | 0.0177 | 0.0166 | 0.995 | **0.020** | **0.011** |
| Non-default | 0.0051 | 0.0047 | 0.283 | 0.444 | 0.567 |

**What this led to:** non-default snapshots look like independent draws. Default snapshots show a **mild** excess overlap vs the null (p≈0.02) while still sitting at the reuse boundary \(R\approx 1\). Worth watching; not a disaster like \(R=25\) under \(l=500\).

### 7.3 Other datasets (effective defaults; mean bal. acc. by \(t\))

| Dataset | Eff. \(l\) train/test | \(t\) values | Mean bal. acc. | Best bal. acc. |
|---------|----------------------:|-------------:|---------------:|---------------:|
| Polish bankruptcy | 60 / 15 | 3, 4, **6** | 0.61 → 0.70 → **0.90** | XGB **0.933** at \(t=6\) |
| Statlog German | 48 / 12 | 3, 5 | 0.48 → 0.56 | logistic 0.708 at \(t=5\) |
| South German | 48 / 12 | 3, 5 | 0.50 → 0.43 | logistic 0.625 at \(t=5\) |
| Taiwan bankruptcy | 35 / 8 | 3, 5 | 0.54 → 0.51 | XGB 0.625 at \(t=3\) |
| PKDD Czech | 12 / 3 | 3, 5 | 0.63 → 0.43 | logistic 0.667 at \(t=3\) |

Polish Zaniar at \(t=3\): mean bal. acc. ≈ 0.62 (below its best default \(t=6\) @ 60/15).  
Zaniar upper grid is **reuse-infeasible** on Statlog / South German / PKDD / Taiwan at any practical \(t\) — Concern B correctly skipped those cells (logged in `reuse_skips.csv`).

### 7.4 How to interpret barcode ML (critical)

- Each row is a **snapshot**, not a customer  
- Labels are assigned by which class pool was sampled  
- Success means “topological summaries of the two clouds differ enough to classify clouds”  
- It is **not** automatically a deployable individual default score  
- Tiny test libraries (PKDD test \(l=3\)) make single-run accuracy noisy — quote trends, not one lucky cell  

---

## 8. Per-class safe minima (how we pick conservatively)

For each class at the chosen \(t\):

\[
l_{\max}^{(c)} = \left\lfloor n_c / t \right\rfloor
\]

Conservative experimental \(l\) must satisfy

\[
l \le \min_c l_{\max}^{(c)}.
\]

On DCCCD train at \(t=88\): \(\lfloor 5304/88\rfloor = 60\) — we sit exactly on the minority bound for train \(l=60\). That is intentional and tight.

---

## 9. File map

| Path | Role |
|------|------|
| `5_Experiments/28_Revised_Snapshot_Protocol/protocol_lib.py` | Formulas, overlap tests, fixed-\(t\) snapshots |
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
We fix \(t\), set train/test snapshot counts to 60/15 by default, and sweep three points in Zaniar’s ranges.  
Intrinsic dimension from scikit-dimension feeds the email formula \(l\sim(t/\log t)^{2/b}\) — on DCCCD that gives ≈7, which we report separately from reuse.  
Reuse \(R=tl/n\) shows 60/15 is legal on DCCCD at \(t=88\) with \(R\approx 1\).  
We also measure pairwise snapshot overlap and test it against an independence null.  
ML is then run on barcode rows under the clean early-split protocol — and we interpret it as cloud separation, not individual scoring.”
