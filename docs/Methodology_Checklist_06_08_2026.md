# Methodology checklist (06/08/2026) — what is in the repo

Team list from 6 August 2026, mapped onto the implementation. Stage order is in `docs/Statistical_Approach_Flow.md`. Landmark / PCA / ID rationale is in `docs/Design_Decisions.md`.

---

## 1. scikit-dimension (arXiv:2109.02596)

**Done.** Package: `scikit-dimension` (`skdim`) in `requirements.txt`.

| Where | Estimators |
|-------|------------|
| Exp 26 `utils.estimate_intrinsic_dimension_skdim` (`5_Experiments/Statistics/1_Intrinsic_Dimension_Estimation/`) | TwoNN, MLE (Levina–Bickel, k=20), MiND_ML, lPCA |
| Arm Exp 9 `protocol_lib.estimate_intrinsic_dimensions` | TwoNN, MLE, lPCA, MiND_ML, DANCo when n ≤ 800 |

Hand-coded Two-NN / Levina–Bickel in `utils.py` stay as the transparent Facco / LB formulae. Exp 26 writes **both** so a reviewer can see they agree.

---

## 2. dadapy for TwoNN

**Not added, on purpose.** dadapy’s TwoNN is the same Facco estimator skdim already runs. Adding a second package for the same number would raise a dependency without changing the science. Quote skdim (the cited package) plus the hand-coded formula.

---

## 3. MIND_M, MMk, MLE, MMi — is hand-coding a problem?

**Hand-coding Two-NN alone would raise questions. We do not rely on it alone.**

| Name on the list | In this repo |
|------------------|--------------|
| MLE (Levina–Bickel) | Hand-coded (`k=10`) **and** `skdim.id.MLE(K=20)` (current skdim uses capital `K`, not `k`) |
| MIND_M / MMi | `skdim.id.MiND_ML` (the scikit-dimension name for the MiND family) |
| MMk | Not a separate skdim class we call; MOM-style estimators overlap lPCA / MLE. lPCA is reported. |
| TwoNN | Hand-coded **and** `skdim.id.TwoNN` |

Answer for a viva / reviewer: the Facco formula is written out so the methods section is auditable; the published package is run on the same matrices; we report both. We did not invent a private ID estimator and hide the package.

---

## 4. Repeat results with the previous technique of splitting barcode statistics

**Done, as three related arms — not a missing experiment.**

| Arm | What “splitting barcode statistics” means here |
|-----|------------------------------------------------|
| **Arm Exp 3** (`3_H0_Only`) | H0-only columns (drop the 12 H1 stats). |
| **Arm Exp 5** (`5_Linear_Regression_For_Prediction`; paper Exp 10 / old Exp 19) | Linear model on the full 24-D table and on the H0 slice from Historical Exp 1. |
| **Early Split TDA Exp 1** (historical Exp 23) | Split **customers** before PCA / landmarks (protocol split, not column split). |
| **Arm Exp 4** | Drop correlated barcode columns (a different split: keep / drop by correlation). |

If the intended repeat is “rerun arm experiments 6–8 on H0-only matrices”, that is a cheap follow-up (same scripts, point them at arm Exp 3 CSVs). It is **not** missing infrastructure.

---

## 5. Play with number of snapshots and points per snapshot

**Done, in two layers.**

| Layer | What it does |
|-------|----------------|
| **Arm Exp 6** | Audits historical 500 snapshots vs suggested snapshot count ≈ round(minority class count / points per snapshot). Does not rebuild Ripser. |
| **Arm Exp 9** | Rebuilds with **fixed points per snapshot**, default 60 training / 15 test snapshots, no undersampling on the canonical arm, customer split first. |

Playing with points per snapshot and number of snapshots on the historical percent grid (L10 vs L20, L5 vs L15, …) is already Historical Exp 1’s two-percent design plus arm experiments 7/8 reading both files. The dated 13/08/2026 factorial is `5_Experiments/Snapshot_Sample_Size/`.

---

## 6. Focus on intrinsic dimensionality

**Done.** Statistics experiment 1 (both datasets, before **and** after PCA, hand-coded + skdim). Arm experiment 9 **design** stage uses intrinsic dimension to talk about snapshot size. The “is intrinsic dimension ≈ 7?” check uses **after-PCA Two-NN**.

---

## 7. A section for statistics

**Done, now in three documents:**

| Document | Role |
|----------|------|
| `docs/Statistical_Approach_Flow.md` | When each stage runs; what it needs |
| `docs/Statistical_Experiments_24_27_Results.md` | Worked numbers for both datasets |
| `docs/Design_Decisions.md` | Why knobs differ (L, PCA, ID before/after) |

Plus `REPORT.md` in every experiment folder (arm experiments 6–9 and Statistics Exp 1 also have a short per-dataset report).

---

## Not missing (already in the pipeline, easy to confuse with the list)

- Early train/test split: Early Split TDA Exp 1 and arm Exp 9.
- Variance-matched PCA: archived Exp 13. Component sweeps: archived Exp 16 / 18.
- Mapper shape of barcode space: archived Exp 21 (consumes Historical Exp 1 CSVs).
- Imbalanced mixed-class training: archived Exp 14.

## Deliberately not in scope

- **dadapy** (duplicate TwoNN).
- **Re-ranking Historical Exp 1 PCA** after the fact to force a round 90% on Statlog (would invalidate every downstream `data_L*.csv`). Documented miss; archived Exp 13 is the variance-matched alternative.
- **TensorFlow deep learning (archived Exp 20)** — placeholder, out of scope.
