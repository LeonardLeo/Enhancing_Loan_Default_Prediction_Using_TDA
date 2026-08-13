# Methodology checklist (06/08/2026) — what is in the repo

Team list from 6 August 2026, mapped onto the implementation. Stage order is in `docs/Statistical_Approach_Flow.md`. Landmark / PCA / ID rationale is in `docs/Design_Decisions.md`.

---

## 1. scikit-dimension (arXiv:2109.02596)

**Done.** Package: `scikit-dimension` (`skdim`) in `requirements.txt`.

| Where | Estimators |
|-------|------------|
| Exp 26 `utils.estimate_intrinsic_dimension_skdim` | TwoNN, MLE (Levina–Bickel, k=20), MiND_ML, lPCA |
| Exp 28 `protocol_lib.estimate_intrinsic_dimensions` | TwoNN, MLE, lPCA, MiND_ML, DANCo when n ≤ 800 |

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
| **Exp 6** | H0-only columns (drop the 12 H1 stats). |
| **Exp 19** | Linear model on the full 24-D table and on the H0 slice from Exp 3. |
| **Exp 23** | Split **customers** before PCA / landmarks (protocol split, not column split). |
| **Exp 11** | Drop correlated barcode columns (a different split: keep / drop by correlation). |

If the intended repeat is “rerun Exp 24–27 on H0-only matrices”, that is a cheap follow-up (same scripts, point them at Exp 6 CSVs). It is **not** missing infrastructure.

---

## 5. Play with number of snapshots and points per snapshot

**Done, in two layers.**

| Layer | What it does |
|-------|----------------|
| **Exp 24** | Audits historical `l = 500` vs `l_star = round(n1 / t)`. Does not rebuild Ripser. |
| **Exp 28** | Rebuilds with a **fixed t**, default `l_train = 60` / `l_test = 15`, no undersampling, customer split first. |

Playing with `t` and `l` on the historical percent grid (L10 vs L20, L5 vs L15, …) is already Exp 3’s two-percent design plus Exp 25/27 reading both files.

---

## 6. Focus on intrinsic dimensionality

**Done.** Exp 26 (all six datasets, before **and** after PCA, hand-coded + skdim). Exp 28 **design** stage uses `b` to talk about snapshot size. The “is `b ≈ 7`?” check uses **after-PCA Two-NN**.

---

## 7. A section for statistics

**Done, now in three documents:**

| Document | Role |
|----------|------|
| `docs/Statistical_Approach_Flow.md` | When each stage runs; what it needs |
| `docs/Statistical_Experiments_24_27_Results.md` | Worked numbers for all six datasets |
| `docs/Design_Decisions.md` | Why knobs differ (L, PCA, ID before/after) |

Plus `REPORT.md` in every experiment folder (24–28 also have a short per-dataset report).

---

## Not missing (already in the pipeline, easy to confuse with the list)

- Early train/test split: Exp 23 and Exp 28.
- Variance-matched PCA: Exp 13. Component sweeps: Exp 16 / 18.
- Mapper shape of barcode space: Exp 21 (consumes Exp 3 CSVs).
- Imbalanced mixed-class training: Exp 14.

## Deliberately not in scope

- **dadapy** (duplicate TwoNN).
- **Re-ranking Exp 3 PCA** after the fact to force 90% on PKDD / Polish / South German (would invalidate every downstream `data_L*.csv`). Documented miss; Exp 13 is the variance-matched alternative.
- **TensorFlow deep learning (Exp 20)** — placeholder, out of scope.
