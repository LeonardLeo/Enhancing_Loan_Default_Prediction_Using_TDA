# Statistical approach — stage-by-stage flow

This is the “at what stage did we do what?” map for the current buckets. Numbers live in `docs/Statistical_Experiments_24_27_Results.md`. Design knobs live in `docs/Design_Decisions.md`. English names for snapshot quantities are used throughout; see `docs/Notation.md`. Folder map: `docs/Repository_Layout.md`.

Older notes still say Experiments 23–28. Those labels are **historical checklist numbers**, not live folders at the root of `5_Experiments/`. Live homes:

| Historical checklist # | Live path |
|------------------------|-----------|
| Paper Exp 3 (builds barcodes) | `Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters` |
| Exp 23 (Protocol B) | `Early_Split_TDA/1_PH_Default_Parameters` |
| Exp 24 | `{TDA arm}/6_Sampling_Ratio_Audit` |
| Exp 25 | `{TDA arm}/7_Snapshot_Mean_Variance` |
| Exp 26 | `Statistics/1_Intrinsic_Dimension_Estimation` |
| Exp 27 | `{TDA arm}/8_Null_Hypothesis_Algorithm2` |
| Exp 28 | `{TDA arm}/9_Revised_Snapshot_Protocol` |
| Sample-size study (13/08/2026) | `Snapshot_Sample_Size/` |

Active TDA arms (same experiments 1–9 inside each):
`Historical_Late_Split_Balanced_TDA`, `Early_Split_TDA`, `No_Undersampling`, `Early_Split_TDA_And_No_Undersampling`.

```text
processed table
      |
      |  Arm Exp 1  (builds that arm's TDA artefacts)
      |  protocol knobs → landmarks → Ripser → data_L*.csv
      |
      +-- Arm Exp 6  sampling-ratio audit (class counts + snapshot-size percents + number of snapshots; no barcodes)
      +-- Statistics/1  intrinsic dimension (no barcodes; protocol-independent)
      +-- Arm Exp 7  reads that arm's data_L*.csv
      +-- Arm Exp 8  reads that arm's data_L*.csv
      |
      +-- Arm Exp 9  Revised Snapshot Protocol (fixed points per snapshot, 60 training / 15 test snapshots)
      |
      +-- Snapshot_Sample_Size  dated 13/08/2026 (items 1, 2, and 4; item 3 is this study, not a third grid)
```

Barcode **consumers** (arm experiments 2–5, 7–8) **read** that arm's experiment-1 `data_L*.csv`. They must not regenerate 500 Ripser jobs.

---

## Stage 0 — Processed table (before any TDA)

`1_Data/ingest_registry_datasets.py` (four new tables) or the historical Excel processors (Default of Credit Card Client, Statlog) write:

`1_Data/Processed_Datasets/{Folder}/processed_data.{xlsx|csv}`

This is the common starting file for tabular Default Parameters and for every TDA arm.

---

## Stage 1 — Historical arm experiment 1: original snapshot protocol

**When:** first TDA run on a dataset. Everything statistical later either audits this run or replaces it.

**Live folder:** `5_Experiments/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/`

**Order inside the script:**

1. Load the processed table; encode / fill.
2. MinMax-scale **all rows**, then PCA. (Leaky: scaler/PCA see the whole table. Early Split TDA Exp 1 and arm Exp 9 split first.)
3. Undersample the majority so both classes have the minority class count.
4. For each snapshot-size percent, draw 500 subsets per class, each of size floor(minority class count × percent / 100).
5. Ripser: H0 + H1. Twelve barcode statistics per homology dimension → **24 numbers** per snapshot.
6. Write matrices:

```text
1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/{Folder}/data_L{percent}.csv
```

1,000 rows (500 + 500) × 24 features + `label`.

7. Train five default classifiers on an 80/20 split of those rows (a *barcode-row* split, not a customer split).

**Percents and PCA rank** differ by dataset on purpose — see `docs/Design_Decisions.md`. They are not copy-paste leftovers.

**What this stage is not:** an independent sample of 1,000 loans. Arm experiment 6 exists because 500 snapshots remix the same minority-class people.

---

## Stage 2 — Arm experiment 6: sampling-ratio audit (historical Exp 24)

**When:** after the minority class count and the snapshot-size percent are known. **Does not need** barcodes or Ripser.

**Question:** with the historical 500 snapshots, how many times is a typical customer drawn?

```text
reuse ratio = (points per snapshot × number of snapshots) / minority class count
suggested snapshot count ≈ round(minority class count / points per snapshot)    # reuse ≈ 1
```

**Finding:** the reuse ratio is 25–300 on every table. Historical 500 snapshots fail the “reuse near 1” checklist everywhere. This is why arm experiment 9 exists.

---

## Stage 3 — Statistics experiment 1: intrinsic dimension (historical Exp 26)

**When:** on the processed table. **Does not need** Historical Exp 1 artefacts. Can run in parallel with arm experiment 6.

**Live folder:** `5_Experiments/Statistics/1_Intrinsic_Dimension_Estimation/`

**Question:** how many degrees of freedom does the cloud really have? PCA rank is not intrinsic dimension.

**Order:**

1. Scale the encoded features.
2. Estimate intrinsic dimension **before PCA** (geometry of the credit table).
3. Fit the **same PCA Historical Exp 1 uses** (7 / 10 / 15) and estimate intrinsic dimension **after PCA** (geometry Ripser sees).
4. Also record how many components would hit ~90% variance.

Headline: Two-NN (hand-coded Facco formula **and** `skdim.id.TwoNN`). Secondary: Levina–Bickel, MiND_ML, lPCA.

Use **after-PCA Two-NN** when talking about snapshot-size theory. Use **before-PCA Two-NN** when talking about the dataset itself. Both go in the paper. Details: `docs/Design_Decisions.md` §3.

---

## Stage 4 — Arm experiment 7: barcode-feature stability (historical Exp 25)

**When:** **after** that arm's Exp 1 `data_L*.csv` exist.

**Question:** do the 24 barcode numbers jump around from snapshot to snapshot, or are they a stable class fingerprint?

Mean and sample variance of each column over the 1,000 rows. The 24-vector of means is stored as `lambda_bar_proxy` — a cheap stand-in for a persistence-landscape average, **not** the landscape itself (Chazal et al.).

Small variance + different class means ⇒ stable fingerprint. Large variance ⇒ successive snapshots disagree, which is the remix-noise arm experiment 6 already flagged.

---

## Stage 5 — Arm experiment 8: do the two classes differ? (historical Exp 27, Algorithm 2)

**When:** **after** that arm's Exp 1 `data_L*.csv` exist. Independent of arm experiment 7 except that both read the same files.

**Question:** if we shuffle labels, is the gap between default and non-default barcode rows still surprising?

Robinson & Turner Algorithm 2 (arXiv:1310.7467) on **24-D barcode vectors**, not bottleneck/Wasserstein on raw diagrams. Cap 100 snapshots per class, `B = 200` permutations, `(p, q) ∈ {(2,2), (1,1), (2,1)}`.

Tiny p-value ⇒ the two clouds are probably not the same process. It does **not** by itself mean a classifier will generalise to new customers (see arm experiment 6 reuse).

---

## Stage 6 — Arm experiment 9: revised protocol (historical Exp 28)

**When:** after arm experiment 6 / Statistics experiment 1 have told us that 500 snapshots over-reuse and what intrinsic dimension looks like.

**Does not consume** Historical Exp 1 `data_L*.csv`. Starts from the processed table. Lives as `9_Revised_Snapshot_Protocol` in every TDA bucket. The original meeting protocol (early split + no undersample) is `Early_Split_TDA_And_No_Undersampling`; the other three arms reuse the same engine with that arm's split/undersample knobs.

**Stages inside arm experiment 9:**

1. **design** — estimate intrinsic dimension (skdim TwoNN / MLE / lPCA / DANCo when cheap), choose a joint points-per-snapshot value, print the reuse ratio.
2. **split_ml** — apply the arm's split + undersample factory; draw independent snapshots (60 training / 15 test by default); fit models.
3. **full_ml** — Default of Credit Card Client extra arm that skips the split (documented in the launcher).

Points per snapshot is an absolute count, not a percent of class size. PCA ranks still come from `DatasetConfig` / `docs/Design_Decisions.md`.

---

## Stage 7 — Snapshot sample size (13/08/2026)

**Live folder:** `5_Experiments/Snapshot_Sample_Size/`

A separate factorial on **all four protocol arms**. Items 1 and 2 are different x-factors (snapshot count vs points per snapshot), not the same sweep twice; item 4 is families of cloud size. Item 3 is this study, not a third grid. Shared Ripser pools sit in `0_Shared_Pools`. Thesis-length narrative: `5_Experiments/Snapshot_Sample_Size/README.md`.

---

## Related stages that are *not* 24–27 but get asked about in the same breath

| Experiment | Live path | What |
|------------|-----------|------|
| **Arm Exp 3** | `{TDA arm}/3_H0_Only` | Repeat models on **H0-only** barcode columns. |
| **Arm Exp 4** | `{TDA arm}/4_Dropping_Correlated_Barcode_Statistics_Columns` | Drop correlated barcode columns. |
| **Archived Exp 13** | `Archives/13_Similar_Variance_Retained_After_PCA` | Match **PCA variance** across datasets instead of matching component **count**. |
| **Archived Exp 16 / 18** | `Archives/16_…` and `Archives/18_…` | Sweep PCA rank. |
| **Arm Exp 5** | `{TDA arm}/5_Linear_Regression_For_Prediction` | Linear regression on the same barcode table. |
| **Early Split TDA Exp 1** | `Early_Split_TDA/1_PH_Default_Parameters` | Historical snapshot percents, but train/test customers **before** PCA / landmarks (Protocol B). |

---

## One-sentence cheat sheet

| Historical # | Live experiment | Needs barcodes? | Asks |
|--------------|-----------------|-----------------|------|
| 3 | Historical arm Exp 1 | Builds them | Historical TDA fingerprints + default models |
| 24 | Arm Exp 6 | No | Are we over-reusing people? |
| 26 | Statistics Exp 1 | No | What is intrinsic dimension, before and after PCA? |
| 25 | Arm Exp 7 | Yes (read) | Are barcode numbers stable? |
| 27 | Arm Exp 8 | Yes (read) | Do the two classes differ under Algorithm 2? |
| 28 | Arm Exp 9 | Builds new ones | Honest points per snapshot, small snapshot counts |
| — | Snapshot_Sample_Size | Builds shared pools | Item 1: F1 vs number of snapshots (cloud size fixed). Item 2: F1 vs points per snapshot (always 60 snapshots). Item 4: families of cloud size |
