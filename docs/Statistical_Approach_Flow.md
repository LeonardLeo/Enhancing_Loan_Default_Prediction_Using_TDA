# Statistical approach — stage-by-stage flow

This is the “at what stage did we do what?” map for the four TDA protocol arms. Numbers live in `docs/Statistical_Experiments_24_27_Results.md`. Design knobs live in `docs/Design_Decisions.md`.

Active TDA arms (same experiments 1–9 inside each):
`Historical_Late_Split_Balanced_TDA`, `Early_Split_TDA`, `No_Undersampling`, `Early_Split_TDA_And_No_Undersampling`.

```text
processed table
      |
      |  Arm Exp 1  (builds that arm's TDA artefacts)
      |  protocol knobs → landmarks → Ripser → data_L*.csv
      |
      +-- Arm Exp 6  sampling-ratio audit (class counts + L + l; no barcodes)
      +-- Statistics/1  intrinsic dimension (no barcodes; protocol-independent)
      +-- Arm Exp 7  reads that arm's data_L*.csv
      +-- Arm Exp 8  reads that arm's data_L*.csv
      |
      +-- Arm Exp 9  Revised Snapshot Protocol (fixed t, l_train/l_test=60/15)
```

Barcode **consumers** (arm experiments 2–5, 7–8) **read** that arm's experiment-1 `data_L*.csv`. They must not regenerate 500 Ripser jobs.

---

## Stage 0 — Processed table (before any TDA)

`1_Data/ingest_registry_datasets.py` (four new tables) or the historical Excel processors (DCCCD, Statlog) write:

`1_Data/Processed_Datasets/{Folder}/processed_data.{xlsx|csv}`

This is the common starting file for Exp 1 (tabular ML) and Exp 3 (TDA).

---

## Stage 1 — Experiment 3: historical snapshot protocol

**When:** first TDA run on a dataset. Everything statistical later either audits this run or replaces it.

**Order inside the script:**

1. Load the processed table; encode / fill.
2. MinMax-scale **all rows**, then PCA. (Leaky: scaler/PCA see the whole table. Exp 23 / 28 split first.)
3. Undersample the majority so both classes have size `n1` = minority count.
4. For each landmark percent, draw `l = 500` subsets per class, each of size `t = floor(n1 * L / 100)`.
5. Ripser: H0 + H1. Twelve barcode statistics per homology dimension → **24 numbers** per snapshot.
6. Write Statlog-style matrices:

```text
1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/{Folder}/data_L{percent}.csv
```

1,000 rows (500 + 500) × 24 features + `label`.

7. Train five default classifiers on an 80/20 split of those rows (a *barcode-row* split, not a customer split).

**Percents and PCA rank** differ by dataset on purpose — see `docs/Design_Decisions.md`. They are not copy-paste leftovers.

**What this stage is not:** an independent sample of 1,000 loans. Experiment 24 exists because `l = 500` remixes the same `n1` people.

---

## Stage 2 — Experiment 24: sampling-ratio audit

**When:** after we know `n1` and `L`. **Does not need** barcodes or Ripser.

**Question:** with historical `l = 500`, how many times is a typical customer drawn?

```text
R = (t * l) / n1
l_star ≈ round(n1 / t)    # snapshot count that would make R ≈ 1
```

**Finding:** `R` is 25–300 on every table. Historical `l = 500` fails the “R near 1” checklist everywhere. This is why Experiment 28 exists.

---

## Stage 3 — Experiment 26: intrinsic dimension (before *and* after PCA)

**When:** on the processed table. **Does not need** Exp 3 artefacts. Can run in parallel with Exp 24.

**Question:** how many degrees of freedom does the cloud really have? PCA rank is not `b`.

**Order:**

1. Scale the encoded features.
2. Estimate `b` **before PCA** (geometry of the credit table).
3. Fit the **same PCA Exp 3 uses** (7 / 10 / 15) and estimate `b` **after PCA** (geometry Ripser sees).
4. Also record how many components would hit ~90% variance.

Headline: Two-NN (hand-coded Facco formula **and** `skdim.id.TwoNN`). Secondary: Levina–Bickel, MiND_ML, lPCA.

Use **after-PCA Two-NN** when talking about snapshot-size theory. Use **before-PCA Two-NN** when talking about the dataset itself. Both go in the paper. Details: `docs/Design_Decisions.md` §3.

---

## Stage 4 — Experiment 25: barcode-feature stability

**When:** **after** Exp 3 `data_L*.csv` exist.

**Question:** do the 24 barcode numbers jump around from snapshot to snapshot, or are they a stable class fingerprint?

Mean and sample variance of each column over the 1,000 rows. The 24-vector of means is stored as `lambda_bar_proxy` — a cheap stand-in for a persistence-landscape average, **not** the landscape itself (Chazal et al.).

Small variance + different class means ⇒ stable fingerprint. Large variance ⇒ successive snapshots disagree, which is the remix-noise Experiment 24 already flagged.

---

## Stage 5 — Experiment 27: do the two classes differ? (Algorithm 2)

**When:** **after** Exp 3 `data_L*.csv` exist. Independent of Exp 25 except that both read the same files.

**Question:** if we shuffle labels, is the gap between default and non-default barcode rows still surprising?

Robinson & Turner Algorithm 2 (arXiv:1310.7467) on **24-D barcode vectors**, not bottleneck/Wasserstein on raw diagrams. Cap 100 snapshots per class, `B = 200` permutations, `(p, q) ∈ {(2,2), (1,1), (2,1)}`.

Tiny p-value ⇒ the two clouds are probably not the same process. It does **not** by itself mean a classifier will generalise to new customers (see Exp 24 reuse).

---

## Stage 6 — Experiment 28: revised protocol (replaces Stage 1 for “what we would do next”)

**When:** after Exp 24 / 26 have told us that `l = 500` over-reuses and what `b` looks like.

**Does not consume** Exp 3 `data_L*.csv`. Starts from the processed table. Lives as arm experiment **9** in every TDA bucket (`9_Revised_Snapshot_Protocol`). The original meeting protocol (early split + no undersample) is `Early_Split_TDA_And_No_Undersampling`; the other three arms reuse the same engine with that arm's split/undersample knobs.

**Stages inside Exp 28 / arm Exp 9:**

1. **design** — estimate `b` (skdim TwoNN / MLE / lPCA / DANCo when cheap), choose a joint snapshot size `t`, print reuse `R`.
2. **split_ml** — apply the arm's split + undersample factory; draw independent snapshots (`l_train = 60`, `l_test = 15` by default); fit models.
3. **full_ml** — DCCCD-only extra arm that skips the split (documented in the launcher).

`t` is an absolute count, not a percent of class size. PCA ranks still come from `DatasetConfig` / `docs/Design_Decisions.md`.

---

## Related stages that are *not* 24–27 but get asked about in the same breath

| Experiment | When | What |
|------------|------|------|
| **Exp 6** | After Exp 3 | Repeat models on **H0-only** barcode columns (the “previous technique of splitting barcode statistics”). |
| **Exp 11** | After Exp 3 | Drop correlated barcode columns (variance rule vs target rule). |
| **Exp 13** | Rebuilds landmarks | Match **PCA variance** across datasets instead of matching component **count**. |
| **Exp 16 / 18** | Rebuilds landmarks | Sweep PCA rank; named after DCCCD / Statlog historically. |
| **Exp 19** | After Exp 3 | Linear regression on the same barcode table (and H0 slice). |
| **Exp 23** | Rebuilds with split first | Historical protocol but train/test customers **before** PCA / landmarks. |

---

## One-sentence cheat sheet

| # | Needs barcodes? | Asks |
|---|-----------------|------|
| 3 | Builds them | Historical TDA fingerprints + default models |
| 24 | No | Are we over-reusing people? |
| 26 | No | What is `b`, before and after PCA? |
| 25 | Yes (read) | Are barcode numbers stable? |
| 27 | Yes (read) | Do the two classes differ under Algorithm 2? |
| 28 | Builds new ones | Honest `t`, small `l`, split first |
