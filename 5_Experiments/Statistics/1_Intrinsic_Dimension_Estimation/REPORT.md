# Experiment 26 — How many degrees of freedom does the cloud really have?

## In one sentence

PCA rank (7, 10, or 15 components) is **not** intrinsic dimension `b`. We estimate `b` **before PCA** (the credit table) and **after PCA** (the space Ripser samples), with Two-NN as the headline.

## Who this is for

A photograph of a face is thousands of pixels but only a handful of “real” degrees of freedom (pose, expression, lighting). Intrinsic dimension asks: **how many knobs is this dataset actually turning?** If that number is near the number of PCA axes we kept, snapshot-size theory for TDA is on thin ice.

**Before vs after — we need both.** Details in `docs/Design_Decisions.md` §3.

| Estimate | Question it answers |
|----------|---------------------|
| Before PCA | Geometry of the scaled (encoded) loan / bankruptcy table. Independent of our PCA choice. |
| After PCA | Geometry of the box Exp 3 actually feeds Ripser. This is the `b` snapshot-size theory should use. |

Using only the after number hides whether PCA flattened a high-d table or an already-low-d one (and `b` cannot exceed the rank we kept). Using only the before number sizes snapshots for a space Ripser never sees.

## Datasets (all six)

This experiment does **not** need barcodes.

`5_Experiments/Statistics/1_Intrinsic_Dimension_Estimation/<Dataset>/run_intrinsic_dimension.py`

## What we do (in order)

1. Load processed tabular features; drop the target column.
2. Median-fill numeric holes; dummy-encode leftover categoricals.
3. MinMax-scale.
4. Estimate `b` on that scaled table (**before PCA**): hand-coded Two-NN + Levina–Bickel, and scikit-dimension TwoNN / MLE / MiND_ML / lPCA ([arXiv:2109.02596](https://arxiv.org/abs/2109.02596)).
5. Fit the **same PCA Experiment 3 uses** → estimate `b` again (**after PCA**).
6. Record how many components would be needed to keep ~90% variance (the design target for the new tables).
7. Write one CSV row plus a pickle of the full estimator dicts.

Large tables are capped at 5,000 points so Two-NN stays affordable.

Hand-coded Two-NN is the transparent Facco formula. skdim is the published package on the same matrix. We did **not** add dadapy (same Two-NN estimator, extra dependency). See `docs/Methodology_Checklist_06_08_2026.md`.

## What we found (Two-NN is the number to quote)

| Dataset | PCA components in Exp 3 | Variance kept by that PCA | PCs needed for 90% | Two-NN **before** PCA | Two-NN **after** PCA |
|---------|-------------------------|---------------------------|-------------------:|----------------------:|---------------------:|
| DCCCD | 7 | 94.0% | 6 | 3.95 | **2.81** |
| Statlog German Credit | 15 | 89.3% | 16 | 5.34 | **4.06** |


None of the Two-NN-after-PCA values sit at 7. The “is b ≈ 7?” alarm from the snapshot-size discussion does **not** fire on these six tables.

Levina–Bickel with k=10 came out much smaller than Two-NN on these tables. Treat it as a secondary check, not the headline. The CSV also has `skdim_TwoNN_*`, `skdim_MLE_*`, `skdim_MiND_ML_*`, `skdim_lPCA_*`, and `n_components_for_90pct`.


Polish `skdim MLE = 0` and `MiND_ML = 1` collapse on that table (likely near-duplicate ratio rows after median fill). Hand-coded Two-NN does **not** collapse (5.73 / 4.35). Do not headline the collapsed package numbers; they are why we keep the Facco formula in `utils.py`.

## Where the files live

`6_Results/Statistics/1_Intrinsic_Dimension_Estimation/{Folder}/intrinsic_dimension_estimates.csv`


## Bucket

This folder now lives under `5_Experiments/Statistics/`. Headline estimates are before-PCA and after the **same Exp 3 PCA rank**. Early-split (train-only) PCA is a sensitivity, not the headline.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.
