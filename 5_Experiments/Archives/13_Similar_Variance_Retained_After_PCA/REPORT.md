# Experiment 13 — Fair PCA *variance* across datasets

## In one sentence

Statlog keeps ~89% variance with 15 components; DCCCD keeps ~94% with 7. Matching the **component count** is not the same as matching the **variance kept**.

## Who this is for

If we give one table 15 axes and another 7, we are not running the same experiment. Here we pick PCA rank so each table keeps about the same fraction of variance, then rebuild landmarks.

## Why this experiment exists (vs Exp 3)

Experiment 3 on the four new tables used a **shared rank of 10** so Ripser boxes were comparable. The *design target* was ~90% variance; 10 was chosen because Taiwan landed nearest that (~88%). Polish (~83%), South German (~78%), and especially PKDD’s dummy-expanded table (~46.5%) **miss**. This folder is the honest alternative: match the **percentage of variance**, let the component count float.

DCCCD (7 comps, ~94%) and Statlog (15 comps, ~89%) were already near the target by paper choice. They still belong in this comparison so “matched variance” is defined against the original pair, not only against Taiwan.

Do **not** silently patch Exp 3 to these ranks after the fact — every `data_L*.csv` consumer would be invalidated. This experiment writes its **own** artefacts.

Full rationale: `docs/Design_Decisions.md` §2.

## Datasets

All six. Compare with Experiment 16 / 18 (component *sweeps*, F1 vs rank) and Experiment 26 (intrinsic dimension, which does not rebuild barcodes, and now also reports `n_components_for_90pct`).

## What we look for

Does TDA F1 move when we stop over/under-compressing a table relative to Statlog? If PKDD F1 jumps once we keep ~90% instead of 46%, the Exp 3 PKDD number was a compression artefact, not a TDA result.

## Results

`6_Results/Archives/13_Similar_Variance_Retained_After_PCA/{Folder}/`
