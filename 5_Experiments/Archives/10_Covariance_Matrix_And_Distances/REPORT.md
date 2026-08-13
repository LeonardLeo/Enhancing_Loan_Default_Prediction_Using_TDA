# Experiment 10 — Distances between barcode snapshots

## In one sentence

If we treat each snapshot as a point in 24-D, how far is it from the default cloud vs the non-default cloud? A nearest-centroid rule is a very simple classifier.

## Who this is for

You do not need covariance theory. We ask: “does this snapshot sit closer to the average default snapshot or the average non-default snapshot?” Mean, farthest, and random centroids are three ways of picking that average.

## Datasets

All six. **Prerequisite:** Experiment 3 `data_L*.csv`. No landmark rebuild.

## What we do

1. Load L10/L20 (or L5/L15, L30/L60) barcode tables.
2. Build a distance view (all pairs, and class-centroid views).
3. Score nearest-centroid predictions with accuracy / precision / recall / F1.
4. Save pickles under `6_Results/Archives/10_Covariance_Matrix_And_Distances/{Folder}/L{percent}/`.

## How to read the result

If mean-centroid F1 is near 0.5 on a balanced table, the two class averages sit on top of each other. If it is high, even this naïve rule can tell the clouds apart — a lower bound for what SVM / forests might do in Experiment 3.
