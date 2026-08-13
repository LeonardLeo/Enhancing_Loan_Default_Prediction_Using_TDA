# Experiment 4 — TDA fingerprints with tuned classifiers

## In one sentence

Same barcode matrices as Experiment 3, but we grid-search the five models (as Experiment 2 does for raw features).

## Who this is for

A fair “does topology win?” comparison is **tuned raw features (Exp 2)** vs **tuned barcode features (this folder)**. Default-vs-default (Exp 1 vs Exp 3) can fool you if one side was under-fit.

## Datasets

All six. **Prefer** loading existing Experiment 3 `data_L*.csv` rather than rebuilding 500 landmarks. If a dataset script still rebuilds them, treat that as the historical Statlog behaviour, not something to copy into every folder.

## What we look for

Tuned-TDA F1 minus tuned-raw F1. A small gap means topology is not the story; a large gap (in either direction) is.

## Results

`6_Results/4_PH_Tuned_Parameters/{Folder}/`
