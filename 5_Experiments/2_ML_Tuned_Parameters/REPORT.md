# Experiment 2 — Same models as Experiment 1, but we tune the knobs

## In one sentence

Does a grid search on the **original** features close the gap that TDA might otherwise appear to win?

## Who this is for

Off-the-shelf models (Experiment 1) are a lower bound. If tuned logistic / forests already match TDA, topology is not buying much — we were just under-fitting the table.

## Datasets

Same six tables as Experiment 1. Run Experiment 1 first: this script **reuses** the train/test matrices it saved.

## What we do (in order)

1. Load resampled train features and the held-out test set from Experiment 1.
2. Scale again (same MinMax pattern).
3. For each of the five classifiers, run `GridSearchCV` with stratified 5-fold CV, scoring F1.
4. Evaluate the winning estimator on the test set.

## What we look for

Whether tuned F1 on raw features already matches Experiment 3 / 4. Quote both.

## Results

`6_Results/2_ML_Tuned_Parameters/{Folder}/model_results.pkl` — best params plus test metrics per model.

## How to read the numbers

Accuracy is misleading when few people default. Quote **F1** and **recall**. These scores are on **customers**, not barcode snapshots.
