# Experiment 19 — Can a straight line on H0 stats beat the five classifiers?

## In one sentence

Linear regression is a blunt instrument. We keep only the H0 columns (`g*_0`, 12 numbers) from Experiment 3 and threshold the predicted value into a class. Predictions are clipped to {0, 1} so a wild linear score cannot create a fake third class.

## Who this is for

If a straight line on “connected-component” stats already matches SVM, the loops (H1) were not doing the work. If the line is near 0.5 accuracy, you need non-linear models or the H1 features.

Statlog’s original script recomputed H0-only barcodes (another 500 Ripser jobs). The new-dataset scripts **do not**: they slice H0 columns out of the existing Exp 3 table.

## Datasets

All six. **Prerequisite:** Experiment 3 `data_L*.csv`.

## What we found (PKDD)

- L10 linear accuracy **0.50**, F1 **0.47** — a coin flip.
- L20 linear accuracy **0.62**, F1 **0.61** — a real lift, still below a good non-linear model.

## Results

`6_Results/19_Linear_Regression_For_Prediction/{Folder}/model_results.pkl`
