# Historical_Late_Split_Balanced_TDA

late (80/20 on barcode rows after Ripser)

- **Undersample:** yes — majority downsampled to minority count before landmarks
- **PCA:** MinMax + PCA on the FULL processed table (historical / slightly leaky)
- **t:** t = floor(n1 * L / 100) with n1 = n2 = minority count
- **l:** 500 (dataset historical n_files)
- **PCA ranks / L percents:** same as historical Exp 3 (`docs/Design_Decisions.md`). Not Exp 28's revised t/l.

## Experiments in this arm (1-based)

1. `1_PH_Default_Parameters` — builds landmarks + Ripser barcodes + default classifiers
2. `2_PH_Tuned_Parameters` — consumes experiment 1 (GridSearchCV)
3. `3_H0_Only` — consumes experiment 1 (H0 columns only)
4. `4_Dropping_Correlated_Barcode_Statistics_Columns` — consumes experiment 1
5. `5_Linear_Regression_For_Prediction` — consumes experiment 1
6. `6_Sampling_Ratio_Audit` — class counts + L + l (no Ripser)
7. `7_Snapshot_Mean_Variance` — consumes experiment 1
8. `8_Null_Hypothesis_Algorithm2` — consumes experiment 1
9. `9_Revised_Snapshot_Protocol` — Exp 28 snapshot rules (fixed absolute t, l_train/l_test = 60/15). Not the historical L-percent / l=500 pipeline.

Not in this arm: tabular Default Parameters (see `Default_Parameters/`), protocol-independent geometry (see `Statistics/`), or retired paper/exploratory work (see `Archives/`).
