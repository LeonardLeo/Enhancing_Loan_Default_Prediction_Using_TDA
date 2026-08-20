# Historical_Late_Split_Balanced_TDA

late (80/20 on barcode rows after Ripser)

- **Undersample:** yes — majority downsampled to minority count before landmarks
- **PCA:** MinMax + PCA on the FULL processed table (historical / slightly leaky)
- **Points per snapshot:** points per snapshot = floor(minority class count × snapshot size percent / 100); after undersampling both classes have the minority count
- **Number of snapshots:** 500 (dataset historical n_files)
- **PCA ranks / snapshot-size percents:** same as historical Exp 3 (`docs/Design_Decisions.md`). Not Exp 28's revised points-per-snapshot / snapshot-count rule. English names for snapshot quantities are used throughout; the symbol mapping is in `docs/Notation.md`.

## Experiments in this arm (1-based)

1. `1_PH_Default_Parameters` — builds landmarks + Ripser barcodes + default classifiers
2. `2_PH_Tuned_Parameters` — consumes experiment 1 (GridSearchCV)
3. `3_H0_Only` — consumes experiment 1 (H0 columns only)
4. `4_Dropping_Correlated_Barcode_Statistics_Columns` — consumes experiment 1
5. `5_Linear_Regression_For_Prediction` — consumes experiment 1
6. `6_Sampling_Ratio_Audit` — class counts + snapshot-size percents + number of snapshots (no Ripser)
7. `7_Snapshot_Mean_Variance` — consumes experiment 1
8. `8_Null_Hypothesis_Algorithm2` — consumes experiment 1
9. `9_Revised_Snapshot_Protocol` — Exp 28 snapshot rules (fixed points per snapshot, default 60 training snapshots / 15 test snapshots). Not the historical snapshot-size-percent / 500-snapshot pipeline.

Not in this arm: tabular Default Parameters (see `Default_Parameters/`), protocol-independent geometry (see `Statistics/`), or retired paper/exploratory work (see `Archives/`).

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.
