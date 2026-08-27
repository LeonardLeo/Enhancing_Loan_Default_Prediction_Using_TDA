# Late split, no undersample, using both H0 and H1

Folder: `Late_Split_No_Undersample_H0_And_H1`

Public name (figures and paper tables): **Late split, no undersample, using both H0 and H1**. Names come from `utils.process_display_name()`.

- **Split:** late (80/20 on barcode rows after Ripser)
- **Undersample:** no — both class pools keep their original sizes
- **PCA:** MinMax + PCA on the full processed table
- **Homology:** both H0 and H1 (24 barcode statistics)

## Live experiments

1. `1_PH_Default_Parameters` — builds landmarks + Ripser barcodes + default classifiers
2. `2_PH_Tuned_Parameters` — GridSearchCV on experiment-1 matrices
3. `6_Sampling_Ratio_Audit` — class counts, snapshot-size percents, number of snapshots (no Ripser)
4. `8_Null_Hypothesis_Algorithm2` — permutation test on this process's barcode vectors
5. `9_Revised_Snapshot_Protocol` — fixed points per snapshot; default 60 training / 15 test snapshots

The matching just-H0 process is `Late_Split_No_Undersample_H0/`. Nested extras live under `Archives/Four_Arm_Nested_Experiments/No_Undersampling/`.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`).
