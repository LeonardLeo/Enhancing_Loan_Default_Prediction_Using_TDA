# Early split and undersample, using both H0 and H1 — Experiment 1 (Protocol B)

This folder is the live home of the early customer split that was originally discussed as Experiment 23. It is **not** a top-level `23_Early_…` directory. Code and artefacts live here:

`5_Experiments/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/`  
`6_Results/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/`  
`1_Data/{Landmark_Sets,Barcode_Statistics,TDA_Datasets}/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/`

Numeric hold-out tables: `docs/Experiment_23_Results.md` (kept as a results note; the title there matches this arm).

## Goal

Remove train/test leakage from the TDA pipeline by splitting **customers** before scaler, PCA, and snapshot generation. Majority undersampling still happens, independently inside the train pool and the test pool.

## Protocol knobs

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | yes — independently inside the train pool and the test pool |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| Snapshot size | points per snapshot = floor(class count × snapshot size percent / 100) on the undersampled pool of that split |
| Number of snapshots | 500 (dataset historical n_files) |

This experiment **builds** landmarks and Ripser barcodes. Downstream experiments 2–5 and 7–8 in this arm consume its `data_L*.csv`.

Same historical PCA ranks and snapshot-size percents (`docs/Design_Decisions.md`, `utils.DatasetConfig`):

| Dataset | Snapshot size as percent of the class | PCA rank | Number of snapshots |
|---------|-------------------|----------|-----|
| Default_Of_Credit_Card_Client_Data | L5 / L15 | 7 | 500 |
| Statlog_German_Credit_Data | L30 / L60 | 15 | 500 |

## Order of operations

1. Stratified 80/20 on processed tabular data.
2. Fit MinMaxScaler + PCA on **train only**; transform train and test.
3. Undersample-balance **within** each split.
4. Generate snapshots independently for train and test.
5. Compute barcodes independently.
6. Train on train barcodes; evaluate on test barcodes.

## Artefacts

```
1_Data/Landmark_Sets/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/
1_Data/Barcode_Statistics/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/
1_Data/TDA_Datasets/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/data_L*.csv
6_Results/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/
6_Results/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/Visualizations/
```

Tuned models, the sampling-ratio audit, and Algorithm 2 read this process's experiment-1 matrices. They must not start 500 Ripser jobs.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/<Dataset>/default_of_credit_cards_client_PH.py
```

 Ensure the repository root is on `PYTHONPATH` (dataset scripts insert it themselves).

## Cost

With 500 snapshots per class, this regenerates landmarks for **both** train and test. Expect long runtimes, especially on Default of Credit Card Client.

## Related

- Leakage discussion: `docs/Pipeline_Issues_And_Leakage.md`
- Hold-out numbers: `docs/Experiment_23_Results.md`
- Revised protocol (arm experiment 9): `docs/Revised_Snapshot_Protocol_Deep_Report.md`
