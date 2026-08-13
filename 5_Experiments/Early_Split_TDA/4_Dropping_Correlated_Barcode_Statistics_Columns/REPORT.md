# Early_Split_TDA — 4_Dropping_Correlated_Barcode_Statistics_Columns

## Protocol knobs

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | yes — independently inside the train pool and the test pool |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| Snapshot size | t = floor(n_class * L / 100) on the undersampled pool of that split |
| Snapshot count `l` | 500 (dataset historical n_files) |

CONSUMER. Drops correlated barcode columns (threshold 0.80) from experiment 1, then retrains.

Same Exp 3 PCA ranks and landmark percents (`docs/Design_Decisions.md`, `utils.DatasetConfig`):

| Dataset | Landmark percents | PCA rank | `l` |
|---------|-------------------|----------|-----|
| Default_Of_Credit_Card_Client_Data | L5 / L15 | 7 | 500 |
| Statlog_German_Credit_Data | L30 / L60 | 15 | 500 |
| PKDD_Czech_Financial | L10 / L20 | 10 | 500 |
| Polish_Bankruptcy_3Year | L10 / L20 | 10 | 500 |
| Taiwan_Bankruptcy | L10 / L20 | 10 | 500 |
| South_German_Credit | L10 / L20 | 10 | 500 |

## Artefacts

```
1_Data/Landmark_Sets/Early_Split_TDA/1_PH_Default_Parameters/{Dataset}/
1_Data/Barcode_Statistics/Early_Split_TDA/1_PH_Default_Parameters/{Dataset}/
1_Data/TDA_Datasets/Early_Split_TDA/1_PH_Default_Parameters/{Dataset}/data_L*.csv
6_Results/Early_Split_TDA/4_Dropping_Correlated_Barcode_Statistics_Columns/{Dataset}/
```

Consumers 2–5 and 7–8 read this arm's experiment-1 matrices. They must not start 500 Ripser jobs.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA/4_Dropping_Correlated_Barcode_Statistics_Columns/<Dataset>/<script>.py
```

Or the folder launcher `run_all.py`.
