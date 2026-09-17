# Early_Split_And_Undersample_H0 — 6_Sampling_Ratio_Audit

## Protocol knobs

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | yes — independently inside the train pool and the test pool |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| Snapshot size | points per snapshot = floor(class count × snapshot size percent / 100) on the undersampled pool of that split |
| Number of snapshots | 500 (dataset historical n_files) |

No Ripser. Scores expected sampling reuse = (points per snapshot × number of snapshots) / class count from the protocol's class pools and snapshot-size percents. It records both the historical 500-snapshot rule and the approximate one-coverage comparator `ceil(class count / points per snapshot)`. Across the live audit the comparator spans 2–21 snapshots (Default of Credit Card Client: 21/7 at 5%/15%; Statlog: 4/2 at 30%/60%) and can sit just above reuse 1. It is not experiment 9's floor-based reuse-safe cap.

Same Exp 3 PCA ranks and snapshot-size percents (`docs/Design_Decisions.md`, `utils.DatasetConfig`):

| Dataset | Snapshot size as percent of the class | PCA rank | Number of snapshots |
|---------|-------------------|----------|-----|
| Default_Of_Credit_Card_Client_Data | L5 / L15 | 7 | 500 |
| Statlog_German_Credit_Data | L30 / L60 | 15 | 500 |

## Artefacts

```
1_Data/Landmark_Sets/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/
1_Data/Barcode_Statistics/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/
1_Data/TDA_Datasets/Early_Split_And_Undersample_H0/1_PH_Default_Parameters/{Dataset}/data_L*.csv
6_Results/Early_Split_And_Undersample_H0/6_Sampling_Ratio_Audit/{Dataset}/
```

Tuned models, the sampling-ratio audit, and permutation test of class difference read this process's experiment-1 matrices. They must not start 500 Ripser jobs.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_And_Undersample_H0/6_Sampling_Ratio_Audit/<Dataset>/<script>.py
```


