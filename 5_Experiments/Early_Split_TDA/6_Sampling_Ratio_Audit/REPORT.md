# Early_Split_TDA — 6_Sampling_Ratio_Audit

## Protocol knobs

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | yes — independently inside the train pool and the test pool |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| Snapshot size | points per snapshot = floor(class count × snapshot size percent / 100) on the undersampled pool of that split |
| Number of snapshots | 500 (dataset historical n_files) |

No Ripser. Scores reuse ratio = (points per snapshot × number of snapshots) / class count from the protocol's class pools, snapshot-size percents, and 500 snapshots.

Same Exp 3 PCA ranks and snapshot-size percents (`docs/Design_Decisions.md`, `utils.DatasetConfig`):

| Dataset | Snapshot size as percent of the class | PCA rank | Number of snapshots |
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
6_Results/Early_Split_TDA/6_Sampling_Ratio_Audit/{Dataset}/
```

Consumers 2–5 and 7–8 read this arm's experiment-1 matrices. They must not start 500 Ripser jobs.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. `run.py` is an optional convenience launcher and is not the method document.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA/6_Sampling_Ratio_Audit/<Dataset>/<script>.py
```

Or the folder launcher `run_all.py`.
