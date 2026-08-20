# Early_Split_TDA_And_No_Undersampling — 5_Linear_Regression_For_Prediction

## Protocol knobs

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | no — full class pools inside each split |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| Snapshot size | points per snapshot = floor(class count × snapshot size percent / 100) on that split's available pool |
| Number of snapshots | 500 (dataset historical n_files) |

CONSUMER. Linear regression on the H0 slice of experiment 1.

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
1_Data/Landmark_Sets/Early_Split_TDA_And_No_Undersampling/1_PH_Default_Parameters/{Dataset}/
1_Data/Barcode_Statistics/Early_Split_TDA_And_No_Undersampling/1_PH_Default_Parameters/{Dataset}/
1_Data/TDA_Datasets/Early_Split_TDA_And_No_Undersampling/1_PH_Default_Parameters/{Dataset}/data_L*.csv
6_Results/Early_Split_TDA_And_No_Undersampling/5_Linear_Regression_For_Prediction/{Dataset}/
```

Consumers 2–5 and 7–8 read this arm's experiment-1 matrices. They must not start 500 Ripser jobs.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA_And_No_Undersampling/5_Linear_Regression_For_Prediction/<Dataset>/<script>.py
```


