# No_Undersampling — 4_Dropping_Correlated_Barcode_Statistics_Columns

## Protocol knobs

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| Snapshot size | points per snapshot = floor(that class's count × snapshot size percent / 100) on the unbalanced pool |
| Number of snapshots | 500 (dataset historical n_files) |

CONSUMER. Drops correlated barcode columns (threshold 0.80) from experiment 1, then retrains.

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
1_Data/Landmark_Sets/No_Undersampling/1_PH_Default_Parameters/{Dataset}/
1_Data/Barcode_Statistics/No_Undersampling/1_PH_Default_Parameters/{Dataset}/
1_Data/TDA_Datasets/No_Undersampling/1_PH_Default_Parameters/{Dataset}/data_L*.csv
6_Results/No_Undersampling/4_Dropping_Correlated_Barcode_Statistics_Columns/{Dataset}/
```

Consumers 2–5 and 7–8 read this arm's experiment-1 matrices. They must not start 500 Ripser jobs.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. `run.py` is an optional convenience launcher and is not the method document.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/No_Undersampling/4_Dropping_Correlated_Barcode_Statistics_Columns/<Dataset>/<script>.py
```

Or the folder launcher `run_all.py`.
