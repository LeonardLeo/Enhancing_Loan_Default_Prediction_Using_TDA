# Late_Split_No_Undersample_H0_And_H1 — 6_Sampling_Ratio_Audit

## Protocol knobs

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| Snapshot size | points per snapshot = floor(that class's count × snapshot size percent / 100) on the unbalanced pool |
| Number of snapshots | 500 (dataset historical n_files) |

No Ripser. Scores reuse ratio = (points per snapshot × number of snapshots) / class count from the protocol's class pools, snapshot-size percents, and 500 snapshots.

Same Exp 3 PCA ranks and snapshot-size percents (`docs/Design_Decisions.md`, `utils.DatasetConfig`):

| Dataset | Snapshot size as percent of the class | PCA rank | Number of snapshots |
|---------|-------------------|----------|-----|
| Default_Of_Credit_Card_Client_Data | L5 / L15 | 7 | 500 |
| Statlog_German_Credit_Data | L30 / L60 | 15 | 500 |

## Artefacts

```
1_Data/Landmark_Sets/Late_Split_No_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/
1_Data/Barcode_Statistics/Late_Split_No_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/
1_Data/TDA_Datasets/Late_Split_No_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/data_L*.csv
6_Results/Late_Split_No_Undersample_H0_And_H1/6_Sampling_Ratio_Audit/{Dataset}/
```

Tuned models, the sampling-ratio audit, and Algorithm 2 read this process's experiment-1 matrices. They must not start 500 Ripser jobs.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/Late_Split_No_Undersample_H0_And_H1/6_Sampling_Ratio_Audit/<Dataset>/<script>.py
```


