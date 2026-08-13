# Historical_Late_Split_Balanced_TDA — 1_PH_Default_Parameters

## Protocol knobs

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | yes — majority downsampled to minority count before landmarks |
| PCA | MinMax + PCA on the FULL processed table (historical / slightly leaky) |
| Snapshot size | t = floor(n1 * L / 100) with n1 = n2 = minority count |
| Snapshot count `l` | 500 (dataset historical n_files) |

This experiment BUILDS landmarks and Ripser barcodes. Downstream 2–5, 7–8 consume its `data_L*.csv`.

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
1_Data/Landmark_Sets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/{Dataset}/
1_Data/Barcode_Statistics/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/{Dataset}/
1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/{Dataset}/data_L*.csv
6_Results/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/{Dataset}/
```

Consumers 2–5 and 7–8 read this arm's experiment-1 matrices. They must not start 500 Ripser jobs.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/<Dataset>/<script>.py
```

Or the folder launcher `run_all.py`.
