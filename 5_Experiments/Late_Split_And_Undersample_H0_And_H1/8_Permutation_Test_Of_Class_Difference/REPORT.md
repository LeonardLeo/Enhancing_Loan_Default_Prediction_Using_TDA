# Late_Split_And_Undersample_H0_And_H1 — 8_Permutation_Test_Of_Class_Difference

## Protocol knobs

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | yes — majority downsampled to minority count before landmarks |
| PCA | MinMax + PCA on the FULL processed table (historical / slightly leaky) |
| Snapshot size | points per snapshot = floor(minority class count × snapshot size percent / 100); after undersampling both classes have the minority count |
| Number of snapshots | 500 (dataset historical n_files) |

CONSUMER. Robinson–Turner permutation test of class difference is used here as a vector-summary proxy on 24-dimensional H0+H1 barcode-statistic vectors, not on raw persistence diagrams. For each contrast, the implementation compares the observed allocation with 199 random row-label relabellings (200 total statistics). Because snapshots redraw customers, barcode rows are overlap-dependent and row exchangeability is questionable: the resulting p-values are conditional, exploratory shuffle diagnostics for this generated library, not calibrated customer-level tests or proof that the underlying class processes differ. Vector coordinates remain on their stored, unstandardised scales, and the family of tests is not multiplicity-adjusted.

Same Exp 3 PCA ranks and snapshot-size percents (`docs/Design_Decisions.md`, `utils.DatasetConfig`):

| Dataset | Snapshot size as percent of the class | PCA rank | Number of snapshots |
|---------|-------------------|----------|-----|
| Default_Of_Credit_Card_Client_Data | L5 / L15 | 7 | 500 |

## Artefacts

```
1_Data/Landmark_Sets/Late_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/
1_Data/Barcode_Statistics/Late_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/
1_Data/TDA_Datasets/Late_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/data_L*.csv
6_Results/Late_Split_And_Undersample_H0_And_H1/8_Permutation_Test_Of_Class_Difference/{Dataset}/
```

Tuned models, the sampling-ratio audit, and permutation test of class difference read this process's experiment-1 matrices. They must not start 500 Ripser jobs.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/Late_Split_And_Undersample_H0_And_H1/8_Permutation_Test_Of_Class_Difference/<Dataset>/<script>.py
```


