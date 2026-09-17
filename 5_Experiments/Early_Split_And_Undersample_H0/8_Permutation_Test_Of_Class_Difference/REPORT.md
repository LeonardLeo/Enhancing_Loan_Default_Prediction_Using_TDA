# Early_Split_And_Undersample_H0 — 8_Permutation_Test_Of_Class_Difference

## Protocol knobs

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | yes — independently inside the train pool and the test pool |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| Snapshot size | points per snapshot = floor(class count × snapshot size percent / 100) on the undersampled pool of that split |
| Number of snapshots | 500 (dataset historical n_files) |

CONSUMER. Robinson–Turner permutation test of class difference is used here as a vector-summary proxy on 12-dimensional H0 barcode-statistic vectors, not on raw persistence diagrams. For each contrast, the implementation compares the observed allocation with 199 random row-label relabellings (200 total statistics). Because snapshots redraw customers, barcode rows are overlap-dependent and row exchangeability is questionable: the resulting p-values are conditional, exploratory shuffle diagnostics for this generated library, not calibrated customer-level tests or proof that the underlying class processes differ. Vector coordinates remain on their stored, unstandardised scales, and the family of tests is not multiplicity-adjusted.

Same Exp 3 PCA ranks and snapshot-size percents (`docs/Design_Decisions.md`, `utils.DatasetConfig`):

| Dataset | Snapshot size as percent of the class | PCA rank | Number of snapshots |
|---------|-------------------|----------|-----|
| Default_Of_Credit_Card_Client_Data | L5 / L15 | 7 | 500 |

## Artefacts

```
1_Data/Landmark_Sets/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/  (sibling H0-and-H1 Ripser run)
1_Data/Barcode_Statistics/Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/{Dataset}/  (sibling H0-and-H1 Ripser run)
1_Data/TDA_Datasets/Early_Split_And_Undersample_H0/1_PH_Default_Parameters/{Dataset}/data_L*.csv
6_Results/Early_Split_And_Undersample_H0/8_Permutation_Test_Of_Class_Difference/{Dataset}/
```

This process slices H0 columns from the sibling H0-and-H1 barcode tables, then trains. It must not start Ripser.

## Hold-out numbers that contradict a universal p = 0.005 claim

`permutation_test_results.csv` on this process. Smallest possible p-value with B = 200 is 0.005.

| Dataset | Library | Setting | F₂,₂ p | F₁,₁ p | F₂,₁ p |
|---------|---------|---------|-------:|-------:|-------:|
| DCCCD | train / test | L5 / L15 | 0.005 | 0.005 | 0.005 |
| Statlog | TRAIN | L30 / L60 | 0.005 | 0.005 | 0.005 |
| Statlog | TEST | L30 | **0.040** | **0.050** | **0.045** |
| Statlog | TEST | L60 | 0.005 | 0.005 | 0.005 |

Statlog TEST L30 is mixed: F₁,₁ sits on the 0.05 threshold; F₂,₂ and F₂,₁ reject only narrowly. Do not quote “permutation test of class difference p = 0.005 everywhere.”

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_And_Undersample_H0/8_Permutation_Test_Of_Class_Difference/<Dataset>/<script>.py
```


