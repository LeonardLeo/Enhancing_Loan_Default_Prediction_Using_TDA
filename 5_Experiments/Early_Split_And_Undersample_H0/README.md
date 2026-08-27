# Early split and undersample, using just H0

Folder: `Early_Split_And_Undersample_H0`

Public name (figures and paper tables): **Early split and undersample, using just H0**. Names come from `utils.process_display_name()`.

This process **slices** homology-0 columns from the sibling `Early_Split_And_Undersample_H0_And_H1` barcode tables. It does not run Ripser.

## Live experiments

1. `1_PH_Default_Parameters` — keep H0 statistics (12 columns + label) and train default classifiers
2. `8_Null_Hypothesis_Algorithm2` — permutation test on those H0 barcode vectors

Nested extras live under `Archives/Four_Arm_Nested_Experiments/Early_Split_TDA/`.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_H0_only.py`).
