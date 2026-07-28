# Experiment 23 — Early Train/Test Split (Protocol B)

## Goal

Remove train/test leakage from the TDA pipeline by splitting **before** PCA and landmark (“snapshot”) generation.

## Protocol

1. Stratified 80/20 on processed tabular data  
2. Fit MinMaxScaler + PCA on **train only**; transform train & test  
3. Undersample-balance **within** each split  
4. Generate landmarks independently for train and test (`add_optional_path`)  
5. Compute barcodes independently  
6. Train on train barcodes; evaluate on test barcodes  
   - Default hyperparameters → `model_results_default.pkl`  
   - GridSearchCV (tuned) → `model_results_tuned.pkl`

## Scripts

| Dataset | Script |
|---------|--------|
| DCCCD | `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py` |
| SGCD | `Statlog_German_Credit_Data/statlog_german_credit_data_PH.py` |

## How to run

```powershell
cd 5_Experiments\23_Early_Train_Test_Split\Default_Of_Credit_Card_Client_Data
python default_of_credit_cards_client_PH.py

cd ..\Statlog_German_Credit_Data
python statlog_german_credit_data_PH.py
```

Ensure the repository root is on `PYTHONPATH` (or run from an IDE that already imports `utils.py`).

## Cost warning

With `n_files_per_percentage = 500`, this regenerates landmarks for **both** train and test. Expect long runtimes (especially DCCCD).

## Related

- Leakage discussion: `docs/Pipeline_Issues_And_Leakage.md`  
- New utils: `stratified_early_split`, `fit_scaler_pca_on_train`, `train_dataset_tda_presplit`, `train_models_on_presplit_dataset`
