# Cross-Validation (K-Fold) Results

Generated from existing `6_Results/**/CV_results.pkl` files only (no new CV runs).

Table labels such as `3_PH_Default_Parameters` and `19_Linear_Regression_For_Prediction` are the **historical pickle experiment names**. Live folders: `Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters` (paper Exp 3) and `Historical_Late_Split_Balanced_TDA/5_Linear_Regression_For_Prediction` (paper Exp 10 / arm Exp 5). Archived paper experiments 12–14 keep their original numbers under `6_Results/Archives/`. Map: `docs/Repository_Layout.md`.

## Protocol (as implemented)

- **Folds:** 10-fold StratifiedKFold (`shuffle=True`, `random_state=42`).
- **Baseline ML (Exp 1–2):** CV on the tabular **training** features with models loaded from `model_results.pkl`.
- **TDA experiments:** `perform_cross_validation_tda` splits each barcode CSV 80/20, then runs 10-fold CV on the train portion using the stored estimators.
- **Primary CV metric:** sklearn `cross_val_score` default (**accuracy**).
- **Caveat:** For TDA experiments, barcode features were built under the historical full-data PCA/landmark pipeline (see `docs/Pipeline_Issues_And_Leakage.md`). CV therefore reflects that protocol.
- **Note:** Legacy CV scripts previously stored `mean_accracy` (typo). Source now writes `mean_accuracy`; visualisation helpers accept either key when reading older pickles. The tables below were generated from existing pickles and are unchanged.

## Summary table (mean ± std)

| experiment                                         | dataset                            | sampling_or_split   | model         | cv_mean_std     |   holdout_accuracy |   holdout_f1 |   holdout_minus_cv_mean |
|:---------------------------------------------------|:-----------------------------------|:--------------------|:--------------|:----------------|-------------------:|-------------:|------------------------:|
| 12_Equivalent_Sample_Size_For_Each_Dataset         | Default_Of_Credit_Card_Client_Data | data_L1.36.csv      | svm           | 0.8713 ± 0.0336 |             0.8550 |       0.8571 |                 -0.0163 |
| 12_Equivalent_Sample_Size_For_Each_Dataset         | Default_Of_Credit_Card_Client_Data | data_L1.36.csv      | knn           | 0.8787 ± 0.0387 |             0.8600 |       0.8586 |                 -0.0187 |
| 12_Equivalent_Sample_Size_For_Each_Dataset         | Default_Of_Credit_Card_Client_Data | data_L1.36.csv      | xgb           | 0.8838 ± 0.0375 |             0.8550 |       0.8571 |                 -0.0288 |
| 12_Equivalent_Sample_Size_For_Each_Dataset         | Default_Of_Credit_Card_Client_Data | data_L1.36.csv      | logistic      | 0.8625 ± 0.0367 |             0.8600 |       0.8600 |                 -0.0025 |
| 12_Equivalent_Sample_Size_For_Each_Dataset         | Default_Of_Credit_Card_Client_Data | data_L1.36.csv      | random_forest | 0.8912 ± 0.0395 |             0.8650 |       0.8657 |                 -0.0262 |
| 12_Equivalent_Sample_Size_For_Each_Dataset         | Default_Of_Credit_Card_Client_Data | data_L2.71.csv      | svm           | 0.9313 ± 0.0196 |             0.9500 |       0.9490 |                  0.0187 |
| 12_Equivalent_Sample_Size_For_Each_Dataset         | Default_Of_Credit_Card_Client_Data | data_L2.71.csv      | knn           | 0.9500 ± 0.0148 |             0.9550 |       0.9543 |                  0.0050 |
| 12_Equivalent_Sample_Size_For_Each_Dataset         | Default_Of_Credit_Card_Client_Data | data_L2.71.csv      | xgb           | 0.9537 ± 0.0177 |             0.9400 |       0.9388 |                 -0.0138 |
| 12_Equivalent_Sample_Size_For_Each_Dataset         | Default_Of_Credit_Card_Client_Data | data_L2.71.csv      | logistic      | 0.9200 ± 0.0263 |             0.9500 |       0.9490 |                  0.0300 |
| 12_Equivalent_Sample_Size_For_Each_Dataset         | Default_Of_Credit_Card_Client_Data | data_L2.71.csv      | random_forest | 0.9575 ± 0.0150 |             0.9450 |       0.9436 |                 -0.0125 |
| 13_Similar_Variance_Retained_After_PCA             | Default_Of_Credit_Card_Client_Data | data_L5.csv         | svm           | 0.9838 ± 0.0138 |             1.0000 |       1.0000 |                  0.0162 |
| 13_Similar_Variance_Retained_After_PCA             | Default_Of_Credit_Card_Client_Data | data_L5.csv         | knn           | 0.9925 ± 0.0083 |             1.0000 |       1.0000 |                  0.0075 |
| 13_Similar_Variance_Retained_After_PCA             | Default_Of_Credit_Card_Client_Data | data_L5.csv         | xgb           | 0.9963 ± 0.0057 |             0.9950 |       0.9950 |                 -0.0013 |
| 13_Similar_Variance_Retained_After_PCA             | Default_Of_Credit_Card_Client_Data | data_L5.csv         | logistic      | 0.9775 ± 0.0156 |             1.0000 |       1.0000 |                  0.0225 |
| 13_Similar_Variance_Retained_After_PCA             | Default_Of_Credit_Card_Client_Data | data_L5.csv         | random_forest | 0.9975 ± 0.0050 |             0.9950 |       0.9950 |                 -0.0025 |
| 13_Similar_Variance_Retained_After_PCA             | Default_Of_Credit_Card_Client_Data | data_L15.csv        | svm           | 0.9938 ± 0.0062 |             1.0000 |       1.0000 |                  0.0062 |
| 13_Similar_Variance_Retained_After_PCA             | Default_Of_Credit_Card_Client_Data | data_L15.csv        | knn           | 0.9975 ± 0.0050 |             1.0000 |       1.0000 |                  0.0025 |
| 13_Similar_Variance_Retained_After_PCA             | Default_Of_Credit_Card_Client_Data | data_L15.csv        | xgb           | 0.9988 ± 0.0037 |             1.0000 |       1.0000 |                  0.0012 |
| 13_Similar_Variance_Retained_After_PCA             | Default_Of_Credit_Card_Client_Data | data_L15.csv        | logistic      | 0.9838 ± 0.0126 |             1.0000 |       1.0000 |                  0.0162 |
| 13_Similar_Variance_Retained_After_PCA             | Default_Of_Credit_Card_Client_Data | data_L15.csv        | random_forest | 1.0000 ± 0.0000 |             1.0000 |       1.0000 |                  0.0000 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Default_Of_Credit_Card_Client_Data | data_L5.csv         | svm           | 0.8000 ± 0.0000 |             1.0000 |       1.0000 |                  0.2000 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Default_Of_Credit_Card_Client_Data | data_L5.csv         | knn           | 0.9788 ± 0.0126 |             0.9900 |       0.9744 |                  0.0112 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Default_Of_Credit_Card_Client_Data | data_L5.csv         | xgb           | 0.9938 ± 0.0062 |             1.0000 |       1.0000 |                  0.0062 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Default_Of_Credit_Card_Client_Data | data_L5.csv         | logistic      | 0.8000 ± 0.0000 |             0.9950 |       0.9873 |                  0.1950 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Default_Of_Credit_Card_Client_Data | data_L5.csv         | random_forest | 0.9913 ± 0.0080 |             1.0000 |       1.0000 |                  0.0087 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Default_Of_Credit_Card_Client_Data | data_L15.csv        | svm           | 0.8000 ± 0.0000 |             1.0000 |       1.0000 |                  0.2000 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Default_Of_Credit_Card_Client_Data | data_L15.csv        | knn           | 0.9925 ± 0.0083 |             1.0000 |       1.0000 |                  0.0075 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Default_Of_Credit_Card_Client_Data | data_L15.csv        | xgb           | 1.0000 ± 0.0000 |             1.0000 |       1.0000 |                  0.0000 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Default_Of_Credit_Card_Client_Data | data_L15.csv        | logistic      | 0.8000 ± 0.0000 |             1.0000 |       1.0000 |                  0.2000 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Default_Of_Credit_Card_Client_Data | data_L15.csv        | random_forest | 1.0000 ± 0.0000 |             1.0000 |       1.0000 |                  0.0000 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Statlog_German_Credit_Data         | data_L30.csv        | svm           | 0.8000 ± 0.0000 |             0.8100 |       0.2400 |                  0.0100 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Statlog_German_Credit_Data         | data_L30.csv        | knn           | 0.8075 ± 0.0448 |             0.7750 |       0.3284 |                 -0.0325 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Statlog_German_Credit_Data         | data_L30.csv        | xgb           | 0.8063 ± 0.0322 |             0.7900 |       0.3438 |                 -0.0162 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Statlog_German_Credit_Data         | data_L30.csv        | logistic      | 0.8063 ± 0.0101 |             0.8200 |       0.3333 |                  0.0137 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Statlog_German_Credit_Data         | data_L30.csv        | random_forest | 0.8175 ± 0.0307 |             0.8150 |       0.3019 |                 -0.0025 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Statlog_German_Credit_Data         | data_L60.csv        | svm           | 0.8000 ± 0.0000 |             0.8500 |       0.5588 |                  0.0500 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Statlog_German_Credit_Data         | data_L60.csv        | knn           | 0.8600 ± 0.0339 |             0.8400 |       0.5556 |                 -0.0200 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Statlog_German_Credit_Data         | data_L60.csv        | xgb           | 0.8738 ± 0.0337 |             0.8600 |       0.5882 |                 -0.0138 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Statlog_German_Credit_Data         | data_L60.csv        | logistic      | 0.8075 ± 0.0195 |             0.8500 |       0.5455 |                  0.0425 |
| 14_Mixed_Classes_Training_With_Imbalanced_Datasets | Statlog_German_Credit_Data         | data_L60.csv        | random_forest | 0.8675 ± 0.0384 |             0.8450 |       0.5231 |                 -0.0225 |
| 19_Linear_Regression_For_Prediction                | Default_Of_Credit_Card_Client_Data | data_L5.csv         | linear        | 0.8450 ± 0.0324 |             0.9900 |       0.9899 |                  0.1450 |
| 19_Linear_Regression_For_Prediction                | Default_Of_Credit_Card_Client_Data | data_L15.csv        | linear        | 0.9386 ± 0.0098 |             1.0000 |       1.0000 |                  0.0614 |
| 19_Linear_Regression_For_Prediction                | Statlog_German_Credit_Data         | data_L30.csv        | linear        | 0.1574 ± 0.0583 |             0.6950 |       0.7163 |                  0.5376 |
| 19_Linear_Regression_For_Prediction                | Statlog_German_Credit_Data         | data_L60.csv        | linear        | 0.3079 ± 0.0669 |             0.7450 |       0.7671 |                  0.4371 |
| 1_ML_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | (tabular train set) | svm           | 0.8173 ± 0.0054 |             0.8109 |       0.5152 |                 -0.0064 |
| 1_ML_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | (tabular train set) | knn           | 0.7923 ± 0.0099 |             0.7485 |       0.4554 |                 -0.0438 |
| 1_ML_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | (tabular train set) | xgb           | 0.8127 ± 0.0055 |             0.8134 |       0.4676 |                  0.0007 |
| 1_ML_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | (tabular train set) | logistic      | 0.8096 ± 0.0038 |             0.6781 |       0.4673 |                 -0.1315 |
| 1_ML_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | (tabular train set) | random_forest | 0.8139 ± 0.0076 |             0.8123 |       0.4842 |                 -0.0017 |
| 1_ML_Default_Parameters                            | Statlog_German_Credit_Data         | (tabular train set) | svm           | 0.7625 ± 0.0262 |             0.7400 |       0.5938 |                 -0.0225 |
| 1_ML_Default_Parameters                            | Statlog_German_Credit_Data         | (tabular train set) | knn           | 0.7238 ± 0.0276 |             0.6800 |       0.5429 |                 -0.0438 |
| 1_ML_Default_Parameters                            | Statlog_German_Credit_Data         | (tabular train set) | xgb           | 0.7412 ± 0.0415 |             0.7550 |       0.5333 |                  0.0138 |
| 1_ML_Default_Parameters                            | Statlog_German_Credit_Data         | (tabular train set) | logistic      | 0.7675 ± 0.0275 |             0.7450 |       0.6047 |                 -0.0225 |
| 1_ML_Default_Parameters                            | Statlog_German_Credit_Data         | (tabular train set) | random_forest | 0.7525 ± 0.0156 |             0.7500 |       0.5283 |                 -0.0025 |
| 2_ML_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | (tabular train set) | svm           | 0.8197 ± 0.0069 |             0.8091 |       0.5132 |                 -0.0106 |
| 2_ML_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | (tabular train set) | knn           | 0.7651 ± 0.0103 |             0.7499 |       0.4160 |                 -0.0152 |
| 2_ML_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | (tabular train set) | xgb           | 0.8164 ± 0.0068 |             0.8160 |       0.4720 |                 -0.0004 |
| 2_ML_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | (tabular train set) | logistic      | 0.8096 ± 0.0038 |             0.6781 |       0.4673 |                 -0.1315 |
| 2_ML_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | (tabular train set) | random_forest | 0.8162 ± 0.0075 |             0.8129 |       0.4827 |                 -0.0033 |
| 2_ML_Tuned_Parameters                              | Statlog_German_Credit_Data         | (tabular train set) | svm           | 0.7300 ± 0.0458 |             0.7300 |       0.5345 |                  0.0000 |
| 2_ML_Tuned_Parameters                              | Statlog_German_Credit_Data         | (tabular train set) | knn           | 0.7350 ± 0.0295 |             0.7100 |       0.5085 |                 -0.0250 |
| 2_ML_Tuned_Parameters                              | Statlog_German_Credit_Data         | (tabular train set) | xgb           | 0.7787 ± 0.0244 |             0.7350 |       0.5310 |                 -0.0437 |
| 2_ML_Tuned_Parameters                              | Statlog_German_Credit_Data         | (tabular train set) | logistic      | 0.7762 ± 0.0342 |             0.7100 |       0.5672 |                 -0.0663 |
| 2_ML_Tuned_Parameters                              | Statlog_German_Credit_Data         | (tabular train set) | random_forest | 0.7688 ± 0.0151 |             0.7500 |       0.5614 |                 -0.0188 |
| 3_PH_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | data_L5.csv         | svm           | 0.9537 ± 0.0231 |             0.9850 |       0.9848 |                  0.0312 |
| 3_PH_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | data_L5.csv         | knn           | 0.9850 ± 0.0146 |             0.9900 |       0.9899 |                  0.0050 |
| 3_PH_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | data_L5.csv         | xgb           | 0.9938 ± 0.0084 |             0.9900 |       0.9899 |                 -0.0038 |
| 3_PH_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | data_L5.csv         | logistic      | 0.9337 ± 0.0256 |             0.9850 |       0.9848 |                  0.0513 |
| 3_PH_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | data_L5.csv         | random_forest | 0.9925 ± 0.0115 |             0.9850 |       0.9848 |                 -0.0075 |
| 3_PH_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | data_L15.csv        | svm           | 0.9800 ± 0.0218 |             1.0000 |       1.0000 |                  0.0200 |
| 3_PH_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | data_L15.csv        | knn           | 0.9925 ± 0.0083 |             1.0000 |       1.0000 |                  0.0075 |
| 3_PH_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | data_L15.csv        | xgb           | 0.9988 ± 0.0037 |             1.0000 |       1.0000 |                  0.0012 |
| 3_PH_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | data_L15.csv        | logistic      | 0.9738 ± 0.0247 |             1.0000 |       1.0000 |                  0.0262 |
| 3_PH_Default_Parameters                            | Default_Of_Credit_Card_Client_Data | data_L15.csv        | random_forest | 1.0000 ± 0.0000 |             1.0000 |       1.0000 |                  0.0000 |
| 3_PH_Default_Parameters                            | Statlog_German_Credit_Data         | data_L30.csv        | svm           | 0.7063 ± 0.0448 |             0.7400 |       0.7658 |                  0.0337 |
| 3_PH_Default_Parameters                            | Statlog_German_Credit_Data         | data_L30.csv        | knn           | 0.6725 ± 0.0453 |             0.6850 |       0.7225 |                  0.0125 |
| 3_PH_Default_Parameters                            | Statlog_German_Credit_Data         | data_L30.csv        | xgb           | 0.7213 ± 0.0291 |             0.7450 |       0.7628 |                  0.0237 |
| 3_PH_Default_Parameters                            | Statlog_German_Credit_Data         | data_L30.csv        | logistic      | 0.7113 ± 0.0479 |             0.7250 |       0.7368 |                  0.0137 |
| 3_PH_Default_Parameters                            | Statlog_German_Credit_Data         | data_L30.csv        | random_forest | 0.7288 ± 0.0274 |             0.7100 |       0.7238 |                 -0.0188 |
| 3_PH_Default_Parameters                            | Statlog_German_Credit_Data         | data_L60.csv        | svm           | 0.8100 ± 0.0348 |             0.8250 |       0.8341 |                  0.0150 |
| 3_PH_Default_Parameters                            | Statlog_German_Credit_Data         | data_L60.csv        | knn           | 0.8287 ± 0.0224 |             0.7750 |       0.7887 |                 -0.0537 |
| 3_PH_Default_Parameters                            | Statlog_German_Credit_Data         | data_L60.csv        | xgb           | 0.8612 ± 0.0276 |             0.8500 |       0.8529 |                 -0.0112 |
| 3_PH_Default_Parameters                            | Statlog_German_Credit_Data         | data_L60.csv        | logistic      | 0.8088 ± 0.0296 |             0.8050 |       0.8169 |                 -0.0038 |
| 3_PH_Default_Parameters                            | Statlog_German_Credit_Data         | data_L60.csv        | random_forest | 0.8487 ± 0.0142 |             0.8100 |       0.8137 |                 -0.0387 |
| 4_PH_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | data_L5.csv         | svm           | 0.8862 ± 0.0333 |             0.9850 |       0.9848 |                  0.0988 |
| 4_PH_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | data_L5.csv         | knn           | 0.9887 ± 0.0131 |             0.9850 |       0.9848 |                 -0.0037 |
| 4_PH_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | data_L5.csv         | xgb           | 0.9913 ± 0.0112 |             0.9900 |       0.9899 |                 -0.0013 |
| 4_PH_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | data_L5.csv         | logistic      | 0.9750 ± 0.0168 |             0.9900 |       0.9899 |                  0.0150 |
| 4_PH_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | data_L5.csv         | random_forest | 0.9913 ± 0.0112 |             0.9900 |       0.9899 |                 -0.0013 |
| 4_PH_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | data_L15.csv        | svm           | 0.9700 ± 0.0251 |             1.0000 |       1.0000 |                  0.0300 |
| 4_PH_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | data_L15.csv        | knn           | 0.9963 ± 0.0057 |             1.0000 |       1.0000 |                  0.0037 |
| 4_PH_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | data_L15.csv        | xgb           | 0.9988 ± 0.0037 |             1.0000 |       1.0000 |                  0.0012 |
| 4_PH_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | data_L15.csv        | logistic      | 0.9738 ± 0.0247 |             1.0000 |       1.0000 |                  0.0262 |
| 4_PH_Tuned_Parameters                              | Default_Of_Credit_Card_Client_Data | data_L15.csv        | random_forest | 1.0000 ± 0.0000 |             1.0000 |       1.0000 |                  0.0000 |
| 4_PH_Tuned_Parameters                              | Statlog_German_Credit_Data         | data_L30.csv        | svm           | 0.7213 ± 0.0399 |             0.7100 |       0.7456 |                 -0.0113 |
| 4_PH_Tuned_Parameters                              | Statlog_German_Credit_Data         | data_L30.csv        | knn           | 0.7000 ± 0.0433 |             0.7350 |       0.7580 |                  0.0350 |
| 4_PH_Tuned_Parameters                              | Statlog_German_Credit_Data         | data_L30.csv        | xgb           | 0.7288 ± 0.0263 |             0.7300 |       0.7500 |                  0.0012 |
| 4_PH_Tuned_Parameters                              | Statlog_German_Credit_Data         | data_L30.csv        | logistic      | 0.7237 ± 0.0298 |             0.7200 |       0.7358 |                 -0.0037 |
| 4_PH_Tuned_Parameters                              | Statlog_German_Credit_Data         | data_L30.csv        | random_forest | 0.7312 ± 0.0275 |             0.7350 |       0.7512 |                  0.0038 |
| 4_PH_Tuned_Parameters                              | Statlog_German_Credit_Data         | data_L60.csv        | svm           | 0.8100 ± 0.0348 |             0.8250 |       0.8341 |                  0.0150 |
| 4_PH_Tuned_Parameters                              | Statlog_German_Credit_Data         | data_L60.csv        | knn           | 0.8450 ± 0.0238 |             0.8150 |       0.8213 |                 -0.0300 |
| 4_PH_Tuned_Parameters                              | Statlog_German_Credit_Data         | data_L60.csv        | xgb           | 0.8700 ± 0.0179 |             0.8400 |       0.8491 |                 -0.0300 |
| 4_PH_Tuned_Parameters                              | Statlog_German_Credit_Data         | data_L60.csv        | logistic      | 0.8375 ± 0.0274 |             0.7850 |       0.7943 |                 -0.0525 |
| 4_PH_Tuned_Parameters                              | Statlog_German_Credit_Data         | data_L60.csv        | random_forest | 0.8625 ± 0.0177 |             0.8050 |       0.8079 |                 -0.0575 |
| 6_Experiment_Impact_of_H0_Only                     | Default_Of_Credit_Card_Client_Data | data_L5.csv         | svm           | 0.9750 ± 0.0125 |             0.9750 |       0.9746 |                  0.0000 |
| 6_Experiment_Impact_of_H0_Only                     | Default_Of_Credit_Card_Client_Data | data_L5.csv         | knn           | 0.9837 ± 0.0148 |             0.9800 |       0.9798 |                 -0.0037 |
| 6_Experiment_Impact_of_H0_Only                     | Default_Of_Credit_Card_Client_Data | data_L5.csv         | xgb           | 0.9875 ± 0.0168 |             0.9800 |       0.9796 |                 -0.0075 |
| 6_Experiment_Impact_of_H0_Only                     | Default_Of_Credit_Card_Client_Data | data_L5.csv         | logistic      | 0.9738 ± 0.0142 |             0.9800 |       0.9798 |                  0.0062 |
| 6_Experiment_Impact_of_H0_Only                     | Default_Of_Credit_Card_Client_Data | data_L5.csv         | random_forest | 0.9863 ± 0.0172 |             0.9800 |       0.9798 |                 -0.0063 |
| 6_Experiment_Impact_of_H0_Only                     | Default_Of_Credit_Card_Client_Data | data_L15.csv        | svm           | 0.9900 ± 0.0122 |             1.0000 |       1.0000 |                  0.0100 |
| 6_Experiment_Impact_of_H0_Only                     | Default_Of_Credit_Card_Client_Data | data_L15.csv        | knn           | 0.9988 ± 0.0037 |             1.0000 |       1.0000 |                  0.0012 |
| 6_Experiment_Impact_of_H0_Only                     | Default_Of_Credit_Card_Client_Data | data_L15.csv        | xgb           | 0.9988 ± 0.0037 |             1.0000 |       1.0000 |                  0.0012 |
| 6_Experiment_Impact_of_H0_Only                     | Default_Of_Credit_Card_Client_Data | data_L15.csv        | logistic      | 0.9900 ± 0.0122 |             1.0000 |       1.0000 |                  0.0100 |
| 6_Experiment_Impact_of_H0_Only                     | Default_Of_Credit_Card_Client_Data | data_L15.csv        | random_forest | 1.0000 ± 0.0000 |             1.0000 |       1.0000 |                  0.0000 |
| 6_Experiment_Impact_of_H0_Only                     | Statlog_German_Credit_Data         | data_L30.csv        | svm           | 0.6713 ± 0.0363 |             0.6900 |       0.7328 |                  0.0187 |
| 6_Experiment_Impact_of_H0_Only                     | Statlog_German_Credit_Data         | data_L30.csv        | knn           | 0.6850 ± 0.0310 |             0.6400 |       0.6505 |                 -0.0450 |
| 6_Experiment_Impact_of_H0_Only                     | Statlog_German_Credit_Data         | data_L30.csv        | xgb           | 0.6800 ± 0.0400 |             0.7000 |       0.7143 |                  0.0200 |
| 6_Experiment_Impact_of_H0_Only                     | Statlog_German_Credit_Data         | data_L30.csv        | logistic      | 0.6713 ± 0.0285 |             0.6700 |       0.6916 |                 -0.0012 |
| 6_Experiment_Impact_of_H0_Only                     | Statlog_German_Credit_Data         | data_L30.csv        | random_forest | 0.6938 ± 0.0225 |             0.7150 |       0.7421 |                  0.0212 |
| 6_Experiment_Impact_of_H0_Only                     | Statlog_German_Credit_Data         | data_L60.csv        | svm           | 0.6900 ± 0.0612 |             0.7350 |       0.7558 |                  0.0450 |
| 6_Experiment_Impact_of_H0_Only                     | Statlog_German_Credit_Data         | data_L60.csv        | knn           | 0.7937 ± 0.0322 |             0.7750 |       0.7907 |                 -0.0187 |
| 6_Experiment_Impact_of_H0_Only                     | Statlog_German_Credit_Data         | data_L60.csv        | xgb           | 0.8575 ± 0.0404 |             0.8600 |       0.8667 |                  0.0025 |
| 6_Experiment_Impact_of_H0_Only                     | Statlog_German_Credit_Data         | data_L60.csv        | logistic      | 0.6838 ± 0.0554 |             0.7200 |       0.7333 |                  0.0362 |
| 6_Experiment_Impact_of_H0_Only                     | Statlog_German_Credit_Data         | data_L60.csv        | random_forest | 0.8100 ± 0.0457 |             0.7950 |       0.8057 |                 -0.0150 |

## Interpretation vs hold-out

### 12_Equivalent_Sample_Size_For_Each_Dataset — Default_Of_Credit_Card_Client_Data — `data_L1.36.csv`

- Best CV model: **random_forest** (0.8912 ± 0.0395).
- Mean (hold-out accuracy − CV mean) across models: **-0.0185** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 12_Equivalent_Sample_Size_For_Each_Dataset — Default_Of_Credit_Card_Client_Data — `data_L2.71.csv`

- Best CV model: **random_forest** (0.9575 ± 0.0150).
- Mean (hold-out accuracy − CV mean) across models: **+0.0055** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 13_Similar_Variance_Retained_After_PCA — Default_Of_Credit_Card_Client_Data — `data_L5.csv`

- Best CV model: **random_forest** (0.9975 ± 0.0050).
- Mean (hold-out accuracy − CV mean) across models: **+0.0085** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 13_Similar_Variance_Retained_After_PCA — Default_Of_Credit_Card_Client_Data — `data_L15.csv`

- Best CV model: **random_forest** (1.0000 ± 0.0000).
- Mean (hold-out accuracy − CV mean) across models: **+0.0052** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 14_Mixed_Classes_Training_With_Imbalanced_Datasets — Default_Of_Credit_Card_Client_Data — `data_L5.csv`

- Best CV model: **xgb** (0.9938 ± 0.0062).
- Mean (hold-out accuracy − CV mean) across models: **+0.0842** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 14_Mixed_Classes_Training_With_Imbalanced_Datasets — Default_Of_Credit_Card_Client_Data — `data_L15.csv`

- Best CV model: **xgb** (1.0000 ± 0.0000).
- Mean (hold-out accuracy − CV mean) across models: **+0.0815** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 14_Mixed_Classes_Training_With_Imbalanced_Datasets — Statlog_German_Credit_Data — `data_L30.csv`

- Best CV model: **random_forest** (0.8175 ± 0.0307).
- Mean (hold-out accuracy − CV mean) across models: **-0.0055** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 14_Mixed_Classes_Training_With_Imbalanced_Datasets — Statlog_German_Credit_Data — `data_L60.csv`

- Best CV model: **xgb** (0.8738 ± 0.0337).
- Mean (hold-out accuracy − CV mean) across models: **+0.0072** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 19_Linear_Regression_For_Prediction — Default_Of_Credit_Card_Client_Data — `data_L5.csv`

- Best CV model: **linear** (0.8450 ± 0.0324).
- Mean (hold-out accuracy − CV mean) across models: **+0.1450** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 19_Linear_Regression_For_Prediction — Default_Of_Credit_Card_Client_Data — `data_L15.csv`

- Best CV model: **linear** (0.9386 ± 0.0098).
- Mean (hold-out accuracy − CV mean) across models: **+0.0614** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 19_Linear_Regression_For_Prediction — Statlog_German_Credit_Data — `data_L30.csv`

- Best CV model: **linear** (0.1574 ± 0.0583).
- Mean (hold-out accuracy − CV mean) across models: **+0.5376** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 19_Linear_Regression_For_Prediction — Statlog_German_Credit_Data — `data_L60.csv`

- Best CV model: **linear** (0.3079 ± 0.0669).
- Mean (hold-out accuracy − CV mean) across models: **+0.4371** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 1_ML_Default_Parameters — Default_Of_Credit_Card_Client_Data — `(tabular train set)`

- Best CV model: **svm** (0.8173 ± 0.0054).
- Mean (hold-out accuracy − CV mean) across models: **-0.0365** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 1_ML_Default_Parameters — Statlog_German_Credit_Data — `(tabular train set)`

- Best CV model: **logistic** (0.7675 ± 0.0275).
- Mean (hold-out accuracy − CV mean) across models: **-0.0155** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 2_ML_Tuned_Parameters — Default_Of_Credit_Card_Client_Data — `(tabular train set)`

- Best CV model: **svm** (0.8197 ± 0.0069).
- Mean (hold-out accuracy − CV mean) across models: **-0.0322** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 2_ML_Tuned_Parameters — Statlog_German_Credit_Data — `(tabular train set)`

- Best CV model: **xgb** (0.7787 ± 0.0244).
- Mean (hold-out accuracy − CV mean) across models: **-0.0308** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 3_PH_Default_Parameters — Default_Of_Credit_Card_Client_Data — `data_L5.csv`

- Best CV model: **xgb** (0.9938 ± 0.0084).
- Mean (hold-out accuracy − CV mean) across models: **+0.0153** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 3_PH_Default_Parameters — Default_Of_Credit_Card_Client_Data — `data_L15.csv`

- Best CV model: **random_forest** (1.0000 ± 0.0000).
- Mean (hold-out accuracy − CV mean) across models: **+0.0110** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 3_PH_Default_Parameters — Statlog_German_Credit_Data — `data_L30.csv`

- Best CV model: **random_forest** (0.7288 ± 0.0274).
- Mean (hold-out accuracy − CV mean) across models: **+0.0130** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 3_PH_Default_Parameters — Statlog_German_Credit_Data — `data_L60.csv`

- Best CV model: **xgb** (0.8612 ± 0.0276).
- Mean (hold-out accuracy − CV mean) across models: **-0.0185** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 4_PH_Tuned_Parameters — Default_Of_Credit_Card_Client_Data — `data_L5.csv`

- Best CV model: **xgb** (0.9913 ± 0.0112).
- Mean (hold-out accuracy − CV mean) across models: **+0.0215** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 4_PH_Tuned_Parameters — Default_Of_Credit_Card_Client_Data — `data_L15.csv`

- Best CV model: **random_forest** (1.0000 ± 0.0000).
- Mean (hold-out accuracy − CV mean) across models: **+0.0122** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 4_PH_Tuned_Parameters — Statlog_German_Credit_Data — `data_L30.csv`

- Best CV model: **random_forest** (0.7312 ± 0.0275).
- Mean (hold-out accuracy − CV mean) across models: **+0.0050** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 4_PH_Tuned_Parameters — Statlog_German_Credit_Data — `data_L60.csv`

- Best CV model: **xgb** (0.8700 ± 0.0179).
- Mean (hold-out accuracy − CV mean) across models: **-0.0310** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 6_Experiment_Impact_of_H0_Only — Default_Of_Credit_Card_Client_Data — `data_L5.csv`

- Best CV model: **xgb** (0.9875 ± 0.0168).
- Mean (hold-out accuracy − CV mean) across models: **-0.0023** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 6_Experiment_Impact_of_H0_Only — Default_Of_Credit_Card_Client_Data — `data_L15.csv`

- Best CV model: **random_forest** (1.0000 ± 0.0000).
- Mean (hold-out accuracy − CV mean) across models: **+0.0045** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 6_Experiment_Impact_of_H0_Only — Statlog_German_Credit_Data — `data_L30.csv`

- Best CV model: **random_forest** (0.6938 ± 0.0225).
- Mean (hold-out accuracy − CV mean) across models: **+0.0027** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

### 6_Experiment_Impact_of_H0_Only — Statlog_German_Credit_Data — `data_L60.csv`

- Best CV model: **xgb** (0.8575 ± 0.0404).
- Mean (hold-out accuracy − CV mean) across models: **+0.0100** (positive ⇒ hold-out higher than CV mean).
- CV mean is accuracy on stratified 10-fold of the training portion; compare to hold-out accuracy from model_results.pkl.

## Fold-level scores

Full fold table saved to `docs/cv_fold_scores.csv`.

Total fold rows: **1240**.

## Files covered

- `12_Equivalent_Sample_Size_For_Each_Dataset`
- `13_Similar_Variance_Retained_After_PCA`
- `14_Mixed_Classes_Training_With_Imbalanced_Datasets`
- `19_Linear_Regression_For_Prediction`
- `1_ML_Default_Parameters`
- `2_ML_Tuned_Parameters`
- `3_PH_Default_Parameters`
- `4_PH_Tuned_Parameters`
- `6_Experiment_Impact_of_H0_Only`

## Missing CV artefacts

- Experiment **11** (paper) has no `CV_results.pkl` in `6_Results/` at documentation time.
- Exploratory experiments generally were not CV-scored.
