# Experiment 23 Results — Early Train/Test Split

**Protocol:** 80/20 stratified split on processed tabular data → PCA fit on train only → independent train/test landmarks and barcodes → train on train barcodes, evaluate on test barcodes.

**Code / results layout:**
- Legacy: `5_Experiments/Early_Split_TDA/1_PH_Default_Parameters/{Default_Of_Credit_Card_Client_Data,Statlog_German_Credit_Data}/`
- Registry (clean-protocol barcodes also namespaced here):  
  `6_Results/Early_Split_TDA/1_PH_Default_Parameters/{PKDD_Czech_Financial,Polish_Bankruptcy_3Year,Taiwan_Bankruptcy,South_German_Credit}/`  
  plus matching barcode trees under `1_Data/TDA_Datasets/Early_Split_TDA/1_PH_Default_Parameters/{Folder}/`.

---

## Legacy hold-out metrics (DCCCD + Statlog)

| Dataset | Mode | Sampling | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| Default of Credit Card Client | default | L5 | svm | 0.499 | 0.0 | 0.0 | 0.0 |
| Default of Credit Card Client | default | L5 | knn | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | default | L5 | xgb | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | default | L5 | logistic | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | default | L5 | random_forest | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | default | L15 | svm | 0.5 | 0.0 | 0.0 | 0.0 |
| Default of Credit Card Client | default | L15 | knn | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | default | L15 | xgb | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | default | L15 | logistic | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | default | L15 | random_forest | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | tuned | L5 | svm | 0.499 | 0.0 | 0.0 | 0.0 |
| Default of Credit Card Client | tuned | L5 | knn | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | tuned | L5 | xgb | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | tuned | L5 | logistic | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | tuned | L5 | random_forest | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | tuned | L15 | svm | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | tuned | L15 | knn | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | tuned | L15 | xgb | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | tuned | L15 | logistic | 0.5 | 0.5 | 1.0 | 0.6667 |
| Default of Credit Card Client | tuned | L15 | random_forest | 0.5 | 0.5 | 1.0 | 0.6667 |
| Statlog German Credit | default | L30 | svm | 0.436 | 0.2746 | 0.078 | 0.1215 |
| Statlog German Credit | default | L30 | knn | 0.508 | 0.504 | 0.998 | 0.6698 |
| Statlog German Credit | default | L30 | xgb | 0.504 | 0.502 | 0.984 | 0.6649 |
| Statlog German Credit | default | L30 | logistic | 0.503 | 0.5015 | 1.0 | 0.668 |
| Statlog German Credit | default | L30 | random_forest | 0.505 | 0.5026 | 0.984 | 0.6653 |
| Statlog German Credit | default | L60 | svm | 0.493 | 0.2941 | 0.01 | 0.0193 |
| Statlog German Credit | default | L60 | knn | 0.5 | 0.5 | 1.0 | 0.6667 |
| Statlog German Credit | default | L60 | xgb | 0.5 | 0.5 | 1.0 | 0.6667 |
| Statlog German Credit | default | L60 | logistic | 0.5 | 0.5 | 1.0 | 0.6667 |
| Statlog German Credit | default | L60 | random_forest | 0.5 | 0.5 | 1.0 | 0.6667 |
| Statlog German Credit | tuned | L30 | svm | 0.503 | 0.5015 | 1.0 | 0.668 |
| Statlog German Credit | tuned | L30 | knn | 0.506 | 0.503 | 0.996 | 0.6685 |
| Statlog German Credit | tuned | L30 | xgb | 0.498 | 0.499 | 0.976 | 0.6604 |
| Statlog German Credit | tuned | L30 | logistic | 0.502 | 0.501 | 1.0 | 0.6676 |
| Statlog German Credit | tuned | L30 | random_forest | 0.505 | 0.5026 | 0.982 | 0.6649 |
| Statlog German Credit | tuned | L60 | svm | 0.5 | 0.5 | 1.0 | 0.6667 |
| Statlog German Credit | tuned | L60 | knn | 0.5 | 0.5 | 1.0 | 0.6667 |
| Statlog German Credit | tuned | L60 | xgb | 0.5 | 0.5 | 1.0 | 0.6667 |
| Statlog German Credit | tuned | L60 | logistic | 0.5 | 0.5 | 1.0 | 0.6667 |
| Statlog German Credit | tuned | L60 | random_forest | 0.5 | 0.5 | 1.0 | 0.6667 |

---

## Notes

- Most Statlog models sit near 0.50 accuracy with recall ≈ 1.0 (predicting the positive class).
- This differs sharply from older full-data (leaky) barcode experiments and is an important early-split finding.
- Registry clean-protocol TDA metrics for the four additional datasets are stored as `tda_results.csv` under each Exp 23 dataset folder (sourced from the Exp 3 clean Protocol B runs).
- For the meeting-driven fixed-`t` redesign across all six datasets, see Experiment **28** and `docs/Revised_Snapshot_Protocol_Deep_Report.md`.

Artefacts: `6_Results/Early_Split_TDA/1_PH_Default_Parameters/`
