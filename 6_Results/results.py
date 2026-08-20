# -*- coding: utf-8 -*-
"""
Paper-table aggregator.

Run from ``6_Results/``:

    python results.py

LaTeX/CSV outputs are written to ``6_Results/Paper_Tables/``.
Pickle inputs are still read from the protocol buckets beside this file.
"""

import sys
from pathlib import Path

import joblib

RESULTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = RESULTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import build_results_dataframe_v3

PAPER_TABLES = RESULTS_DIR / "Paper_Tables"
PAPER_TABLES.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Experiment 1 - 1_ML_Default_Parameters
# =============================================================================
# --- Statlog
Experiment_1_Model_Results_SGCD = joblib.load("Default_Parameters/1_ML_Default_Parameters/Statlog_German_Credit_Data/model_results.pkl")
Experiment_1_CV_Results_SGCD = joblib.load("Default_Parameters/1_ML_Default_Parameters/Statlog_German_Credit_Data/CV_results.pkl")

Experiment_1_Model_Results_SGCD = {"Statlog German Credit Dataset": Experiment_1_Model_Results_SGCD}
Experiment_1_CV_Results_SGCD = {"Statlog German Credit Dataset": Experiment_1_CV_Results_SGCD}

# --- Default of Credit Card Client
Experiment_1_Model_Results_DCCCD = joblib.load("Default_Parameters/1_ML_Default_Parameters/Default_Of_Credit_Card_Client_Data/model_results.pkl")
Experiment_1_CV_Results_DCCCD = joblib.load("Default_Parameters/1_ML_Default_Parameters/Default_Of_Credit_Card_Client_Data/CV_results.pkl")

Experiment_1_Model_Results_DCCCD = {"Default of Credit Card Client Dataset": Experiment_1_Model_Results_DCCCD}
Experiment_1_CV_Results_DCCCD = {"Default of Credit Card Client Dataset": Experiment_1_CV_Results_DCCCD}

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 2 - 2_ML_Tuned_Parameters
# =============================================================================
# --- Statlog
Experiment_2_Model_Results_SGCD = joblib.load("Default_Parameters/2_ML_Tuned_Parameters/Statlog_German_Credit_Data/model_results.pkl")
Experiment_2_CV_Results_SGCD = joblib.load("Default_Parameters/2_ML_Tuned_Parameters/Statlog_German_Credit_Data/CV_results.pkl")

Experiment_2_Model_Results_SGCD = {"Statlog German Credit Dataset": Experiment_2_Model_Results_SGCD}
Experiment_2_CV_Results_SGCD = {"Statlog German Credit Dataset": Experiment_2_CV_Results_SGCD}

# --- Default of Credit Card Client
Experiment_2_Model_Results_DCCCD = joblib.load("Default_Parameters/2_ML_Tuned_Parameters/Default_Of_Credit_Card_Client_Data/model_results.pkl")
Experiment_2_CV_Results_DCCCD = joblib.load("Default_Parameters/2_ML_Tuned_Parameters/Default_Of_Credit_Card_Client_Data/CV_results.pkl")

Experiment_2_Model_Results_DCCCD = {"Default of Credit Card Client Dataset": Experiment_2_Model_Results_DCCCD}
Experiment_2_CV_Results_DCCCD = {"Default of Credit Card Client Dataset": Experiment_2_CV_Results_DCCCD}

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 3 - Historical_Late_Split_Balanced_TDA / 1_PH_Default_Parameters
# =============================================================================
# --- Statlog
Experiment_3_Model_Results_SGCD = joblib.load("Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/model_results.pkl")
Experiment_3_CV_Results_SGCD = joblib.load("Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/CV_results.pkl")

# --- Default of Credit Card Client
Experiment_3_Model_Results_DCCCD = joblib.load("Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/model_results.pkl")
Experiment_3_CV_Results_DCCCD = joblib.load("Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/CV_results.pkl")

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 4 - Historical_Late_Split_Balanced_TDA / 2_PH_Tuned_Parameters
# =============================================================================
# --- Statlog
Experiment_4_Model_Results_SGCD = joblib.load("Historical_Late_Split_Balanced_TDA/2_PH_Tuned_Parameters/Statlog_German_Credit_Data/model_results.pkl")
Experiment_4_CV_Results_SGCD = joblib.load("Historical_Late_Split_Balanced_TDA/2_PH_Tuned_Parameters/Statlog_German_Credit_Data/CV_results.pkl")

# --- Default of Credit Card Client
Experiment_4_Model_Results_DCCCD = joblib.load("Historical_Late_Split_Balanced_TDA/2_PH_Tuned_Parameters/Default_Of_Credit_Card_Client_Data/model_results.pkl")
Experiment_4_CV_Results_DCCCD = joblib.load("Historical_Late_Split_Balanced_TDA/2_PH_Tuned_Parameters/Default_Of_Credit_Card_Client_Data/CV_results.pkl")

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 5 - 5_Mapper
# =============================================================================
# --- Mapper Algorithm. Refer to results folder

# =============================================================================
# Experiment 6 - Historical_Late_Split_Balanced_TDA / 3_H0_Only  (paper table #5)
# =============================================================================
# --- Statlog
Experiment_6_Model_Results_SGCD = joblib.load("Historical_Late_Split_Balanced_TDA/3_H0_Only/Statlog_German_Credit_Data/model_results.pkl")
Experiment_6_CV_Results_SGCD = joblib.load("Historical_Late_Split_Balanced_TDA/3_H0_Only/Statlog_German_Credit_Data/CV_results.pkl")

# --- Default of Credit Card Client
Experiment_6_Model_Results_DCCCD = joblib.load("Historical_Late_Split_Balanced_TDA/3_H0_Only/Default_Of_Credit_Card_Client_Data/model_results.pkl")
Experiment_6_CV_Results_DCCCD = joblib.load("Historical_Late_Split_Balanced_TDA/3_H0_Only/Default_Of_Credit_Card_Client_Data/CV_results.pkl")

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 7 - 7_EDA_Barcode_Statistics
# =============================================================================
# --- Statlog
Experiment_7_per_class_BS_SGCD = joblib.load("Archives/7_EDA_Barcode_Statistics/Statlog_German_Credit_Data/eda_each_class_BS.pkl")
Experiment_7_BS_SGCD = joblib.load("Archives/7_EDA_Barcode_Statistics/Statlog_German_Credit_Data/eda_entire_BS.pkl")
Experiment_7_per_class_BS_H0_Only_SGCD = joblib.load("Archives/7_EDA_Barcode_Statistics/Statlog_German_Credit_Data/eda_each_class_BS_H0.pkl")
Experiment_7_BS_H0_Only_SGCD = joblib.load("Archives/7_EDA_Barcode_Statistics/Statlog_German_Credit_Data/eda_entire_BS_H0.pkl")

# --- Default of Credit Card Clients
Experiment_7_per_class_BS_DCCCD = joblib.load("Archives/7_EDA_Barcode_Statistics/Default_Of_Credit_Card_Client_Data/eda_each_class_BS_DCCD.pkl")
Experiment_7_BS_DCCCD = joblib.load("Archives/7_EDA_Barcode_Statistics/Default_Of_Credit_Card_Client_Data/eda_entire_BS_DCCD.pkl")
Experiment_7_per_class_BS_H0_DCCCD = joblib.load("Archives/7_EDA_Barcode_Statistics/Default_Of_Credit_Card_Client_Data/eda_each_class_BS_DCCD_H0_Only.pkl")
Experiment_7_BS_H0_DCCCD = joblib.load("Archives/7_EDA_Barcode_Statistics/Default_Of_Credit_Card_Client_Data/eda_entire_BS_DCCD_H0_Only.pkl")

# =============================================================================
# Experiment 8 - 8_Dimensionality_Reduction_On_Barcode_Statistics
# =============================================================================
# --- Dimensionality Reduction on Barcode Statistics. 
# --- Multiple results here. Refer to results folder

# =============================================================================
# Experiment 9 - 9_Dimensionality_Reduction_On_Original_Dataset
# =============================================================================
# --- Dimensionality Reduction on Original Dataset. 
# --- Multiple results here. Refer to results folder

# =============================================================================
# Experiment 10 - 10_Covariance_Matrix_And_Distances
# =============================================================================
# --- Default of Credit Card Clients (L5)
Experiment_10_data_L5_distance_matrix = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Default_Of_Credit_Card_Client_Data/L5/distance_matrix_L5.pkl")
Experiment_10_data_L5_farthest_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Default_Of_Credit_Card_Client_Data/L5/distance_farthest_centriod_L5.pkl")
Experiment_10_data_L5_mean_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Default_Of_Credit_Card_Client_Data/L5/distance_mean_centriod_L5.pkl")
Experiment_10_data_L5_random_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Default_Of_Credit_Card_Client_Data/L5/distance_random_centriod_L5.pkl")

# --- Default of Credit Card Clients (L15)
Experiment_10_data_L15_distance_matrix = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Default_Of_Credit_Card_Client_Data/L15/distance_matrix_L15.pkl")
Experiment_10_data_L15_farthest_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Default_Of_Credit_Card_Client_Data/L15/distance_farthest_centriod_L15.pkl")
Experiment_10_data_L15_mean_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Default_Of_Credit_Card_Client_Data/L15/distance_mean_centriod_L15.pkl")
Experiment_10_data_L15_random_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Default_Of_Credit_Card_Client_Data/L15/distance_random_centriod_L15.pkl")

# --- Statlog (L30)
Experiment_10_data_L30_distance_matrix = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Statlog_German_Credit_Data/L30/distance_matrix_L30.pkl")
Experiment_10_data_L30_farthest_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Statlog_German_Credit_Data/L30/distance_farthest_centriod_L30.pkl")
Experiment_10_data_L30_mean_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Statlog_German_Credit_Data/L30/distance_mean_centriod_L30.pkl")
Experiment_10_data_L30_random_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Statlog_German_Credit_Data/L30/distance_random_centriod_L30.pkl")

# --- Statlog (L60)
Experiment_10_data_L60_distance_matrix = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Statlog_German_Credit_Data/L60/distance_matrix_L60.pkl")
Experiment_10_data_L60_farthest_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Statlog_German_Credit_Data/L60/distance_farthest_centriod_L60.pkl")
Experiment_10_data_L60_mean_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Statlog_German_Credit_Data/L60/distance_mean_centriod_L60.pkl")
Experiment_10_data_L60_random_centriod = joblib.load("Archives/10_Covariance_Matrix_And_Distances/Statlog_German_Credit_Data/L60/distance_random_centriod_L60.pkl")

# =============================================================================
# Experiment 11 - 11_Dropping_Correlated_Barcode_Statistics_Columns
# =============================================================================
# --- Statlog
Experiment_11_Model_Results_SGCD = joblib.load("Historical_Late_Split_Balanced_TDA/4_Dropping_Correlated_Barcode_Statistics_Columns/Statlog_German_Credit_Data/model_results.pkl")

# --- Default of Credit Card Client
Experiment_11_Model_Results_DCCCD = joblib.load("Historical_Late_Split_Balanced_TDA/4_Dropping_Correlated_Barcode_Statistics_Columns/Default_Of_Credit_Card_Client_Data/model_results.pkl")

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 12 - 12_Equivalent_Sample_Size_For_Each_Dataset
# =============================================================================
# --- Default of Credit Card Client
Experiment_12_Model_Results_DCCCD = joblib.load("Archives/12_Equivalent_Sample_Size_For_Each_Dataset/Default_Of_Credit_Card_Client_Data/model_results.pkl")
Experiment_12_CV_Results_DCCCD = joblib.load("Archives/12_Equivalent_Sample_Size_For_Each_Dataset/Default_Of_Credit_Card_Client_Data/CV_results.pkl")

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 13 - 13_Similar_Variance_Retained_After_PCA
# =============================================================================
# --- Default of Credit Card Client
Experiment_13_Model_Results_DCCCD = joblib.load("Archives/13_Similar_Variance_Retained_After_PCA/Default_Of_Credit_Card_Client_Data/model_results.pkl")
Experiment_13_CV_Results_DCCCD = joblib.load("Archives/13_Similar_Variance_Retained_After_PCA/Default_Of_Credit_Card_Client_Data/CV_results.pkl")

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 14 - 14_Mixed_Classes_Training_With_Imbalanced_Datasets
# =============================================================================
# --- Statlog
Experiment_14_Model_Results_SGCD = joblib.load("Archives/14_Mixed_Classes_Training_With_Imbalanced_Datasets/Statlog_German_Credit_Data/model_results.pkl")
Experiment_14_CV_Results_SGCD = joblib.load("Archives/14_Mixed_Classes_Training_With_Imbalanced_Datasets/Statlog_German_Credit_Data/CV_results.pkl")

# --- Default of Credit Card Client
Experiment_14_Model_Results_DCCCD = joblib.load("Archives/14_Mixed_Classes_Training_With_Imbalanced_Datasets/Default_Of_Credit_Card_Client_Data/model_results.pkl")
Experiment_14_CV_Results_DCCCD = joblib.load("Archives/14_Mixed_Classes_Training_With_Imbalanced_Datasets/Default_Of_Credit_Card_Client_Data/CV_results.pkl")

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 15 - 15_Working_With_K_in_KNN
# =============================================================================
# --- Statlog
Experiment_15_Model_Results_SGCD = joblib.load("Archives/15_Working_With_K_in_KNN/Statlog_German_Credit_Data/model_results.pkl")

# --- Default of Credit Card Client
Experiment_15_Model_Results_DCCCD = joblib.load("Archives/15_Working_With_K_in_KNN/Default_Of_Credit_Card_Client_Data/model_results.pkl")

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 16 - 16_Variance_Retained_for_Default_of_Credit_Card_Client_Dataset
# =============================================================================
# --- Default of Credit Card Client
Experiment_16_Model_Results_DCCCD = joblib.load("Archives/16_Variance_Retained_for_Default_of_Credit_Card_Client_Dataset/Default_Of_Credit_Card_Client_Data/model_results.pkl")
Experiment_16_Model_Results_3_Components_DCCCD = joblib.load("Archives/16_Variance_Retained_for_Default_of_Credit_Card_Client_Dataset/Default_Of_Credit_Card_Client_Data/model_results_using_3_components.pkl")
Experiment_16_Model_Results_5_Components_DCCCD = joblib.load("Archives/16_Variance_Retained_for_Default_of_Credit_Card_Client_Dataset/Default_Of_Credit_Card_Client_Data/model_results_using_5_components.pkl")
Experiment_16_Model_Results_7_Components_DCCCD = joblib.load("Archives/16_Variance_Retained_for_Default_of_Credit_Card_Client_Dataset/Default_Of_Credit_Card_Client_Data/model_results_using_7_components.pkl")
Experiment_16_Model_Results_9_Components_DCCCD = joblib.load("Archives/16_Variance_Retained_for_Default_of_Credit_Card_Client_Dataset/Default_Of_Credit_Card_Client_Data/model_results_using_9_components.pkl")
Experiment_16_Model_Results_11_Components_DCCCD = joblib.load("Archives/16_Variance_Retained_for_Default_of_Credit_Card_Client_Dataset/Default_Of_Credit_Card_Client_Data/model_results_using_11_components.pkl")

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 17 - 17_Distribution_For_Each_Class
# =============================================================================
# --- Distribution for each class using dimensionality reduction. 
# --- Multiple results here. Refer to results folder

# =============================================================================
# Experiment 18 - 18_Variance_Retained_for_Statlog_German_Credit_Dataset
# =============================================================================
# --- Default of Credit Card Client
Experiment_18_Model_Results_SGCD = joblib.load("Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/model_results.pkl")
Experiment_18_Model_Results_5_Components_SGCD = joblib.load("Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/model_results_using_5_components.pkl")
Experiment_18_Model_Results_7_Components_SGCD = joblib.load("Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/model_results_using_7_components.pkl")
Experiment_18_Model_Results_9_Components_SGCD = joblib.load("Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/model_results_using_9_components.pkl")
Experiment_18_Model_Results_11_Components_SGCD = joblib.load("Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/model_results_using_11_components.pkl")
Experiment_18_Model_Results_13_Components_SGCD = joblib.load("Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/model_results_using_13_components.pkl")
Experiment_18_Model_Results_15_Components_SGCD = joblib.load("Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/model_results_using_15_components.pkl")
Experiment_18_Model_Results_17_Components_SGCD = joblib.load("Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/model_results_using_17_components.pkl")
Experiment_18_Model_Results_19_Components_SGCD = joblib.load("Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/model_results_using_19_components.pkl")

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 19 - 19_Linear_Regression_For_Prediction
# =============================================================================
# --- Statlog
Experiment_19_Model_Results_SGCD = joblib.load("Historical_Late_Split_Balanced_TDA/5_Linear_Regression_For_Prediction/Statlog_German_Credit_Data/model_results.pkl")
Experiment_19_CV_Results_SGCD = joblib.load("Historical_Late_Split_Balanced_TDA/5_Linear_Regression_For_Prediction/Statlog_German_Credit_Data/CV_results.pkl")

# --- Default of Credit Card Client
Experiment_19_Model_Results_DCCCD = joblib.load("Historical_Late_Split_Balanced_TDA/5_Linear_Regression_For_Prediction/Default_Of_Credit_Card_Client_Data/model_results.pkl")
Experiment_19_CV_Results_DCCCD = joblib.load("Historical_Late_Split_Balanced_TDA/5_Linear_Regression_For_Prediction/Default_Of_Credit_Card_Client_Data/CV_results.pkl")

# CHECK RESULTS FOLDER FOR GRAPHS

# =============================================================================
# Experiment 20 - 20_Deep_Learning_For_Prediction
# =============================================================================
# --- Experiment not yet done.

# =============================================================================
# Experiment 21 - 21_Visualizing_Data_Shape_For_Barcode_Statistics_Using_TDA
# =============================================================================
# --- Visualizing data shape using Mapper. 
# --- Multiple results here. Refer to results folder

# --- Default of Credit Card Client
"""
TDA_Experiment_Data_L5 (Check the following):
    Experiment 1:
        - Lens: PCA
        - Resolution: 40
        - Overlap: 0.5
        - Clustering Algorithm: KMeans
        - N_Clusters: 2
    
    Experiment 2:
        - Lens: UMAP
        - Resolution: 30
        - Overlap: 0.5
        - Clustering Algorithm: KMeans
        - N_Clusters: 2
    
    Experiment 3:
        - Lens: UMAP
        - Resolution: 20
        - Overlap: 0.5
        - Clustering Algorithm: KMeans
        - N_Clusters: 2
"""


# =============================================================================
# BUILD TABLE
# =============================================================================
experiment_data = {
    "Experiment 1": {
        "DESCRIPTION": "Default parameter performance using ML on original dataset (Baseline Performance)",
        "RESULT": {
            "Statlog German Credit Dataset": Experiment_1_Model_Results_SGCD,
            "Default of Credit Card Client Dataset": Experiment_1_Model_Results_DCCCD
        }
    },
    "Experiment 2": {
        "DESCRIPTION": "Tuned parameter performance using ML on original dataset (Baseline Performance) for improved performance",
        "RESULT": {
            "Statlog German Credit Dataset": Experiment_2_Model_Results_SGCD,
            "Default of Credit Card Client Dataset": Experiment_2_Model_Results_DCCCD
        }
    },
    "Experiment 3": {
        "DESCRIPTION": "Default parameter performance using ML on generated barcode statistics",
        "RESULT": {
            "Statlog German Credit Dataset": Experiment_3_Model_Results_SGCD,
            "Default of Credit Card Client Dataset": Experiment_3_Model_Results_DCCCD
        }
    },
    "Experiment 4": {
        "DESCRIPTION": "Tuned parameter performance using ML on generated barcode statistics for improved performance",
        "RESULT": {
            "Statlog German Credit Dataset": Experiment_4_Model_Results_SGCD,
            "Default of Credit Card Client Dataset": Experiment_4_Model_Results_DCCCD
        }
    },
    "Experiment 5": { # Experiment 6 in Experiment Folder
        "DESCRIPTION": "Effect of H0 barcodes only from barcode statistics on model performance",
        "RESULT": {
            "Statlog German Credit Dataset": Experiment_6_Model_Results_SGCD,
            "Default of Credit Card Client Dataset": Experiment_6_Model_Results_DCCCD
        }
    },
    "Experiment 6": { # Experiment 11 in Experiment Folder
        "DESCRIPTION": "Dropping correlated barcode statistics columns before evaluating model performance",
        "RESULT": {
            "Statlog German Credit Dataset": Experiment_11_Model_Results_SGCD,
            "Default of Credit Card Client Dataset": Experiment_11_Model_Results_DCCCD
        }
    },
    "Experiment 7": { # Experiment 12 in Experiment Folder
        "DESCRIPTION": "Evaluating equivalent sample sizes for datasets (Setting the Default of Credit Card Client Dataset to same sampling size as Statlog's L30 and L60.",
        "RESULT": {
            "Default of Credit Card Client Dataset": Experiment_12_Model_Results_DCCCD
        }
    },
    "Experiment 8": { # Experiment 13 in Experiment Folder
        "DESCRIPTION": "Retaining similar variance via PCA (Setting the Default of Credit Card Dataset to same variance retained as Statlog's 89%)",
        "RESULT": {
            "Default of Credit Card Client Dataset": Experiment_13_Model_Results_DCCCD
        }
    },
    "Experiment 9": { # Experiment 14 in Experiment Folder
        "DESCRIPTION": "Training with mixed classes on imbalanced data - How do models perform under class imbalance",
        "RESULT": {
            "Statlog German Credit Dataset": Experiment_14_Model_Results_SGCD,
            "Default of Credit Card Client Dataset": Experiment_14_Model_Results_DCCCD
        }
    },
    "Experiment 10": { # Experiment 19 in Experiment Folder
        "DESCRIPTION": "Linear regression for binary prediction to understand if a straight line can easily seperate classes",
        "RESULT": {
            "Statlog German Credit Dataset": Experiment_19_Model_Results_SGCD,
            "Default of Credit Card Client Dataset": Experiment_19_Model_Results_DCCCD
        }
    }
}

dataframe_experiments = build_results_dataframe_v3(experiment_data)

# Convert to latex
latex_table = dataframe_experiments.to_latex(
    multirow=True,
    longtable=False,          # VGTC template doesn't support longtable well
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics Across Experiments",
    label="tab:results",
    escape=True              # Allows underscores in index values like L_1.82
)

with open(PAPER_TABLES / "results_table.tex", "w") as f:
    f.write(latex_table)


# =============================================================================
# BUILD TABLE - STATLOG GERMAN CREDIT DATASET
# =============================================================================
statlog_table_analysis = dataframe_experiments.xs("Statlog German Credit Dataset", level="Dataset")

# Convert to latex
latex_table = statlog_table_analysis.to_latex(
    multirow=True,
    longtable=False,          # VGTC template doesn't support longtable well
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics Across Experiments for Statlog German Credit Dataset",
    label="tab:results for Statlog German Credit Dataset",
    escape=True              # Allows underscores in index values like L_1.82
)

with open(PAPER_TABLES / "statlog_german_credit_results_table.tex", "w") as f:
    f.write(latex_table)

# =============================================================================
# BUILD TABLE - DEFAULT OF CREDIT CARD CLIENT DATASET
# =============================================================================
default_credit_card_client_table_analysis = dataframe_experiments.xs("Default of Credit Card Client Dataset", level="Dataset")

# Convert to latex
latex_table = default_credit_card_client_table_analysis.to_latex(
    multirow=True,
    longtable=False,          # VGTC template doesn't support longtable well
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics Across Experiments for Default of Credit Card Client Dataset",
    label="tab:results for Default of Credit Card Client Dataset",
    escape=True              # Allows underscores in index values like L_1.82
)

with open(PAPER_TABLES / "default_of_credit_card_client_results_table.tex", "w") as f:
    f.write(latex_table)
    
# =============================================================================
# BUILD SEPARATE EXPERIMENTS
# =============================================================================
experiment_1 = {
    "DESCRIPTION": "Default parameter performance using ML on original dataset (Baseline Performance)",
    "RESULT": {
        "Statlog German Credit Dataset": Experiment_1_Model_Results_SGCD,
        "Default of Credit Card Client Dataset": Experiment_1_Model_Results_DCCCD
    }
}

experiment_2 = {
    "DESCRIPTION": "Tuned parameter performance using ML on original dataset (Baseline Performance) for improved performance",
    "RESULT": {
        "Statlog German Credit Dataset": Experiment_2_Model_Results_SGCD,
        "Default of Credit Card Client Dataset": Experiment_2_Model_Results_DCCCD
    }
}

experiment_3 = {
    "DESCRIPTION": "Default parameter performance using ML on generated barcode statistics",
    "RESULT": {
        "Statlog German Credit Dataset": Experiment_3_Model_Results_SGCD,
        "Default of Credit Card Client Dataset": Experiment_3_Model_Results_DCCCD
    }
}

experiment_4 = {
    "DESCRIPTION": "Tuned parameter performance using ML on generated barcode statistics for improved performance",
    "RESULT": {
        "Statlog German Credit Dataset": Experiment_4_Model_Results_SGCD,
        "Default of Credit Card Client Dataset": Experiment_4_Model_Results_DCCCD
    }
}

experiment_5 = {  # Experiment 6 in Experiment Folder
    "DESCRIPTION": "Effect of H0 barcodes only from barcode statistics on model performance",
    "RESULT": {
        "Statlog German Credit Dataset": Experiment_6_Model_Results_SGCD,
        "Default of Credit Card Client Dataset": Experiment_6_Model_Results_DCCCD
    }
}

experiment_6 = {  # Experiment 11 in Experiment Folder
    "DESCRIPTION": "Dropping correlated barcode statistics columns before evaluating model performance",
    "RESULT": {
        "Statlog German Credit Dataset": Experiment_11_Model_Results_SGCD,
        "Default of Credit Card Client Dataset": Experiment_11_Model_Results_DCCCD
    }
}

experiment_7 = {  # Experiment 12 in Experiment Folder
    "DESCRIPTION": "Evaluating equivalent sample sizes for datasets (Setting the Default of Credit Card Client Dataset to same sampling size as Statlog's L30 and L60.",
    "RESULT": {
        "Default of Credit Card Client Dataset": Experiment_12_Model_Results_DCCCD
    }
}

experiment_8 = {  # Experiment 13 in Experiment Folder
    "DESCRIPTION": "Retaining similar variance via PCA (Setting the Default of Credit Card Dataset to same variance retained as Statlog's 89%)",
    "RESULT": {
        "Default of Credit Card Client Dataset": Experiment_13_Model_Results_DCCCD
    }
}

experiment_9 = {  # Experiment 14 in Experiment Folder
    "DESCRIPTION": "Training with mixed classes on imbalanced data - How do models perform under class imbalance",
    "RESULT": {
        "Statlog German Credit Dataset": Experiment_14_Model_Results_SGCD,
        "Default of Credit Card Client Dataset": Experiment_14_Model_Results_DCCCD
    }
}

experiment_10 = {  # Experiment 19 in Experiment Folder
    "DESCRIPTION": "Linear regression for binary prediction to understand if a straight line can easily seperate classes",
    "RESULT": {
        "Statlog German Credit Dataset": Experiment_19_Model_Results_SGCD,
        "Default of Credit Card Client Dataset": Experiment_19_Model_Results_DCCCD
    }
}


# Create Experiments Table
df_experiment_1 = build_results_dataframe_v3({"Experiment 1": experiment_1})
df_experiment_2 = build_results_dataframe_v3({"Experiment 2": experiment_2})
df_experiment_3 = build_results_dataframe_v3({"Experiment 3": experiment_3})
df_experiment_4 = build_results_dataframe_v3({"Experiment 4": experiment_4})
df_experiment_5 = build_results_dataframe_v3({"Experiment 5": experiment_5})
df_experiment_6 = build_results_dataframe_v3({"Experiment 6": experiment_6})
df_experiment_7 = build_results_dataframe_v3({"Experiment 7": experiment_7})
df_experiment_8 = build_results_dataframe_v3({"Experiment 8": experiment_8})
df_experiment_9 = build_results_dataframe_v3({"Experiment 9": experiment_9})
df_experiment_10 = build_results_dataframe_v3({"Experiment 10": experiment_10})

# Drop Experiment and Description Table
df_experiment_1.index = df_experiment_1.index.droplevel(["Exp.", "Desc."])
df_experiment_2.index = df_experiment_2.index.droplevel(["Exp.", "Desc."])
df_experiment_3.index = df_experiment_3.index.droplevel(["Exp.", "Desc."])
df_experiment_4.index = df_experiment_4.index.droplevel(["Exp.", "Desc."])
df_experiment_5.index = df_experiment_5.index.droplevel(["Exp.", "Desc."])
df_experiment_6.index = df_experiment_6.index.droplevel(["Exp.", "Desc."])
df_experiment_7.index = df_experiment_7.index.droplevel(["Exp.", "Desc."])
df_experiment_8.index = df_experiment_8.index.droplevel(["Exp.", "Desc."])
df_experiment_9.index = df_experiment_9.index.droplevel(["Exp.", "Desc."])
df_experiment_10.index = df_experiment_10.index.droplevel(["Exp.", "Desc."])

# Save to Latex
latex_table_1 = df_experiment_1.to_latex(
    multirow=True,
    longtable=False,
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics for Experiment 1",
    label="tab:experiment_1_results",
    escape=True
)
with open(PAPER_TABLES / "results_experiment_1.tex", "w") as f:
    f.write(latex_table_1)

latex_table_2 = df_experiment_2.to_latex(
    multirow=True,
    longtable=False,
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics for Experiment 2",
    label="tab:experiment_2_results",
    escape=True
)
with open(PAPER_TABLES / "results_experiment_2.tex", "w") as f:
    f.write(latex_table_2)

latex_table_3 = df_experiment_3.to_latex(
    multirow=True,
    longtable=False,
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics for Experiment 3",
    label="tab:experiment_3_results",
    escape=True
)
with open(PAPER_TABLES / "results_experiment_3.tex", "w") as f:
    f.write(latex_table_3)

latex_table_4 = df_experiment_4.to_latex(
    multirow=True,
    longtable=False,
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics for Experiment 4",
    label="tab:experiment_4_results",
    escape=True
)
with open(PAPER_TABLES / "results_experiment_4.tex", "w") as f:
    f.write(latex_table_4)

latex_table_5 = df_experiment_5.to_latex(
    multirow=True,
    longtable=False,
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics for Experiment 5",
    label="tab:experiment_5_results",
    escape=True
)
with open(PAPER_TABLES / "results_experiment_5.tex", "w") as f:
    f.write(latex_table_5)

latex_table_6 = df_experiment_6.to_latex(
    multirow=True,
    longtable=False,
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics for Experiment 6",
    label="tab:experiment_6_results",
    escape=True
)
with open(PAPER_TABLES / "results_experiment_6.tex", "w") as f:
    f.write(latex_table_6)

latex_table_7 = df_experiment_7.to_latex(
    multirow=True,
    longtable=False,
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics for Experiment 7",
    label="tab:experiment_7_results",
    escape=True
)
with open(PAPER_TABLES / "results_experiment_7.tex", "w") as f:
    f.write(latex_table_7)

latex_table_8 = df_experiment_8.to_latex(
    multirow=True,
    longtable=False,
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics for Experiment 8",
    label="tab:experiment_8_results",
    escape=True
)
with open(PAPER_TABLES / "results_experiment_8.tex", "w") as f:
    f.write(latex_table_8)

latex_table_9 = df_experiment_9.to_latex(
    multirow=True,
    longtable=False,
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics for Experiment 9",
    label="tab:experiment_9_results",
    escape=True
)
with open(PAPER_TABLES / "results_experiment_9.tex", "w") as f:
    f.write(latex_table_9)

latex_table_10 = df_experiment_10.to_latex(
    multirow=True,
    longtable=False,
    index=True,
    bold_rows=False,
    float_format="%.3f",
    caption="Performance Metrics for Experiment 10",
    label="tab:experiment_10_results",
    escape=True
)
with open(PAPER_TABLES / "results_experiment_10.tex", "w") as f:
    f.write(latex_table_10)
