# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:59:58 2024

@author: lEO
"""

import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from utils import (train_models_on_multiple_datasets,
                   generate_landmark_sets,
                   compute_barcodes_from_multiple_landmarks,
                   build_final_barcode_statistics_data,
                   store_results)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Handle warnings
warnings.filterwarnings("ignore")

# Step 1: Load and normalize the dataset
data = pd.read_excel("../../../1_Data/Processed_Datasets/Statlog_German_Credit_Data/processed_data.xlsx")

# Split dataset into features (X) and target (y)
X = data.drop(columns = ["Class"])
y = data["Class"]

# Normalize the features
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# APPLYING PCA
pca = PCA(n_components = 15)
X_reduced = pd.DataFrame(pca.fit_transform(X_normalized), columns = [f"PCA_{num}" for num in range(1, 16)])
# Get the explained variance ratio for each PCA setup
variance_ratio = pca.explained_variance_ratio_.sum()
# Print or log the amount of variance retained
print(f"Variance retained with PCA components: {variance_ratio:.2%}")


# Combine normalized features with the target
reduced_data = X_reduced.copy()
reduced_data["Class"] = y

# Separate data into default and non-default
default_data = reduced_data[reduced_data["Class"] == 1].reset_index(drop = True)
non_default_data = reduced_data[reduced_data["Class"] == 0].reset_index(drop = True)

# Ensure class balance (optional: undersample non-default to match default count)
n_samples = len(default_data)
balanced_non_default = non_default_data.sample(n = n_samples, random_state = 42)



# =============================================================================
# SET SAMPLING PERCENTAGE
# =============================================================================
percentages = [30, 60]

# =============================================================================
# LANDMARK SELECTION
# =============================================================================
generate_landmark_sets(class_label_and_data = {"default": default_data.copy().drop("Class", 
                                                                                   axis = 1),
                                               "non-default": balanced_non_default.copy().drop("Class", 
                                                                                               axis = 1)}, 
                       landmark_percentages = percentages, 
                       dataset_to_use = "Statlog",
                       n_files_per_percentage = 500,
                       experiment_name = "4_PH_Tuned_Parameters")

# =============================================================================
# COMPUTE BARCODE STATISTICS
# =============================================================================
compute_barcodes_from_multiple_landmarks(landmark_percentages = percentages, # Should be same as used in GENERATE LANDMARK SETS function
                                         landmark_dir = "../../../1_Data/Landmark_Sets/Statlog_German_Credit_Data/4_PH_Tuned_Parameters", 
                                         barcode_output_dir = "../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/4_PH_Tuned_Parameters", 
                                         dim = 2, 
                                         label = {1: "default",
                                                  0: "non-default"})

# =============================================================================
# MERGE BARCODE STATISTICS - Create TDA Dataset for Model Building
# =============================================================================
build_final_barcode_statistics_data(landmark_percentages = percentages,
                                    barcode_dir = "../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/4_PH_Tuned_Parameters",
                                    output_dir = "../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/4_PH_Tuned_Parameters",
                                    label = {1: "default",
                                             0: "non-default"})

# =============================================================================
# TRAIN MACHINE LEARNING MODEL WITH DEFAULT PARAMETERS
# =============================================================================
paths = ["../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/4_PH_Tuned_Parameters/data_L30.csv",  
         "../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/4_PH_Tuned_Parameters/data_L60.csv"]

model_configs = {
                "svm": {
                    "model": SVC(),
                    "params": {
                        "C": [0.1, 1, 10, 100],  # Regularization parameter
                        "kernel": ["linear", "rbf", "poly", "sigmoid"],  # Various kernel types
                        "degree": [2, 3, 4],  # Degree for polynomial kernel
                        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1]  # Kernel coefficient
                    }
                },
                "knn": {
                    "model": KNeighborsClassifier(),
                    "params": {
                        "n_neighbors": [3, 5, 7, 10],  # Number of neighbors
                        "weights": ["uniform", "distance"],  # Weight function
                        "p": [1, 2],  # Manhattan (L1) or Euclidean (L2) distances
                        "leaf_size": [10, 20, 30, 50],  # Size of leaf in the tree
                        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]  # Algorithm used
                    }
                },
                "xgb": {
                    "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7]
                    }
                },
                "logistic": {
                    "model": LogisticRegression(),
                    "params": {
                        "C": [0.01, 0.1, 1, 10, 100],  # Inverse regularization strength
                        "solver": ["liblinear", "lbfgs", "sag", "saga", "newton-cg"],  # Optimization algorithm
                        "penalty": ["l1", "l2", "elasticnet", "none"],  # Regularization terms
                        "max_iter": [100, 200, 500]  # Maximum number of iterations
                    }
                },
                "random_forest": {
                    "model": RandomForestClassifier(),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 5, 10, None],
                        "min_samples_split": [2, 5, 10]
                    }
                },
            }

model_results = train_models_on_multiple_datasets(data_paths = paths,
                                                  model_configs = model_configs,
                                                  target_column = "label",
                                                  test_size = 0.2,
                                                  scoring_metric = "f1",
                                                  scale_features = True,
                                                  random_state = 42,
                                                  n_splits_kfold = 5)

print(model_results)

# =============================================================================
# STORE MODEL RESULTS
# =============================================================================
save_path = "../../../6_Results/4_PH_Tuned_Parameters/Statlog_German_Credit_Data"

store_results(path = save_path, 
              save_name = "model_results", 
              result_object = model_results)
