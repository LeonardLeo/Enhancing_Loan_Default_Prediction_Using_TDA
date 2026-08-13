# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:59:58 2024

@author: lEO
"""

import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from utils import (train_multiple_dataset_tda,
                   generate_landmark_sets_v2,
                   compute_barcodes_from_multiple_landmarks,
                   build_final_barcode_statistics_data,
                   store_results)

# Handle warnings
warnings.filterwarnings("ignore")

# Step 1: Load and normalize the dataset
data = pd.read_excel(r"../../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/processed_data.xlsx")

# Split dataset into features (X) and target (y)
X = data.drop(columns = ["default payment next month", "Unnamed: 0"])
y = data["default payment next month"]

# Normalize the features
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# APPLYING PCA
pca = PCA(n_components = 7)
X_reduced = pd.DataFrame(pca.fit_transform(X_normalized), columns = [f"PCA_{num}" for num in range(1, 8)])
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
percentages = [5, 15]

# =============================================================================
# LANDMARK SELECTION
# =============================================================================
files_per_class = {'default': 200, 'non-default': 800}
generate_landmark_sets_v2(class_label_and_data = {"default": default_data.copy().drop("Class", 
                                                                                   axis = 1),
                                                  "non-default": balanced_non_default.copy().drop("Class", 
                                                                                               axis = 1)}, 
                       landmark_percentages = percentages, 
                       dataset_to_use = "dataset2",
                       num_files_per_class = files_per_class,
                       experiment_name = "14_Mixed_Classes_Training_With_Imbalanced_Datasets")

# =============================================================================
# COMPUTE BARCODE STATISTICS
# =============================================================================
compute_barcodes_from_multiple_landmarks(landmark_percentages = percentages, # Should be same as used in GENERATE LANDMARK SETS function
                                         landmark_dir = "../../../../1_Data/Landmark_Sets/Default_Of_Credit_Card_Client_Data/14_Mixed_Classes_Training_With_Imbalanced_Datasets", 
                                         barcode_output_dir = "../../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/14_Mixed_Classes_Training_With_Imbalanced_Datasets", 
                                         dim = 2, 
                                         label = {1: "default",
                                                  0: "non-default"})

# =============================================================================
# MERGE BARCODE STATISTICS - Create TDA Dataset for Model Building
# =============================================================================
build_final_barcode_statistics_data(landmark_percentages = percentages,
                                    barcode_dir = "../../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/14_Mixed_Classes_Training_With_Imbalanced_Datasets",
                                    output_dir = "../../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/14_Mixed_Classes_Training_With_Imbalanced_Datasets",
                                    label = {1: "default",
                                             0: "non-default"})

# =============================================================================
# TRAIN MACHINE LEARNING MODEL WITH DEFAULT PARAMETERS
# =============================================================================
paths = ["../../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/14_Mixed_Classes_Training_With_Imbalanced_Datasets/data_L5.csv", 
         "../../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/14_Mixed_Classes_Training_With_Imbalanced_Datasets/data_L15.csv",]

model_results = train_multiple_dataset_tda(path_datasets = paths,
                                           y_col_name = "label",
                                           test_size = 0.2,
                                           random_state = 42,
                                           xgb = {"eval_metric":"logloss"})

print(model_results)

# =============================================================================
# STORE MODEL RESULTS
# =============================================================================
save_path = "../../../../6_Results/Archives/14_Mixed_Classes_Training_With_Imbalanced_Datasets/Default_Of_Credit_Card_Client_Data"

store_results(path = save_path, 
              save_name = "model_results", 
              result_object = model_results)
