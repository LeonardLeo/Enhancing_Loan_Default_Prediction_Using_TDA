# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:58:18 2024

CODE: Mapper Algorithm, Experiment 21

@author: lEO
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import pandas as pd
import warnings
from utils import build_mapper_viz

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Data - L30
# =============================================================================
data_L30 = pd.read_csv(os.path.abspath("../../../../1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/data_L30.csv"))
data_L30 = data_L30.sample(frac=1, random_state=0).reset_index(drop=True)
features_L30 = data_L30.drop("label", axis = 1)
label_L30 = data_L30["label"]

# =============================================================================
# Define Mapper Algorithm - L30
# =============================================================================
save_path_L30 = "../../../../6_Results/Archives/21_Visualizing_Data_Shape_For_Barcode_Statistics_Using_TDA/Statlog_German_Credit_Data/TDA_Experiments_Data_L30"

mapper_algorithm_L30 = build_mapper_viz(
                            data=features_L30,
                            resampled_data_label=label_L30,
                            resolution=[20, 30, 40],
                            percentage_overlap=[0.3, 0.4, 0.5],
                            clustering_grid={
                                "kmeans": [{"n_clusters": 2}],
                                # "dbscan": [{"eps": 0.5, "min_samples": 5}]
                            },
                            lens_methods=["pca", "umap"],
                            lens_params={
                                "pca": {"n_components": 2},
                                "umap": {"n_components": 2, "random_state": 42}
                            },
                            color_functions=["labels"],
                            color_function_name=["Default Status"],
                            output_dir=save_path_L30,
                            n_jobs=-1
                        )

# =============================================================================
# Get Data - L60
# =============================================================================
data_L60 = pd.read_csv(os.path.abspath("../../../../1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/data_L60.csv"))
data_L60 = data_L60.sample(frac=1, random_state=0).reset_index(drop=True)
features_L60 = data_L60.drop("label", axis = 1)
label_L60 = data_L60["label"]

# =============================================================================
# Define Mapper Algorithm - L60
# =============================================================================
save_path_L60 = "../../../../6_Results/Archives/21_Visualizing_Data_Shape_For_Barcode_Statistics_Using_TDA/Statlog_German_Credit_Data/TDA_Experiments_Data_L60"

mapper_algorithm_L60 = build_mapper_viz(
                            data=features_L60,
                            resampled_data_label=label_L60,
                            resolution=[20, 30, 40],
                            percentage_overlap=[0.3, 0.4, 0.5],
                            clustering_grid={
                                "kmeans": [{"n_clusters": 2}],
                                # "dbscan": [{"eps": 0.5, "min_samples": 5}]
                            },
                            lens_methods=["pca", "umap"],
                            lens_params={
                                "pca": {"n_components": 2},
                                "umap": {"n_components": 2, "random_state": 42}
                            },
                            color_functions=["labels"],
                            color_function_name=["Default Status"],
                            output_dir=save_path_L60,
                            n_jobs=-1
                        )
