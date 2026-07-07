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
# Get Data - L5
# =============================================================================
data_L5 = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/data_L5.csv"))
data_L5 = data_L5.sample(frac=1, random_state=0).reset_index(drop=True)
features_L5 = data_L5.drop("label", axis = 1)
label_L5 = data_L5["label"]

# =============================================================================
# Define Mapper Algorithm - L5
# =============================================================================
save_path_L5 = "../../../6_Results/21_Visualizing_Data_Shape_For_Barcode_Statistics_Using_TDA/Default_Of_Credit_Card_Client_Data/TDA_Experiments_Data_L5"

mapper_algorithm_L5 = build_mapper_viz(
                            data=features_L5,
                            resampled_data_label=label_L5,
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
                            output_dir=save_path_L5,
                            n_jobs=-1
                        )

# =============================================================================
# Get Data - L15
# =============================================================================
data_L15 = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/data_L15.csv"))
data_L15 = data_L15.sample(frac=1, random_state=0).reset_index(drop=True)
features_L15 = data_L15.drop("label", axis = 1)
label_L15 = data_L15["label"]

# =============================================================================
# Define Mapper Algorithm - L15
# =============================================================================
save_path_L15 = "../../../6_Results/21_Visualizing_Data_Shape_For_Barcode_Statistics_Using_TDA/Default_Of_Credit_Card_Client_Data/TDA_Experiments_Data_L15"

mapper_algorithm_L15 = build_mapper_viz(
                            data=features_L15,
                            resampled_data_label=label_L15,
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
                            output_dir=save_path_L15,
                            n_jobs=-1
                        )
