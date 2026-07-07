# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:58:18 2024

CODE: Mapper Algorithm, Experiment 3

@author: lEO
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import kmapper as km
import warnings

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Processed Default of Credit Cards Client Data
# =============================================================================
resampled_data_features = pd.read_excel("../../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/X_resampled.xlsx")
resampled_data_features = resampled_data_features.drop("Unnamed: 0", axis = 1)
resampled_data_label = pd.read_excel("../../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/y_resampled.xlsx")
resampled_data_label = resampled_data_label["default payment next month"]
data_columns = resampled_data_features.columns

# =============================================================================
#  Initialize Kepler Mapper
# =============================================================================
dir_kmapper = dir(km)
mapper = km.KeplerMapper(verbose=1)

# =============================================================================
# Experiment 1: Working with BALANCED FEATURE SELECTED DATASET
# Preprocess: Scale the data
# =============================================================================
# Selecting relevant columns to be scaled
selected_columns = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
scaled_data = resampled_data_features

scaler = StandardScaler()
scaled_data.iloc[:, selected_columns] = scaler.fit_transform(scaled_data.iloc[:, selected_columns])

# =============================================================================
# Parameters to Tune
# =============================================================================
"""
1) Filter Function / Lens (TRY DIFFERENT COMBINATION OF LENSES)
2) Number of Cubes / Resolution for the Cover
3) Percentage Overlap for the Cover
4) Type of Clustering Algorithm
5) Color Function 
6) Color Function Name
7) Add Eigen Values for PCA
"""

# =============================================================================
# Define Iterative Parameters
# =============================================================================
parameters = {"Resolution": [50, 60, 70, 80, 90, 100],
              "Percentage_Overlap": [0.25, 0.2, 0.15, 0.10, 0.05],
              "KMeans_N_CLusters": [2, 3, 4]}


for default_resolution in parameters["Resolution"]:
    for default_perc_overlap in parameters["Percentage_Overlap"]:
        for default_kmeans_nclusters in parameters["KMeans_N_CLusters"]: 
            # Defining a filter function/lens (e.g., PCA for dimensionality reduction)
            default_pca = 1
            pca = PCA(n_components = default_pca)
            filter_func = pca.fit_transform(scaled_data)
            
            # Apply the Mapper algorithm
            # ---> Parameters such as cover overlap and clusterer are customizable
            graph = mapper.map(X = scaled_data, 
                               lens = filter_func, 
                               cover = km.Cover(n_cubes = default_resolution, # Also called Resolution
                                                perc_overlap = default_perc_overlap),
                               clusterer = km.cluster.KMeans(n_clusters = default_kmeans_nclusters, 
                                                 random_state = 0),
                               )
            
            # Validate Path to Store Mapper Graph Experiment
            # ---> Define the Experiment
            experiment = f"PCA{default_pca}_Res{default_resolution}_Overlap{default_perc_overlap}_KMeans{default_kmeans_nclusters}"
            
            # Define the directory and file paths
            folder_path = experiment
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # Save the parameters used in this experiment to a text file
            params_file_path = f"{experiment}/parameters.txt"
            with open(params_file_path, "w") as f:
                f.write("Parameters for this Mapper graph:\n")
                f.write(f"PCA Components: {default_pca}\n")
                f.write(f"Resolution (n_cubes): {default_resolution}\n")
                f.write(f"Percentage Overlap: {default_perc_overlap}\n")
                f.write(f"KMeans Clusters: {default_kmeans_nclusters}\n")
            
            # Specifying Color Function Based On Lens and Label
            color_value_chosen = {"lens": filter_func, 
                                  "labels": resampled_data_label.values}
            
            for each_color_scheme, each_color_value in color_value_chosen.items():
                file_path = f"{experiment}/mapper_output_{each_color_scheme}.html"
                # Visualize the Network 
                # ---> This will save the interactive visualization as an HTML file that can be opened in a browser
                mapper.visualize(graph, 
                                 path_html = file_path, 
                                 title = "Loan Default Prediction Using Mapper Graph", 
                                 color_values = each_color_value,
                                 color_function_name = ["Default Status"],
                                 include_searchbar = True)
                
                print("Visualization saved as 'mapper_output.html'")
    