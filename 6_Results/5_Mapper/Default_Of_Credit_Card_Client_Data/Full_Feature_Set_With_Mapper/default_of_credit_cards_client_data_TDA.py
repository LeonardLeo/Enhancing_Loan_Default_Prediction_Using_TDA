# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:58:18 2024

CODE: Mapper Algorithm, Experiment 2

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
processed_data = pd.read_excel("../../../Processed_Datasets/default+of+credit+card+clients/processed_data.xlsx")
data_features = processed_data.drop(["Unnamed: 0", "default payment next month"], axis = 1)
data_label = processed_data["default payment next month"]
data_columns = data_features.columns

# =============================================================================
#  Initialize Kepler Mapper
# =============================================================================
dir_kmapper = dir(km)
mapper = km.KeplerMapper(verbose=1)

# =============================================================================
# Experiment 1: Working with FEATURE SELECTED DATASET
# Preprocess: Scale the data
# =============================================================================
# Selecting relevant columns to be scaled
selected_columns = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,  
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
scaled_data = data_features

scaler = StandardScaler()
scaled_data.iloc[:, selected_columns] = scaler.fit_transform(scaled_data.iloc[:, selected_columns])

# =============================================================================
# Define Iterative Parameters
# =============================================================================
parameters = {"Resolution": [50, 60, 70, 80, 90, 100],
              "Percentage_Overlap": [0.75, 0.6, 0.4, 0.25, 0.2, 0.15, 0.10, 0.05],
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
            experiment = f"PCA_{default_pca}/PCA{default_pca}_Res{default_resolution}_Overlap{default_perc_overlap}_KMeans{default_kmeans_nclusters}"
            
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
            
            # Dictionary of color values for visualization (lens and label)
            color_values_dict = {
                "PCA_Component": filter_func, # Lens as color value
                "Default_Status": data_label  # Default status (class labels)
            }

            # Loop through each color value, save separate visualizations
            for color_name, color_value in color_values_dict.items():
                file_path = os.path.join(folder_path, f"mapper_output_{color_name}.html")

                # Generate visualization with color toggling enabled
                mapper.visualize(
                    graph,
                    path_html = file_path,
                    title = "Loan Default Prediction Using Mapper Graph",
                    color_values = color_value,
                    color_function_name = [color_name],  # Display color value name
                    include_searchbar = True,
                    node_color_function = ['mean', 'std', 'median', 'max']
                )

                print(f"Visualization saved as '{file_path}'")