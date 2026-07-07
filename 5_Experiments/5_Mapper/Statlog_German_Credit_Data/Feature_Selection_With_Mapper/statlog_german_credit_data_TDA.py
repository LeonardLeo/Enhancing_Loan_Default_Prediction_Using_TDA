# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:58:18 2024

@author: Leo
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import numpy as np
import kmapper as km
import warnings

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Processed Default of Credit Cards Client Data
# =============================================================================
processed_data = pd.read_excel("../../../Processed_Datasets/statlog+german+credit+data/processed_data.xlsx")
data_features = pd.read_excel("../../../Processed_Datasets/statlog+german+credit+data/X_train.xlsx")
data_label = pd.read_excel("../../../Processed_Datasets/statlog+german+credit+data/y_train.xlsx")
resampled_data_features = pd.read_excel("../../../Processed_Datasets/statlog+german+credit+data/X_resampled.xlsx")
resampled_data_label = pd.read_excel("../../../Processed_Datasets/statlog+german+credit+data/y_resampled.xlsx")
data_columns = data_features.columns

# =============================================================================
#  Initialize Kepler Mapper
# =============================================================================
mapper = km.KeplerMapper(verbose=1)

# =============================================================================
# Experiment 1: Working with FEATURE SELECTED DATASET
# Preprocess: Scale the data
# =============================================================================
# Selecting relevant columns to be scaled
scaler = StandardScaler()
scaled_data = data_features.copy()  # Copy to avoid modifying the original data

scaled_data.loc[:, ["Duration", "Credit amount"]] = scaler.fit_transform(scaled_data.loc[:, ["Duration", "Credit amount"]])

def lens_pca():
    # =============================================================================
    # Precompute PCA components to avoid redundant computations in the loop
    # =============================================================================
    # Perform PCA with n_components = 1 and n_components = 2, storing both the transformed data and explained variance
    pca_1 = PCA(n_components=1)
    pca_2 = PCA(n_components=2)
    
    # Fit and transform the scaled data with PCA
    pca_results = {
        1: pca_1.fit_transform(scaled_data),
        2: pca_2.fit_transform(scaled_data)
    }
    
    # Get the explained variance ratio for each PCA setup
    variance_ratio_1 = pca_1.explained_variance_ratio_.sum()  # Information retained with 1 component
    variance_ratio_2 = pca_2.explained_variance_ratio_.sum()  # Information retained with 2 components
    
    # Print or log the amount of variance retained
    print(f"Variance retained with 1 PCA component: {variance_ratio_1:.2%}")
    print(f"Variance retained with 2 PCA components: {variance_ratio_2:.2%}")
    
    # =============================================================================
    # Define Iterative Parameters
    # =============================================================================
    parameters = {
        "Resolution": [50, 60, 70, 80, 90, 100],
        "Percentage_Overlap": [0.75, 0.6, 0.4, 0.25, 0.2, 0.15, 0.10, 0.05],
        "KMeans_N_Clusters": [2, 3, 4]
    }
    
    # =============================================================================
    # Run Mapper Algorithm with Optimized Iterative Loop
    # =============================================================================
    default_pca = 1  # Define PCA component setting once
    
    for default_resolution in parameters["Resolution"]:
        for default_perc_overlap in parameters["Percentage_Overlap"]:
            for default_kmeans_nclusters in parameters["KMeans_N_Clusters"]:
                # Use precomputed PCA result
                filter_func = pca_results[default_pca]
    
                # Apply the Mapper algorithm
                graph = mapper.map(
                    X = scaled_data,
                    lens = filter_func,
                    cover = km.Cover(n_cubes = default_resolution, 
                                      perc_overlap = default_perc_overlap),
                    clusterer = km.cluster.KMeans(n_clusters = default_kmeans_nclusters, 
                                                  random_state = 0)
                )
    
                # Define the experiment identifier
                experiment = f"PCA_{default_pca}/PCA{default_pca}_Res{default_resolution}_Overlap{default_perc_overlap}_KMeans{default_kmeans_nclusters}"
    
                # Define the directory and file paths
                folder_path = experiment
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
    
                # Save the parameters used in this experiment to a text file
                params_file_path = os.path.join(folder_path, "parameters.txt")
                with open(params_file_path, "w") as f:
                    f.write("Parameters for this Mapper graph:\n")
                    f.write(f"PCA Components: {default_pca}\n")
                    f.write(f"Resolution (n_cubes): {default_resolution}\n")
                    f.write(f"Percentage Overlap: {default_perc_overlap}\n")
                    f.write(f"KMeans Clusters: {default_kmeans_nclusters}\n")
    
                # Dictionary of color values for visualization (lens and label)
                color_values_dict = {
                    "PCA_Component": filter_func, # Lens as color value
                    "Default_Status": data_label.iloc[:, 0].values  # Default status (class labels)
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
    

def lens_custom():
    # =============================================================================
    # Compute Isolation Forest Lens
    # =============================================================================
    isolation_forest = IsolationForest(n_estimators = 100, 
                                       contamination = 'auto', 
                                       random_state = 0)
    isolation_forest.fit(scaled_data)
    isolation_forest_lens = isolation_forest.decision_function(scaled_data)

    # =============================================================================
    # Compute L2 Norm Lens
    # =============================================================================
    l2_norm_lens = np.linalg.norm(scaled_data, axis=1)

    # Combine lenses into a single array
    lens = np.c_[isolation_forest_lens, l2_norm_lens]

    # =============================================================================
    # Define Iterative Parameters
    # =============================================================================
    parameters = {
        "Resolution": [50, 60, 70, 80, 90, 100],
        "Percentage_Overlap": [0.75, 0.6, 0.4, 0.25, 0.2, 0.15, 0.10, 0.05],
        "KMeans_N_Clusters": [2, 3, 4]
    }

    # =============================================================================
    # Run Mapper Algorithm with Optimized Iterative Loop
    # =============================================================================
    for default_resolution in parameters["Resolution"]:
        for default_perc_overlap in parameters["Percentage_Overlap"]:
            for default_kmeans_nclusters in parameters["KMeans_N_Clusters"]:
                # Apply the Mapper algorithm with the custom lens
                graph = mapper.map(
                    X = scaled_data,
                    lens = lens,
                    cover = km.Cover(n_cubes = default_resolution, 
                                   perc_overlap = default_perc_overlap),
                    clusterer = km.cluster.KMeans(n_clusters = default_kmeans_nclusters, 
                                                random_state=0)
                )

                # Define the experiment identifier
                experiment = f"CustomLens/Res{default_resolution}_Overlap{default_perc_overlap}_KMeans{default_kmeans_nclusters}"

                # Define the directory and file paths
                folder_path = experiment
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # Save the parameters used in this experiment to a text file
                params_file_path = os.path.join(folder_path, "parameters.txt")
                with open(params_file_path, "w") as f:
                    f.write("Parameters for this Mapper graph:\n")
                    f.write("Lens: Isolation Forest and L2 Norm\n")
                    f.write(f"Resolution (n_cubes): {default_resolution}\n")
                    f.write(f"Percentage Overlap: {default_perc_overlap}\n")
                    f.write(f"KMeans Clusters: {default_kmeans_nclusters}\n")

                # Define combined color values and color function names
                color_values = np.c_[isolation_forest_lens, l2_norm_lens, data_label.iloc[:, 0].values]
                color_function_names = ['Isolation Forest', 'L2-norm', 'Default Status']

                # Generate a single visualization with all color values
                file_path = os.path.join(folder_path, "mapper_output_combined.html")
                mapper.visualize(
                    graph,
                    path_html = file_path,
                    title = "Loan Default Prediction Using Mapper Graph",
                    color_values = color_values,
                    color_function_name = color_function_names,
                    include_searchbar = True,
                    node_color_function = ['mean', 'std', 'median', 'max']
                )

                print(f"Visualization saved as '{file_path}'")

# Run Lens
lens_custom()
     