# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:59:58 2024

@author: lEO
"""

import warnings
from utils import visualize_class_separability

# Handle Warnings
warnings.filterwarnings("ignore")

# =============================================================================
#  Load the dataset
# =============================================================================
data_paths = [r"../../../../1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/data_L5.csv",
             r"../../../../1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/data_L15.csv"]

# =============================================================================
# Visualization
# =============================================================================
viz_tsne = visualize_class_separability(
            dataset_paths = data_paths,
            method = "tsne",  # or 'pca' or 'umap'
            label_column = "label",
            save_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data",
            title = "TDA Class Separability"
)

viz_tsne_kernel_density = visualize_class_separability(
            dataset_paths = data_paths,
            method = "tsne",  # or 'pca' or 'umap'
            label_column = "label",
            save_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data/kernel_density",
            density_overlay = True,
            title = "TDA Class Separability"
)

viz_pca = visualize_class_separability(
            dataset_paths = data_paths,
            method = "pca",  # or 'pca' or 'umap'
            label_column = "label",
            save_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data",
            title = "TDA Class Separability"
)

viz_pca_kernel_density = visualize_class_separability(
            dataset_paths = data_paths,
            method = "pca",  # or 'pca' or 'umap'
            label_column = "label",
            save_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data/kernel_density",
            density_overlay = True,
            title = "TDA Class Separability"
)

viz_umap = visualize_class_separability(
            dataset_paths = data_paths,
            method = "umap",  # or 'pca' or 'umap'
            label_column = "label",
            save_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data",
            title = "TDA Class Separability"
)

viz_umap_kernel_density = visualize_class_separability(
            dataset_paths = data_paths,
            method = "umap",  # or 'pca' or 'umap'
            label_column = "label",
            save_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data/kernel_density",
            density_overlay = True,
            title = "TDA Class Separability"
)

viz_pca_3d = visualize_class_separability(
            dataset_paths = data_paths,
            method = "pca",
            plot_3d = True,
            density_overlay = False,
            save_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data/3D",
            show_legend = True,
            use_color_palette = True,
            animate_3d = False,
            title = "3D TDA Class Separability Using"
)

viz_tsne_3d = visualize_class_separability(
            dataset_paths = data_paths,
            method = "tsne",
            plot_3d = True,
            density_overlay = False,
            save_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data/3D",
            show_legend = True,
            use_color_palette = True,
            animate_3d = False,
            title = "3D TDA Class Separability Using"
)

viz_umap_3d = visualize_class_separability(
            dataset_paths = data_paths,
            method = "umap",
            plot_3d = True,
            density_overlay = False,
            save_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data/3D",
            show_legend = True,
            use_color_palette = True,
            animate_3d = False,
            title = "3D TDA Class Separability Using"
)

animated_viz_tsne_3d = visualize_class_separability(
            dataset_paths = data_paths,
            method = "tsne",
            plot_3d = True,
            density_overlay = False,
            save_path = None,
            show_legend = True,
            use_color_palette = True,
            animate_3d = True,
            save_mp4 = True,   # enable both
            save_gif = True,
            animated_plot_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data/animated_3D",
            title = "3D TDA Class Separability Using"
)

animated_viz_pca_3d = visualize_class_separability(
            dataset_paths = data_paths,
            method = "pca",
            plot_3d = True,
            density_overlay = False,
            save_path = None,
            show_legend = True,
            use_color_palette = True,
            animate_3d = True,
            save_mp4 = True,   # enable both
            save_gif = True,
            animated_plot_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data/animated_3D",
            title = "3D TDA Class Separability Using"
)

animated_viz_umap_3d = visualize_class_separability(
            dataset_paths = data_paths,
            method = "umap",
            plot_3d = True,
            density_overlay = False,
            save_path = None,
            show_legend = True,
            use_color_palette = True,
            animate_3d = True,
            save_mp4 = True,   # enable both
            save_gif = True,
            animated_plot_path = "../../../../6_Results/Archives/17_Distribution_For_Each_Class/Default_Of_Credit_Card_Client_Data/animated_3D",
            title = "3D TDA Class Separability Using"
)