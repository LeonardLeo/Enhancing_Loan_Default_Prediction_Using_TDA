# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 22:56:14 2025

@author: leona
"""

import os
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

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
# Default Data
# =============================================================================
data_default = default_data.drop("Class", axis=1)

# Run Ripser
diagrams_default = ripser(data_default)['dgms']

# Plot and save
fig, ax = plt.subplots(figsize=(8, 6))
plot_diagrams(diagrams_default, ax=ax, title="Styled Persistence Diagram for Default Data", legend=True)

ax.set_title("Styled Persistence Diagram - Default Data", fontsize=16)
ax.set_xlabel("Birth", fontsize=14)
ax.set_ylabel("Death", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.5)
ax.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()

# Save figure
default_fig_path = os.path.abspath("persistence_diagram_default.png")
plt.savefig(default_fig_path, dpi=300)
print(f"Saved default diagram to {default_fig_path}")

plt.show()

# =============================================================================
# Non-Default Data
# =============================================================================
data_non_default = balanced_non_default.drop("Class", axis=1)

# Run Ripser
diagrams_non_default = ripser(data_non_default)['dgms']

# Plot and save
fig, ax = plt.subplots(figsize=(8, 6))
plot_diagrams(diagrams_non_default, ax=ax, title="Styled Persistence Diagram for Non-Default Data", legend=True)

ax.set_title("Styled Persistence Diagram - Non-Default Data", fontsize=16)
ax.set_xlabel("Birth", fontsize=14)
ax.set_ylabel("Death", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.5)
ax.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()

# Save figure
non_default_fig_path = os.path.abspath("persistence_diagram_non_default.png")
plt.savefig(non_default_fig_path, dpi=300)
print(f"Saved non-default diagram to {non_default_fig_path}")

plt.show()

