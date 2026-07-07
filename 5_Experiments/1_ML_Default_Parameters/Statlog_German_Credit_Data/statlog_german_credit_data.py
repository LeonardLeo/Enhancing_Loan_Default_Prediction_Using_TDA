# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:32:37 2024

@author: lEO
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, SelectFpr
from ydata_profiling import ProfileReport
from imblearn.over_sampling import ADASYN
from ucimlrepo import fetch_ucirepo
from utils import (eda, 
                   data_preprocessing_pipeline,
                   save_python_object_using_joblib,
                   fix_string,
                   store_results,
                   train_dataset)
import joblib
import warnings

# =============================================================================
# Remove Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
# =============================================================================
raw_data = fetch_ucirepo(id = 144)
# ---> Dataset 1 (German Data)
data = fetch_ucirepo(id = 144).data.original

feature_desc = fetch_ucirepo(id = 144).variables
feature_name = {key:value for key, value in zip(feature_desc.name, feature_desc.description) if key != "class"}

data.rename(columns = feature_name, inplace = True)
# ---> Dataset 2
dataset = pd.read_csv("../../../1_Data/Datasets/Statlog_German_Credit_Data/german.data-numeric", 
                      delim_whitespace=True, 
                      header=None)

"""
Comparison Between German Credit Data (DATASET 1) and German Credit Numeric Data (DATASET 2):
    The differences between the two datasets can be seen in the column names:
        
        - Status of Existing Checking Account (INDEX 0)
        - Duration (INDEX 1)
        - Credit History (INDEX 2)
        - Purpose (None)
        - Credit Amount (INDEX 3)
        - Savings Account/Bonds (INDEX 4)
        - Present Employment Since (INDEX 5)
        - Installment Rate in Percentage of Disposable Income (None)
        - Personal Status and Sex (INDEX 6)
     ✔️ - Other Debtors/Guarantors (INDEX 17 & 18 as DUMMY VARIABLES)
        - Present Residence Since (INDEX 7)
        - Property (INDEX 8)
        - Age (INDEX 9)
        - Other Installment Plans (INDEX 10)
     ✔️ - Housing (INDEX 19 & 20 as DUMMY VARIABLES)
        - Number of Existing Credits at this Bank (INDEX 11)
     ✔️ - Job (INDEX 21, 22, & 23 as DUMMY VARIABLES)
        - Number of People Being Liable to Provide Maintenance for (INDEX 12)
        - Telephone (INDEX 13)
        - Foreign Worker (INDEX 14)
        - Class (INDEX 24)

    The following columns from German Credit Data were dropped in the German Credit
    Numeric Data:
        - Purpose (Categorical)
        - Installment Rate in Percentage of Disposable Income (Numeric)
    
    The following columns from the German Credit Numeric Data were not detected in the
    original German Credit Data:
        - Index 15 (Paired Dummy Variable with Index 16)
        - Index 16 (Paired Dummy Variable with Index 15)
"""

"""
Next, we should also consider replacing the values in the "CLASS" variable from 
1 and 2 to 0 and 1. This is to allow the machine properly read the variables and
avoid encountering any errors.
"""

# =============================================================================
# EXPLORATORY DATA ANALYSIS FOR GERMAN CREDIT DATA (GCD)
# =============================================================================
initial_eda_gcd = eda(data)
save_python_object_using_joblib(python_object = initial_eda_gcd,
                                dataset_to_use = "statlog",
                                save_item = "eda",
                                save_name = "initial_EDA",
                                experiment_name = "1_ML_Default_Parameters")

# =============================================================================
# RENAMING AND REPLACING COLUMNS IN GERMAN CREDIT NUMERIC DATA (GCND)
# =============================================================================
columns_name = {0: "Status of existing checking account",
                1: "Duration",
                2: "Credit history",
                3: "Credit amount",
                4: "Savings account/bonds",
                5: "Present employment since",
                6: "Personal status and sex",
                7: "Present residence since",
                8: "Property",
                9: "Age",
                10: "Other installment plans",
                11: "Number of existing credits at this bank",
                12: "Number of people being liable to provide maintenance for",
                13: "Telephone",
                14: "Foreign worker",
                15: "Unknown_dummy1",
                16: "Unknown_dummy2",
                17: "Other_debtors_or_gurantors_A101",
                18: "Other_debtors_or_gurantors_A102",
                19: "Housing_A151",
                20: "Housing_A152",
                21: "Job_A171",
                22: "Job_A172",
                23: "Job_A173",
                24: "Class"}

dataset.rename(columns = columns_name, inplace = True)
dataset["Class"].replace({1: 0, 2: 1}, inplace = True)

# =============================================================================
# WORKING WITH GERMAN CREDIT NUMERIC DATA (GCND)
# =============================================================================
# Exploratory Data Analysis
initial_eda_gcnd = eda(dataset)
save_python_object_using_joblib(python_object = initial_eda_gcd,
                                dataset_to_use = "statlog",
                                save_item = "eda",
                                save_name = "initial_EDA_numeric_data",
                                experiment_name = "1_ML_Default_Parameters")

# ---> Pandas Profiling
profile = ProfileReport(dataset,
                        explorative = True,
                        title = 'Statlog German Credit Numeric Dataset - EDA')
profile.to_file(os.path.abspath("../../../2_Pandas_Profiling_Report/Statlog_German_Credit_Data/initial_EDA.html"))

# =============================================================================
# THINGS TO CONSIDER
# =============================================================================
"""
1) Do we create new variables that capture non-linear relationship among variables
2) Handling correlated variables
3) Consider dropping the AGE column (TRAINING UNBIASED MODELS)
"""

# =============================================================================
# Preprocessing data
# =============================================================================
# Data Transformation for Skewed Columns
dataset = data_preprocessing_pipeline(dataset,
                                      log_col = ["Credit amount", 
                                                 "Duration", "Age"])

# =============================================================================
# Save Clean Dataset
# =============================================================================
save_python_object_using_joblib(python_object = dataset,
                                dataset_to_use = "statlog",
                                save_item = "processed",
                                save_name = "processed_data",
                                experiment_name = "1_ML_Default_Parameters")

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
final_eda = eda(dataset)
save_python_object_using_joblib(python_object = final_eda,
                                dataset_to_use = "statlog",
                                save_item = "eda",
                                save_name = "final_EDA",
                                experiment_name = "1_ML_Default_Parameters")

# =============================================================================
# Select dependent and independent variables
# =============================================================================
X = dataset.drop("Class", axis=1)
y = dataset.Class

# =============================================================================
# Split dataset into training and test data
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 0,
                                                    stratify = y
                                                    )

# =============================================================================
# Value counts for the target
# =============================================================================
count_target_category_train = y_train.value_counts()
count_target_category_test = y_test.value_counts()

# =============================================================================
# Feature selection
# =============================================================================
# ---> Statistical Approach
selector = SelectFpr(score_func = f_classif)
X_train = pd.DataFrame(selector.fit_transform(X_train, y_train),
                        columns=selector.get_feature_names_out())
X_test = pd.DataFrame(selector.transform(X_test),
                      columns=selector.get_feature_names_out())
# ---> Statistical score for features with associated p-value
feature_info = pd.DataFrame({"Features": selector.feature_names_in_,
                              "Scores": np.around(selector.scores_, 2),
                              "P-Value": np.around(selector.pvalues_, 2)})

# Save the DataFrame (Feature Info) as an Excel file
save_python_object_using_joblib(python_object = feature_info,
                                dataset_to_use = "statlog",
                                save_item = "feature_info",
                                save_name = "feature_selection_info",
                                experiment_name = "1_ML_Default_Parameters")

# =============================================================================
# Resampling minority class
# =============================================================================
# ---> Resampling the training data seperately
resampler_train = ADASYN(random_state = 0)
X_resampled, y_resampled = resampler_train.fit_resample(X_train, y_train)

# =============================================================================
# Save Features and Label
# =============================================================================
save_python_object_using_joblib(python_object = X_resampled,
                                dataset_to_use = "statlog",
                                save_item = "processed",
                                save_name = "X_resampled",
                                experiment_name = "1_ML_Default_Parameters")
save_python_object_using_joblib(python_object = X_train,
                                dataset_to_use = "statlog",
                                save_item = "processed",
                                save_name = "X_train",
                                experiment_name = "1_ML_Default_Parameters")
save_python_object_using_joblib(python_object = X_test,
                                dataset_to_use = "statlog",
                                save_item = "processed",
                                save_name = "X_test",
                                experiment_name = "1_ML_Default_Parameters")
save_python_object_using_joblib(python_object = y_resampled,
                                dataset_to_use = "statlog",
                                save_item = "processed",
                                save_name = "y_resampled",
                                experiment_name = "1_ML_Default_Parameters")
save_python_object_using_joblib(python_object = y_train,
                                dataset_to_use = "statlog",
                                save_item = "processed",
                                save_name = "y_train",
                                experiment_name = "1_ML_Default_Parameters")
save_python_object_using_joblib(python_object = y_test,
                                dataset_to_use = "statlog",
                                save_item = "processed",
                                save_name = "y_test",
                                experiment_name = "1_ML_Default_Parameters")

# =============================================================================
# Value counts for the target
# =============================================================================
count_target_resampled_train = y_resampled.value_counts()

# =============================================================================
# Normalize the features
# =============================================================================
scaler_data = MinMaxScaler()
X_columns = X_resampled.columns
X_resampled = pd.DataFrame(scaler_data.fit_transform(X_resampled), columns = X_columns)
X_train = pd.DataFrame(scaler_data.transform(X_train), columns = X_columns)
X_test = pd.DataFrame(scaler_data.transform(X_test), columns = X_columns)

# =============================================================================
# MODEL BUILDING
# =============================================================================
training_results = train_dataset(X_resampled,
                                 y_resampled,
                                 X_test,
                                 y_test)

# =============================================================================
# STORE MODEL RESULTS
# =============================================================================
save_path = "../../../6_Results/1_ML_Default_Parameters/Statlog_German_Credit_Data"

store_results(path = save_path, 
              save_name = "model_results", 
              result_object = training_results)

# # =============================================================================
# # Selecting the best features
# # =============================================================================
# get_best_features = pd.DataFrame({"Features": X_resampled.columns,
#                                   "Scores": model.feature_importances_})

# # Save the DataFrame (Feature Info) as an Excel file
# get_best_features.to_excel('../Python_Objects/statlog+german+credit+data/Feature_Info/best_features_from_model.xlsx', index = True)
