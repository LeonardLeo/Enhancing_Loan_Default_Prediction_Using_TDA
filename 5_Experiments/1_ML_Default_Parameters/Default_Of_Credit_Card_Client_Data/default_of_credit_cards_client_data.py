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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, SelectFpr
from ydata_profiling import ProfileReport
from imblearn.over_sampling import ADASYN
from utils import (eda, data_preprocessing_pipeline, store_results, fix_string)
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             classification_report,
                             confusion_matrix)
import joblib
import warnings

# =============================================================================
# Initialize a Random State 
# =============================================================================
state = 0

# =============================================================================
# Remove Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
# =============================================================================
# ---> To be used as copy and reference
data = pd.read_excel(os.path.abspath("../../../1_Data/Datasets/Default_Of_Credit_Card_Client_Data/default of credit card clients.xls"), header = 1)
# ---> To be used in data mining tasks
dataset = data.copy()

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
initial_eda = eda(dataset)
# ---> Pandas Profiling
profile = ProfileReport(dataset,
                        explorative = True,
                        title = 'Default of Credit Card Client Dataset - EDA')
profile.to_file(os.path.abspath("../../../2_Pandas_Profiling_Report/Default_Of_Credit_Card_Client_Data/initial_EDA.html"))


# ---> (1) Save EDA Output into Excel File
with pd.ExcelWriter(os.path.abspath('../../../3_Python_Objects/Default_Of_Credit_Card_Client_Data/initial_eda.xlsx'), 
                    engine = 'xlsxwriter') as writer:
    # Loop through the dictionary and save each dataframe/series to a different sheet
    for sheet_name, dataframe in initial_eda.items():
        sheet_name = fix_string(sheet_name)
        if len(sheet_name) > 18:
            sheet_name = sheet_name[:18] + "..."
        if isinstance(dataframe, pd.DataFrame):
            dataframe.to_excel(writer, sheet_name = sheet_name, index = True)
        elif isinstance(dataframe, pd.Series):
            dataframe.to_frame().to_excel(writer, sheet_name = sheet_name, index = True)

# ---> (2) Save EDA Output into Excel File
item = 1
for name_sheet, dictionary in initial_eda.items():
    if isinstance(dictionary, dict):
        with pd.ExcelWriter(os.path.abspath(f'../../../3_Python_Objects/Default_Of_Credit_Card_Client_Data/initial_eda_other{item}.xlsx'), 
                            engine = 'xlsxwriter') as writer:
            # Loop through the dictionary and save each dictionary to a different sheet
            for sheet_name, dataframe in dictionary.items():
                sheet_name = fix_string(sheet_name)
                if len(sheet_name) > 18 and len(dictionary) > 0:
                    sheet_name = sheet_name[:18] + "..."
                if isinstance(dataframe, pd.DataFrame):
                    dataframe.to_excel(writer, sheet_name = sheet_name, index = True)
                elif isinstance(dataframe, pd.Series):
                    dataframe.to_frame().to_excel(writer, sheet_name = sheet_name, index = True)
        item += 1

# =============================================================================
# Preprocessing data
# =============================================================================
# Data Transformation for Skewed Columns
dataset = data_preprocessing_pipeline(dataset,
                                      log_col = ["AGE", "LIMIT_BAL"],
                                      drop_columns = "ID")

# Drop Duplicates
dataset = dataset.drop_duplicates()

# # Creating new features
# # ---> New Features From the Distribution of the Data
# dataset["Age_Group"], bins1 = pd.cut(dataset["Age"], 10, retbins = True, precision = 0, labels = False)
# dataset["Credit_Group"], bins3 = pd.cut(dataset["Credit amount"], 10, retbins = True, precision = 0, labels = False)
# dataset["Duration_Group"], bins2 = pd.qcut(dataset["Duration"], 5, retbins = True, precision = 0, labels = False)

# =============================================================================
# Save Clean Dataset
# =============================================================================
dataset.to_excel("../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/processed_data.xlsx")

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
final_eda = eda(dataset)

# ---> (1) Save EDA Output into Excel File
with pd.ExcelWriter(os.path.abspath('../../../3_Python_Objects/Default_Of_Credit_Card_Client_Data/final_eda.xlsx'), 
                    engine = 'xlsxwriter') as writer:
    # Loop through the dictionary and save each dataframe/series to a different sheet
    for sheet_name, dataframe in final_eda.items():
        sheet_name = fix_string(sheet_name)
        if len(sheet_name) > 18 and len(dictionary) > 0:
            sheet_name = sheet_name[:18] + "..."
        if isinstance(dataframe, pd.DataFrame):
            dataframe.to_excel(writer, sheet_name = sheet_name, index = True)
        elif isinstance(dataframe, pd.Series):
            dataframe.to_frame().to_excel(writer, sheet_name = sheet_name, index = True)

# ---> (2) Save EDA Output into Excel File
item = 1
for name_sheet, dictionary in final_eda.items():
    if isinstance(dictionary, dict):
        with pd.ExcelWriter(os.path.abspath(f'../../../3_Python_Objects/Default_Of_Credit_Card_Client_Data/final_eda_other{item}.xlsx'), 
                            engine = 'xlsxwriter') as writer:
            # Loop through the dictionary and save each dictionary to a different sheet
            for sheet_name, dataframe in dictionary.items():
                sheet_name = fix_string(sheet_name)
                if len(sheet_name) > 18 and len(dictionary) > 0:
                    sheet_name = sheet_name[:18] + "..."
                if isinstance(dataframe, pd.DataFrame):
                    dataframe.to_excel(writer, sheet_name = sheet_name, index = True)
                elif isinstance(dataframe, pd.Series):
                    dataframe.to_frame().to_excel(writer, sheet_name = sheet_name, index = True)
        item += 1

# =============================================================================
# THINGS TO CONSIDER
# =============================================================================
"""
1) Do we create new variables that capture non-linear relationship among variables
2) Handling correlated variables
3) Consider dropping the SEX column (TRAINING UNBIASED MODELS)
"""

# =============================================================================
# Select dependent and independent variables
# =============================================================================
X = dataset.drop("default payment next month", axis = 1)
y = dataset["default payment next month"]

# =============================================================================
# Column Names
# =============================================================================
feature_names = X.columns

# =============================================================================
# Split dataset into training and test data
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = state,
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
                        columns = selector.get_feature_names_out())
X_test = pd.DataFrame(selector.transform(X_test),
                      columns = selector.get_feature_names_out())
# ---> Statistical score for features with associated p-value
feature_info = pd.DataFrame({"Features": selector.feature_names_in_,
                              "Scores": np.around(selector.scores_, 2),
                              "P-Value": np.around(selector.pvalues_, 2)})

# Save the DataFrame (Feature Info) as an Excel file
feature_info.to_excel(os.path.abspath('../../../3_Python_Objects/Default_Of_Credit_Card_Client_Data/feature_selection_info.xlsx'), index = False)

# =============================================================================
# Resampling minority class
# =============================================================================
# ---> Resampling the training data seperately
resampler_train = ADASYN(random_state = state)
X_resampled, y_resampled = resampler_train.fit_resample(X_train, y_train)

# =============================================================================
# Save Features and Label
# =============================================================================
X_resampled.to_excel("../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/X_resampled.xlsx")
X_train.to_excel("../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/X_train.xlsx")
X_test.to_excel("../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/X_test.xlsx")
y_resampled.to_excel("../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/y_resampled.xlsx")
y_train.to_excel("../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/y_train.xlsx")
y_test.to_excel("../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/y_test.xlsx")

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
# Model Building - Logistic Regression, Gaussian Naive Bayes, KNN, SVM
# =============================================================================
def train_dataset(X_resampled, 
                  y_resampled, 
                  X_test, 
                  y_test):
    # Step 1: Initialize models
    models = {
        "svm": SVC(),
        "knn": KNeighborsClassifier(),
        "xgb": XGBClassifier(),
        "logistic": LogisticRegression(),
        "random_forest": RandomForestClassifier()
    }

    # Step 2: Train models with default parameters and evaluate
    results = {}

    for model_name, model in models.items():
        print(f"\n\nTraining {model_name}...")
        if model_name == "knn":
            # Ensure input data is in the correct format
            X_resampled = X_resampled.to_numpy() if isinstance(X_resampled, pd.DataFrame) else X_resampled
            y_resampled = y_resampled.to_numpy().ravel() if isinstance(y_resampled, pd.DataFrame) else y_resampled
            X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
            y_test = y_test.to_numpy().ravel() if isinstance(y_test, pd.DataFrame) else y_test

        # Fit the model with default parameters
        model.fit(X_resampled, y_resampled)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Store results
        results[model_name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

        print(f"{model_name} training completed.")

    return results

# =============================================================================
# MODEL EVALUATION RESULTS
# =============================================================================
training_results = train_dataset(X_resampled,
                                 y_resampled,
                                 X_test,
                                 y_test)

# =============================================================================
# STORE MODEL RESULTS
# =============================================================================
save_path = "../../../6_Results/1_ML_Default_Parameters/Default_Of_Credit_Card_Client_Data"

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
