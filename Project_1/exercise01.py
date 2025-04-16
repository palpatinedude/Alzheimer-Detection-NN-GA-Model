# ============================================================================================
# Project: Alzheimer’s Disease Diagnosis using ANN
# Description: This script performs preprocessing, model training, evaluation, and 
#    hyperparameter tuning using a simple feedforward artificial neural network (ANN).
#  Goal: Classify Alzheimer’s diagnosis based on patient data with performance comparison 
#    between Standardization and Normalization techniques.
#  Author: Marianthi Thodi
#  AM: 1084576
# ============================================================================================


#               MAIN FUNCTION
# ------------------------- Imports ------------------------- #
from config import RESULTS_DIR
from preprocessing.preprocessing import inspect_data, scale_data
from reporting.experiments import select_best_config_hidden, select_best_config_hyper
from modeling.training import k_fold_evaluation
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import print_kfold_results, evaluate_with_transformation

# ------------------------- Main Pipeline ------------------------- #
def main(file_path):

    output_dir = RESULTS_DIR
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("##### A1: Preprocessing and Data Inspection and Use Logistic Regression #####")
    X,y = inspect_data(file_path)

    # ----- Data Splitting keep the test data for final evaluation ----- #
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

     # --- Evaluate Logistic Regression with Standardization and Normalization --- #
    X_train_std, X_test_std = evaluate_with_transformation(X_train, X_test, y_train, method_name='standard')
    X_train_norm, X_test_norm = evaluate_with_transformation(X_train, X_test, y_train, method_name='normalization')

    print("\n##### A2: Create ANN and Tune Best Configuration Based On Transformation Method and Number of Hidden Neuron  #####")
    # --- Hidden layer and transformation method tuning --- #
    scaled_data_options = {
        False: (X_train_std, "Standardization"),
        True: (X_train_norm, "Normalization")
    }

    #  Select best hidden unit config and print results ffrom best config
    best_config, best_result = select_best_config_hidden(scaled_data_options, y_train)
    print_kfold_results('ann', best_result[0], best_result[1])

    #  Extract config info
    best_norm, best_h1 = best_config
    X_scaled = scaled_data_options[best_norm][0]
    
    print("\n##### A3: Find Best Learning Rate,Momentum based on the Best  Number of Hidden Neuron and Transfomartion Method   #####")
    # --- Hyperparameter tuning --- #
    # Return best learning rate, best momentum based on the best number of hidden neurons and transformation method
    best_lr, best_momentum, fold_results, final_metrics = select_best_config_hyper(X_scaled, y_train, best_h1, best_norm)
    print_kfold_results('ann', fold_results, final_metrics)

    
if __name__ == "__main__":
    main("alzheimers_disease_data.csv")







'''
    # ----- Standardization ----- #
    X_train_std, std_scaler = scale_data(X_train, method='standard')
    X_test_std = pd.DataFrame(std_scaler.transform(X_test), columns=X_test.columns)
    print("\n   Logistic Regression (Standardized) - Final Metrics:")
    fold_results, final_metrics = k_fold_evaluation(X_train_std, y_train, model_type='logistic',transformation='standardization')
    print_kfold_results('logistic',fold_results, final_metrics)

    # ----- Normalization ----- #
    X_train_norm, norm_scaler = scale_data(X_train, method='normalization')
    X_test_norm = pd.DataFrame(norm_scaler.transform(X_test), columns=X_test.columns)
    print("\n   Logistic Regression (Normalized) - Final Metrics:")
    fold_results, final_metrics = k_fold_evaluation(X_train_norm, y_train, model_type='logistic',transformation='normalization')
    print_kfold_results('logistic',fold_results, final_metrics)
'''
