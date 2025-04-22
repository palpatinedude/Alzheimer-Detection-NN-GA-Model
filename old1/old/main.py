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
from config import RESULTS_DIR, EPOCHS, BATCH_SIZE, PATIENCE, LEARNING_RATES, MOMENTUM_VALUES
from preprocessing.preprocessing import (load_data, detect_outliers, determine_attribute_type, print_feature_ranges, evaluate_model_A1, scale_data)
from visualization.plot import ( plot_confusion_matrix, plot_histograms, plot_convergence_for_lr_momentum)
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import  select_best_config_hidden, select_best_config_hyper
from modeling.model import create_model, calculate_metrics, evaluate_model_ann, k_fold_evaluation

# ------------------------- Main Pipeline ------------------------- #
def main(file_path):
    print("##### A1: Preprocessing + Logistic Regression #####")
    df = load_data(file_path)
    df.drop(columns=['PatientID', 'DoctorInCharge'], inplace=True, errors='ignore')

    output_dir = RESULTS_DIR
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    X, y = df.drop(columns='Diagnosis'), df['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # ----- Standardization ----- #
    X_train_std, std_scaler = scale_data(X_train, method='standard')
    X_test_std = pd.DataFrame(std_scaler.transform(X_test), columns=X_test.columns)
    print("\n   Logistic Regression (Standardized)")
    _, y_true_std, y_pred_std = evaluate_model_A1(X_train_std, y_train)
    plot_confusion_matrix(y_true_std, y_pred_std, title="Standardized Data")

    # ----- Normalization ----- #
 #   X_train_norm, norm_scaler = scale_data(X_train, method='minmax')
  #  X_test_norm = pd.DataFrame(norm_scaler.transform(X_test), columns=X_test.columns)
   # print("\n Logistic Regression (Normalized)")
    #_, y_true_norm, y_pred_norm = evaluate_model_A1(X_train_norm, y_train)
    #plot_confusion_matrix(y_true_norm, y_pred_norm, title="Normalized Data")


     # --- Hidden layer tuning --- #
    scaled_data_options = {
        False: (X_train_std, "Standardization"),
        # True: (X_train_norm, "Normalization")
    }

    (best_norm, best_h1), _ = select_best_config_hidden(scaled_data_options, y_train)

    # --- Hyperparameter tuning --- #
    # return best learning rate, best momentum based on the best hidden layer
    (best_lr, best_momentum) = select_best_config_hyper(X_train_std, y_train, best_h1, best_norm)

    # best number of hidden units ,best transformation,best learning rate, best momentum use the test set 
    print(f"\n Best Hyperparameters: Hidden Units = {best_h1}, Transformation = {'Normalization' if best_norm else 'Standardization'}, Learning Rate = {best_lr}, Momentum = {best_momentum}")
    # Train the final model with the best hyperparameters and evaluate on the test set with k-fold 
    print("\n############# Final Model Evaluation on Test Set ###############")
    final_X_train = X_train_std if best_norm else X_train_std
    final_X_test = X_test_std if best_norm else X_test_std
    input_dim = final_X_train.shape[1]
    final_model = create_model(input_dim, best_h1, best_lr, best_momentum)
    final_model.fit(final_X_train, y_train, EPOCHS, BATCH_SIZE, verbose=0)
    y_test_pred_prob = final_model.predict(final_X_test)
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)
    accuracy, precision, recall, f1, roc_auc = calculate_metrics(y_test, y_test_pred)
    print(f" Test Set Evaluation ({'Normalization' if best_norm else 'Standardization'}, H1 = {best_h1}):")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")
    plot_confusion_matrix(y_test, y_test_pred, title="Final Model - Test Set Confusion Matrix")

    








# Run everything
if __name__ == "__main__":
    main("alzheimers_disease_data.csv")


'''
     # -------------- Final ANN on Test Set  -------------- #
    print("\n############# Final Model Evaluation on Test Set ###############")
    final_X_train = X_train_norm if best_norm else X_train_std
    final_X_test = X_test_norm if best_norm else X_test_std

    input_dim = final_X_train.shape[1]
    final_model = create_model(input_dim, best_h1)
    final_model.fit(final_X_train, y_train, EPOCHS, BATCH_SIZE, verbose=0)

    y_test_pred_prob = final_model.predict(final_X_test)
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)

    accuracy, precision, recall, f1, roc_auc = calculate_metrics(y_test, y_test_pred)
    print(f" Test Set Evaluation ({'Normalization' if best_norm else 'Standardization'}, H1 = {best_h1}):")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")

    plot_confusion_matrix(y_test, y_test_pred, title="Final Model - Test Set Confusion Matrix")
'''


