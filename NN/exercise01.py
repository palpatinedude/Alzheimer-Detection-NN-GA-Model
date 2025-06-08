# ============================================================================================
# Project: Alzheimer’s Disease Diagnosis using ANN
# Description: This script performs preprocessing, model training, evaluation, and 
#    hyperparameter tuning using a simple feedforward  neural network .
#  Goal: Classify Alzheimer’s diagnosis based on patient data with performance comparison 
#    between Standardization and Normalization techniques,Number of Hidden Neurons,Learning Rate,Momentum and Regularization coefficents .
#  Author: Marianthi Thodi
#  AM: 1084576
# ============================================================================================



# ------------------------- Imports ------------------------- #
from config import RESULTS_DIR_NN
from preprocessing.preprocessing import inspect_data
from reporting.experiments import select_best_config_hidden, select_best_config_hyper,select_best_config_regularization
import os
import shutil
from sklearn.model_selection import train_test_split
from helpers import print_kfold_results, print_test_metrics
from modeling.cross_validation import logistic_scaling
from modeling.architecture import create_model_wrapper
from modeling.training import train_model
from modeling.evaluation import evaluate_performance
from visualization.evalutation_plots import plot_confusion_matrix
from reporting.result_saving import save_best_model_results



# ------------------------- Main Function ------------------------- #
def main(file_path):

    # Setup results directory
    output_dir = RESULTS_DIR_NN
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------- A1: Preprocessing and Logistic Regression Evaluation ------------------------- #
    print("##### A1: Preprocessing and Logistic Regression Evaluation #####")
    X, y = inspect_data(file_path)

    # Split data into training and test sets (keeping 20% for final evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


    # Evaluate Logistic Regression with Standardization and Normalization
    X_train_std, X_test_std,fold_results,final_metrics = logistic_scaling(X_train, X_test, y_train,False)
    print_kfold_results('logistic', fold_results, final_metrics)

    X_train_norm, X_test_norm,fold_results,final_metrics = logistic_scaling(X_train, X_test, y_train,True)
    print_kfold_results('logistic', fold_results, final_metrics)


      # ------------------------- A2: Number of Hidden Neurons and Transformation Method Tuning ------------------------- #
    print("\n##### A2: Tuning ANN Hidden Layer and Transformation Method  Standardization#####")

    # Select best configuration for hidden layer size
    best_h1, best_result,results_all = select_best_config_hidden(X_train_std, y_train)
    print_kfold_results('ann', best_result[0], best_result[1]) # from best config



    # ------------------------- A3: Hyperparameter Tuning (Learning Rate, Momentum) ------------------------- #
    print("\n##### A3: Hyperparameter Tuning (Learning Rate, Momentum) #####")
    best_lr, best_momentum, fold_results, final_metrics = select_best_config_hyper(X_train_std, y_train, best_h1)
    print_kfold_results('ann', fold_results, final_metrics)
    

    # ------------------------- A4: Regularization Coefficient Tuning ------------------------- #
    print("\n##### A4: Regularization Coefficient Tuning #####")
    best_reg, fold_results, final_metrics = select_best_config_regularization(X_train_std, y_train, best_h1, best_lr, best_momentum)
    print_kfold_results('ann', fold_results, final_metrics)
    print(best_reg)


   
    # ------------------------- A5: Final Evaluation on Test Set ------------------------- #
    print("\n##### A5: Final Evaluation on Test Set #####")
    

    # Create, train, and evaluate the final ANN model
    final_model = create_model_wrapper('ann', X_train_std.shape[1], best_h1, best_lr, best_momentum, best_reg,simple_metrics=None)
    train_stats = train_model(model=final_model, X_train=X_train_std, y_train=y_train, X_val=None, y_val=None, model_type='ann')
    test_metrics = evaluate_performance(model=final_model, X_val=X_test_std, y_val=y_test, model_type='ann')


    # Extract test set results and print metrics
    y_pred_test, eval_results_test, metrics_test = test_metrics
    accuracy, precision, recall, f1, roc_auc, confusion = metrics_test
    print_test_metrics(accuracy, precision, recall, f1, roc_auc)

    # Plot confusion matrix
    print("\n##### Confusion Matrix for Final Test Evaluation #####")
    plot_confusion_matrix(y_test, y_pred_test, title="Confusion Matrix - Final Test Evaluation")


    # Prepare output text
    config_summary = (
     f"Best ANN Model Configuration:\n"
     f"Hidden Units: {best_h1}\n"
     f"Learning Rate: {best_lr}\n"
     f"Momentum: {best_momentum}\n"
     f"Regularization (Lambda): {best_reg}\n\n"
     f"Test Set Metrics:\n"
     f"Accuracy: {accuracy:.4f}\n"
     f"Precision: {precision:.4f}\n"
     f"Recall: {recall:.4f}\n"
     f"F1 Score: {f1:.4f}\n"
     f"ROC AUC: {roc_auc:.4f}\n"
   )

   
    save_best_model_results(
      output_dir=RESULTS_DIR_NN,
      model=final_model,
      best_h1=best_h1,
      best_lr=best_lr,
      best_momentum=best_momentum,
      best_reg=best_reg,
      config_summary=config_summary,
      X_val=X_test_std,
      y_val=y_test
    )
   
if __name__ == "__main__":
    main("../alzheimers_disease_data.csv")





