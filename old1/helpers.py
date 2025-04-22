import numpy as np
from preprocessing.preprocessing import scale_data
from modeling.training import k_fold_evaluation
import pandas as pd

# ------------------ Helpers ------------------ #

def get_norm_label(norm):
    return 'Normalization' if norm else 'Standardization'

def get_method_label(norm):
    return 'MinMax Scaled' if norm else 'Standardized'

def print_kfold_results(model_type,fold_results, final_metrics=None):
    print("\n" + "="*50)
    print(f"{model_type.upper()} Cross-Validation Results Summary")
    print("="*50)

    print("\nAverage Per-Fold Metrics:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        score = np.mean(fold_results[metric])
        print(f" {metric.capitalize():<10}: {score:.4f}")

    if model_type == 'ann' and final_metrics is not None:
        print("\nANN-Specific Metrics:")
        print(f" Cross-Entropy Loss : {final_metrics['ce_loss']:.4f}")
        print(f" MSE                : {final_metrics['mse']:.4f}")
        print(f" Avg Training Time  : {final_metrics['avg_training_time']:.2f} seconds")
        print(f" Avg Epochs         : {final_metrics['avg_epochs_to_converge']:.1f}")
    print("="*50 + "\n")

def evaluate_with_transformation(X_train, X_test, y_train, method_name):
    X_train_scaled, scaler = scale_data(X_train, method=method_name)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print(f"\n   Logistic Regression ({method_name.capitalize()}) - Final Metrics:")
    fold_results, final_metrics = k_fold_evaluation(
        X_train_scaled, y_train, model_type='logistic', transformation=method_name
    )
    print_kfold_results('logistic', fold_results, final_metrics)

    return X_train_scaled, X_test_scaled
