# this file includes utility functions for labeling, printing results and creating output folders

import numpy as np
import os


# this function predicts labels based on model type
def predict_labels(model, X, model_type):
    return (model.predict(X) > 0.45).astype(int) if is_ann(model_type) else model.predict(X)

# this prints average k-fold metrics and extra info depending on model type
def print_kfold_results(model_type, fold_results, final_metrics=None):
    print("\n" + "="*50)
    print(f"{model_type.upper()} Cross-Validation Results Summary")
    print("="*50)

    print("\nAverage Per-Fold Metrics:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        score = np.mean(fold_results[metric])
        print(f" {metric.capitalize():<10}: {score:.4f}")
    
    # for logistic regression, print avg training time if available
    print(f" Avg Training Time  : {final_metrics['avg_training_time']:.2f} seconds")

    # for ann models, print additional loss metrics and epochs
    if model_type == 'ann' and final_metrics is not None:
        print("\nANN-Specific Metrics:")
        print(f" Cross-Entropy Loss : {final_metrics['ce_loss']:.4f}")
        print(f" MSE                : {final_metrics['mse']:.4f}")
        print(f" Avg Training Time  : {final_metrics['avg_training_time']:.2f} seconds")
        print(f" Avg Epochs         : {final_metrics['avg_epochs_to_converge']:.1f}")
    print("="*50 + "\n")


# this creates a folder to save model results based on normalization type
def create_results_folder(output_dir):
    folder_path = os.path.join(output_dir)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# this function checks if the model type is ANN
def is_ann(model_type):
    return model_type == 'ann'

# this function prints the final test metrics
def print_test_metrics(accuracy, precision, recall, f1, roc_auc):
    """Prints test set metrics."""
    print("\nTest Set Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
