import os
import pandas as pd

# this function saves the results of the retrained model to a CSV file.
def save_results_to_csv(selected_features, 
                        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc,
                        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc,
                        overfitting_ratio,
                        output_csv="results_comparison/comparison_results.csv"):
    
    retrained_results = {
        "Selected Features": len(selected_features),
        "Validation Loss": f"{val_loss:.4f}" ,
        "Validation Accuracy": f"{val_accuracy:.4f}",
        "Validation Precision": f"{val_precision:.4f}" ,
        "Validation Recall": f"{val_recall:.4f}" ,
        "Validation F1 Score": f"{val_f1:.4f}" ,
        "Validation ROC AUC": f"{val_roc_auc:.4f}" ,
        "Test Loss": f"{test_loss:.4f}" ,
        "Test Accuracy": f"{test_accuracy:.4f}" ,
        "Test Precision": f"{test_precision:.4f}" ,
        "Test Recall": f"{test_recall:.4f}" ,
        "Test F1 Score": f"{test_f1:.4f}",
        "Test ROC AUC": f"{test_roc_auc:.4f}",
        "Overfitting Ratio Accuracy": f"{overfitting_ratio:.4f}"
    }

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(output_csv, index_col=0)

    # Add new column for GA Retrained Model (
    df["GA Retrained Model"] = None

    for metric, value in retrained_results.items():
        if metric in df.index:
            df.at[metric, "GA Retrained Model"] = value
        else:
            print(f"Warning: Metric '{metric}' not found in CSV index.")

    # Save back updated CSV with new column
    df.to_csv(output_csv)
    print(f"Updated CSV saved with GA Retrained Model column at {output_csv}")
