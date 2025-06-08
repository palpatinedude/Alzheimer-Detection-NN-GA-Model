import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
from NN import exercise011 as nn_exercise
from GA import exercise02 as ga_exercise


def main(file_path):
    print("Running Neural Network experiment...")
    nn_results = nn_exercise.main(file_path)

    print("\nRunning Genetic Algorithm feature selection experiment...")
    ga_results = ga_exercise.main()

    # Extract NN results
    nn_selected = nn_results['Selected Features']
    nn_val_metrics = nn_results['Validation Metrics']
    nn_test_metrics = nn_results['Test Metrics']

    # Extract GA results
    ga_selected = ga_results['Average Selected Features']
    ga_val_metrics = ga_results['Validation Metrics']
    ga_test_metrics = ga_results['Test Metrics']

    # Define metrics to compare
    metrics_to_compare = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

    # Build comparison dictionary
    data = {
        "Metric": ["Selected Features"] + [f"Validation {m}" for m in metrics_to_compare] + [f"Test {m}" for m in metrics_to_compare],
        "NN Model": [nn_selected] + [nn_val_metrics.get(m) for m in metrics_to_compare] + [nn_test_metrics.get(m) for m in metrics_to_compare],
        "GA Model": [ga_selected] + [ga_val_metrics.get(m) for m in metrics_to_compare] + [ga_test_metrics.get(m) for m in metrics_to_compare],
    }

    # Create DataFrame
    df_comparison = pd.DataFrame(data)

    # Format numeric columns (except Selected Features, which is int)
    df_comparison["NN Model"] = df_comparison["NN Model"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    df_comparison["GA Model"] = df_comparison["GA Model"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    # Print table
    print("\n--- Comparison Table ---")
    print(df_comparison.to_string(index=False))

    # Save the comparison table to CSV
    results_dir = "results_comparison"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "comparison_results.csv")
    df_comparison.to_csv(output_path, index=False)
    print(f"\nComparison results saved to: {output_path}")

if __name__ == "__main__":
    # Update the dataset path accordingly
    data_path = "alzheimers_disease_data.csv"
    main(data_path)



