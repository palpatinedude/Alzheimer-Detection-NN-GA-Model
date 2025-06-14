# ============================================================================================
# Project: Alzheimer's Disease Diagnosis — Unified Execution for ANN & GA Feature Selection
# Description: This script executes and compares two machine learning approaches:
#              1. A baseline neural network (ANN) trained with all features.
#              2. A genetic algorithm (GA) that selects an optimal subset of features,
#                 then uses the same ANN model architecture to evaluate performance.
#
# Goal: Run both ANN and GA pipelines, evaluate them using common metrics on validation 
#       and test sets, and create a side-by-side comparison table for performance and
#       feature efficiency.
#
# Author: Marianthi Thodi
# AM: 1084576
# ============================================================================================

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from NN import exercise01 as nn_exercise
from GA import exercise02 as ga_exercise
from plot import plot_comparison




def main(file_path):
    print(" Running Neural Network experiment.")
    nn_results = nn_exercise.main(file_path)

    print("\n Running Genetic Algorithm feature selection experiment.")
    ga_results = ga_exercise.main()

    # Extract results from ANN
    nn_selected = nn_results['Selected Features']
    nn_val_metrics = nn_results['Validation Metrics']
    nn_test_metrics = nn_results['Test Metrics']

    # Extract results from GA
    ga_selected = ga_results['Selected Features'] 
    ga_val_metrics = ga_results['Validation Metrics']
    ga_test_metrics = ga_results['Test Metrics']

    # Define common evaluation metrics to compare
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

    # Calculate overfitting ratio for Accuracy
    nn_overfitting_ratio = (nn_val_metrics['Accuracy'] - nn_test_metrics['Accuracy']) / nn_val_metrics['Accuracy']
    ga_overfitting_ratio = (ga_val_metrics['Accuracy'] - ga_test_metrics['Accuracy']) / ga_val_metrics['Accuracy']

    # Construct the comparison table
    comparison_data = {
        "Metric": ["Selected Features"] + [f"Validation {m}" for m in metrics] + [f"Test {m}" for m in metrics] + ["Overfitting Ratio Accuracy"],
        "NN Model": [nn_selected] + [nn_val_metrics.get(m) for m in metrics] + [nn_test_metrics.get(m) for m in metrics] + [nn_overfitting_ratio],
        "GA Model": [ga_selected] + [ga_val_metrics.get(m) for m in metrics] + [ga_test_metrics.get(m) for m in metrics] + [ga_overfitting_ratio],
    }

    df_comparison = pd.DataFrame(comparison_data)

   
    for col in ["NN Model", "GA Model"]:
        df_comparison[col] = df_comparison[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    # Print the comparison table
    print("\n--- Comparison Table ---")
    print(df_comparison.to_string(index=False))

    # Save results to CSV 
    results_dir = "results_comparison"
    os.makedirs(results_dir, exist_ok=True)
    output_csv = os.path.join(results_dir, "comparison_results.csv")
    df_comparison.to_csv(output_csv, index=False)
    print(f"\nComparison results saved to: {output_csv}")

    # Plot comparison
    output_file = os.path.join(results_dir, "comparison_plot_NN_GA.png")
    plot_comparison(output_csv, output_file)

if __name__ == "__main__":
    data_path = "alzheimers_disease_data.csv"
    main(data_path)


