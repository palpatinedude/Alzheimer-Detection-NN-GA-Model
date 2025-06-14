# ============================================================================================
# Project: Alzheimer’s Disease Diagnosis using Genetic Algorithm for Feature Selection
# Description: This script sets up and runs genetic algorithm (GA) experiments using
#    different combinations of population size, crossover probability, and mutation probability.
# Goal: Optimize feature subset selection (from 32 input features) to improve Alzheimer's 
#    prediction performance with a fixed ANN, by maximizing validation accuracy while reducing input dimensionality.
#    The script evaluates GA behavior across different parameter sets and prints summary results.
# Author: Marianthi Thodi
# AM: 1084576
# ============================================================================================

 
# ------------------------- Imports ------------------------- #
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt



from .config import (
    VAL_DATA_PATH, TEST_DATA_PATH, MODEL, DATA,
    BEST_PARAM, WEIGHTS, RESULTS_DIR_GA,
    MAX_GENERATIONS, ELITISM
)
from .experiment import Experiment
from .reporting import save_best_set_config, save_metrics_to_file
from NN.modeling.evaluation import evaluate_performance
from NN.visualization.evalutation_plots import plot_confusion_matrix



'''
from config import ( VAL_DATA_PATH, TEST_DATA_PATH, MODEL, DATA, BEST_PARAM, WEIGHTS, RESULTS_DIR_GA,
    MAX_GENERATIONS, ELITISM )
from experiment import Experiment
from reporting import save_best_set_config, save_metrics_to_file

from NN.modeling.evaluation import evaluate_performance
from NN.visualization import plot_confusion_matrix
from modeling.evaluation import evaluate_performance
from visualization.evalutation_plots import plot_confusion_matrix
'''
#  Plot validation accuracy, test accuracy, and overfitting ratio for both models.
def plot_comparison(output_csv_file, output_file='comparison_plot.png'):
    df = pd.read_csv(output_csv_file)
    metrics = ['Validation Accuracy', 'Test Accuracy', 'Overfitting Ratio Accuracy']
    df_filtered = df.loc[df.Metric.isin(metrics)]

    # Plotting side by side
    bar_width = 0.35
    x = range(len(metrics))

    nn_vals = df_filtered.loc[df_filtered.Metric.isin(metrics), "NN Model"].apply(float).tolist()
    ga_vals = df_filtered.loc[df_filtered.Metric.isin(metrics), "GA Model"].apply(float).tolist()

    plt.bar(x, nn_vals, color='skyblue', label='NN Model', width=bar_width)
    plt.bar([p + bar_width for p in x], ga_vals, color='lightgreen', label='GA Model', width=bar_width)

    plt.xlabel('Metric')
    plt.xticks([p + bar_width/2 for p in x], metrics)
    plt.legend()

    plt.title('Model Performance and Overfitting')

    plt.tight_layout()

    plt.savefig(output_file)
    print(f"Plot successfully saved to {output_file}")


# Example usage (after you have your GA best_summary)
# compare_ga_and_ann(best_summary, RESULTS_DIR_NN)


# ------------------------- Main Function ------------------------- #
def main():
    # ----------- Load Data ----------- #
    val_data = np.load(VAL_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)
    X_val, y_val = val_data["X_val"], val_data["y_val"]
    X_test, y_test = test_data["X_test"], test_data["y_test"]

    # ----------- Load Pretrained Model ----------- #
    if MODEL and os.path.exists(MODEL):
        model = load_model(MODEL)
    else:
        raise FileNotFoundError("Model path not found.")

    # ----------- Define GA Parameter Sets ----------- #
    param_sets = [
        {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.00},
        {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.01},
        {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
        {'pop_size': 20, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
        {'pop_size': 20, 'crossover_prob': 0.1, 'mutation_prob': 0.01},
        {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.00},
        {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.01},
        {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
        {'pop_size': 200, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
        {'pop_size': 200, 'crossover_prob': 0.1, 'mutation_prob': 0.01}
    ]

    # ----------- Run GA Experiment ----------- #
    experiment = Experiment(
        X_val=X_val,
        y_val=y_val,
        param_sets=param_sets,
        best_params_path=BEST_PARAM,
        weights_path=WEIGHTS,
        results_dir=RESULTS_DIR_GA,
        n_trials=10,
        max_generations=MAX_GENERATIONS,
        elitism=ELITISM,
    )
    results, best_set = experiment.run(model)
    save_best_set_config(best_set, RESULTS_DIR_GA)

    # ----------- Apply Best Feature Mask ----------- #
    mask = np.array(best_set['Best Individual Mask'], dtype=bool)
    X_val_masked = np.zeros_like(X_val)
    X_val_masked[:, mask] = X_val[:, mask]

    X_test_masked = np.zeros_like(X_test)
    X_test_masked[:, mask] = X_test[:, mask]

    # ----------- Evaluate on Validation Set ----------- #
    _, eval_results_val, metrics_val = evaluate_performance(model, X_val_masked, y_val, model_type='ann')
    val_loss = eval_results_val[0] if eval_results_val else None
    val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, _ = metrics_val

    # ----------- Evaluate on Test Set ----------- #
    y_pred_test, eval_results_test, metrics_test = evaluate_performance(model, X_test_masked, y_test, model_type='ann')
    test_loss = eval_results_test[0] if eval_results_test else None
    test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, _ = metrics_test

    # ----------- Plot Confusion Matrix for Test Set ----------- #
    plot_confusion_matrix(y_test, y_pred_test, title="Confusion Matrix - Final Test Evaluation")

    # ----------- Save Metrics and Selected Features ----------- #
    # Convert metrics_val and metrics_test to dictionaries if they are tuples
    metrics_val_dict = {
        'Loss': eval_results_val[0] if eval_results_val else None,
        'Accuracy': val_accuracy,
        'Precision': val_precision,
        'Recall': val_recall,
        'F1 Score': val_f1,
        'ROC AUC': val_roc_auc
    }

    metrics_test_dict = {
        'Loss': eval_results_test[0] if eval_results_test else None,
        'Accuracy': test_accuracy,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1 Score': test_f1,
        'ROC AUC': test_roc_auc
    }

    save_metrics_to_file(metrics_val_dict, metrics_test_dict, RESULTS_DIR_GA)


    df = pd.read_csv(DATA)
    df.drop(columns=['PatientID', 'DoctorInCharge'], inplace=True, errors='ignore')
    feature_names = df.columns[:-1]  # Exclude label column
    selected_feature_names = feature_names[mask]

    selected_features_path = os.path.join(RESULTS_DIR_GA, "selected_features.txt")
    with open(selected_features_path, "w") as f:
        f.write("Selected Features:\n")
        for name in selected_feature_names:
            f.write(name + "\n")


    best_summary = {
        'Selected Features': int(best_set['Average Selected Features']),
        'Validation Metrics': {
            'Loss': val_loss,
            'Accuracy': val_accuracy,
            'Precision': val_precision,
            'Recall': val_recall,
            'F1 Score': val_f1,
            'ROC AUC': val_roc_auc
        },
        'Test Metrics': {
            'Loss': test_loss,
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1 Score': test_f1,
            'ROC AUC': test_roc_auc
        }
    }

    return best_summary



if __name__ == "__main__":
    #main("../alzheimers_disease_data.csv")
    main()


'''
    # After building best_summary
    print("Best Summary:")
    print("Selected Features Count :", best_summary['Selected Features'])

    print("\nValidation Metrics:")
    for metric, value in best_summary['Validation Metrics'].items():
        print(f"{metric:<10}: {value}")

    print("\nTest Metrics:")
    for metric, value in best_summary['Test Metrics'].items():
        print(f"{metric:<10}: {value}")


    # from  NN.config import RESULTS_DIR_NN extract the best_ann_model_summary.txt the 
    # --------- Retrieve and print best ANN model summary (filtered)---------#
    best_ann_file = os.path.join("/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Alzheimer-Detection-NN-GA-Model/NN/Results", "best_ann_model_summary.txt")

    if os.path.exists(best_ann_file):
        print("\nBest ANN Model Summary:")
        with open(best_ann_file, "r") as f:
            lines = f.readlines()

        # Filter for Validation Set Metrics and Test Set Metrics
        validation_section = []
        test_section = []
        in_validation = False
        in_test = False

        for line in lines:
            line = line.strip()
            if line.startswith("Validation Set Metrics:"):
                in_validation = True
                in_test = False
                continue
            if line.startswith("Test Set Metrics:"):
                in_validation = False
                in_test = True
                continue

            if in_validation and line.startswith(("Loss:", "Accuracy:", "Precision:", "Recall:", "F1 Score:", "ROC AUC:")):
                validation_section.append(line)
            if in_test and line.startswith(("Loss:", "Accuracy:", "Precision:", "Recall:", "F1 Score:", "ROC AUC:")):
                test_section.append(line)

        print("\nValidation Set Metrics:")
        for item in validation_section:
            print(item)

        print("\nTest Set Metrics:")
        for item in test_section:
            print(item)
    else:
        print(f"Best ANN Model Summary not found at {best_ann_file}")

    compare_ga_and_ann(best_summary, "/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Alzheimer-Detection-NN-GA-Model/NN/Results")


'''