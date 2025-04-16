'''
   Helper functions to select best configuration for ANN model and save results.
'''
import os
import numpy as np
import matplotlib.pyplot as plt

from visualization.plot import plot_convergence_for_hidden_units, plot_convergence_for_lr_momentum
from modeling.model import k_fold_evaluation
from config import HIDDEN_UNIT_RATIOS, RESULTS_DIR, LEARNING_RATES, MOMENTUM_VALUES, PATIENCE


# ------------------ Helpers ------------------ #

def get_norm_label(norm):
    return 'Normalization' if norm else 'Standardization'

def get_method_label(norm):
    return 'MinMax Scaled' if norm else 'Standardized'

def create_results_folder(output_dir, norm):
    folder_path = os.path.join(output_dir, get_norm_label(norm))
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def write_fold_metrics(f, h_units, fold_res):
    f.write(f"\nResults for H1 = {h_units}:\n")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        f.write(f"{metric}: {np.mean(fold_res[metric]):.4f}\n")
    f.write("=" * 80 + "\n")

def write_summary_table(f, summary_table, method_label):
    f.write("\nSummary Table:\nMethod\tH1\tCE Loss\tMSE\tAccuracy\n")
    for row in summary_table:
        if row['method'] == method_label:
            f.write(f"{row['hidden_units']}\t{row['ce_loss']:.4f}\t{row['mse']:.4f}\t{row['accuracy']:.4f}\n")

def write_final_stats(f, final):
    f.write("=" * 80 + "\n")
    f.write(f"Avg Time to Converge: {final['avg_training_time']:.2f} sec\n")
    f.write(f"Avg Epochs to Converge: {final['avg_epochs_to_converge']:.2f}\n")


# ------------------ Save Hidden Units Results ------------------ #

def save_results_hidden(output_dir, norm, summary_table, results, hidden_units_options):
    method_label = get_method_label(norm)
    folder_path = create_results_folder(output_dir, norm)
    results_file = os.path.join(folder_path, 'neural_network_results.txt')

    with open(results_file, 'w') as f:
        f.write(f"Neural Network Results ({method_label})\n{'='*80}\n\n")

        for h_units in hidden_units_options:
            fold_res, final = results[h_units]
            write_fold_metrics(f, h_units, fold_res)

            summary_table.append({
                'method': method_label,
                'hidden_units': h_units,
                'ce_loss': final['ce_loss'],
                'mse': final['mse'],
                'accuracy': final['accuracy']
            })

        plot_convergence_for_hidden_units(results, folder_path, hidden_units_options, norm, method_label)
        write_summary_table(f, summary_table, method_label)
        write_final_stats(f, final)

    print(f"\nResults saved to: {folder_path}")


# ------------------ Save Hyperparameter Tuning Results ------------------ #

def save_results_hyper(output_dir, norm, results):
    folder_path = create_results_folder(output_dir, norm)
    results_file = os.path.join(folder_path, 'hyperparameter_results.txt')

    plot_convergence_for_lr_momentum(results)

    with open(results_file, 'w') as f:
        f.write(f"Hyperparameter Tuning Results ({get_norm_label(norm)})\n{'='*80}\n\n")
        f.write(f"{'η':<8}{'m':<8}{'CE loss':<10}{'MSE':<10}{'Accuracy':<10}\n")

        for res in results:
            f.write(f"{res['learning_rate']:<8.4f}{res['momentum']:<8.2f}"
                    f"{res['ce_loss']:<10.4f}{res['mse']:<10.4f}{res['accuracy']:<10.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nResults saved to: {folder_path}")


# ------------------ Hidden Units Optimization ------------------ #

def select_best_config_hidden(scaled_data_options, y_train):
    best_score = 0
    best_config = None
    best_result = None
    results_all = {True: {}, False: {}}
    summary_table = []

    for norm, (X_scaled, label) in scaled_data_options.items():
        print(f"\nRunning configuration: {label}")
        input_dim = X_scaled.shape[1]
        hidden_units_options = [int(input_dim * ratio) for ratio in HIDDEN_UNIT_RATIOS]

        for h_units in hidden_units_options:
            fold_res, final = k_fold_evaluation(X_scaled, y_train, hidden_units=h_units, norm=norm)
            results_all[norm][h_units] = (fold_res, final)

            if final['accuracy'] > best_score:
                best_score = final['accuracy']
                best_config = (norm, h_units)
                best_result = (fold_res, final)

        save_results_hidden(RESULTS_DIR, norm, summary_table, results_all[norm], hidden_units_options)

    print(f"\nBest Configuration: {get_norm_label(best_config[0])} with H1 = {best_config[1]} (Accuracy = {best_score:.4f})")
    return best_config, best_result


# ------------------ Hyperparameter Optimization ------------------ #
def select_best_config_hyper(X_scaled, y_train, hidden_units, best_norm):
    best_score = 0
    best_config = None
    best_result = None
    results = []

    print(f"\nSearching best learning rate & momentum for H1 = {hidden_units}")

    for lr in LEARNING_RATES:
        for m in MOMENTUM_VALUES:
            print(f"Evaluating: Learning Rate = {lr}, Momentum = {m}")

            _, final = k_fold_evaluation(
                X_scaled, y_train,
                hidden_units=hidden_units,
                norm=best_norm,
                learning_rate=lr,
                momentum=m
            )

            result = {
                'learning_rate': lr,
                'momentum': m,
                'accuracy': final['accuracy'],
                'ce_loss': final['ce_loss'],
                'mse': final['mse'],
                'epoch_accuracy': final['epoch_accuracy'],
                'avg_training_time': final['avg_training_time'],
                'avg_epochs_to_converge': final['avg_epochs_to_converge']
            }

            results.append(result)

            if final['accuracy'] > best_score:
                best_score = final['accuracy']
                best_config = (lr, m)
                best_result = result

    save_results_hyper(RESULTS_DIR, best_norm, results)
    print(f"\nBest Hyperparameter Combination: Learning Rate = {best_config[0]}, Momentum = {best_config[1]} (Accuracy = {best_score:.4f})")
    return best_config[0], best_config[1]  # Return only the learning rate and momentum


''' worked
import os
import matplotlib.pyplot as plt
import numpy as np
from visualization.plot import plot_convergence_for_hidden_units,plot_convergence_for_lr_momentum
from modeling.model import k_fold_evaluation
from config import HIDDEN_UNIT_RATIOS, RESULTS_DIR, LEARNING_RATES, MOMENTUM_VALUES

# ------------------ Save Output Results ------------------ #
def save_results_hidden(output_dir, norm, summary_table, results, hidden_units_options):
    norm_type = 'Normalization' if norm else 'Standardization'
    folder_path = os.path.join(output_dir, norm_type)
    os.makedirs(folder_path, exist_ok=True)

    results_file = os.path.join(folder_path, 'neural_network_results.txt')
    method_label = "MinMax Scaled" if norm else "Standardized"

    with open(results_file, 'w') as f:
        f.write(f"Neural Network Results ({method_label})\n{'='*80}\n\n")
        plt.figure(figsize=(10, 6))  # For summary plot

        for h_units in hidden_units_options:
            fold_res, final = results[h_units]

            f.write(f"\nResults for H1 = {h_units}:\n")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                f.write(f"{metric}: {np.mean(fold_res[metric]):.4f}\n")
            f.write("=" * 80 + "\n")

            summary_table.append({
             'method': method_label,
             'hidden_units': h_units,
             'ce_loss': final['ce_loss'],
             'mse': final['mse'],
             'accuracy': final['accuracy']
            })

           
        # Call function to plot the convergence of accuracy for each hidden unit setup
        plot_convergence_for_hidden_units(results, folder_path, hidden_units_options, norm, method_label)
        # Write summary table
        f.write("\nSummary Table:\nMethod\tH1\tCE Loss\tMSE\tAccuracy\n")
        for row in summary_table:
          if row['method'] == method_label:  # only write rows for current method
           f.write(f"{row['hidden_units']}\t{row['ce_loss']:.4f}\t{row['mse']:.4f}\t{row['accuracy']:.4f}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Avg Time to Converge: {final['avg_training_time']:.2f} sec\n")
        f.write(f"Avg Epochs to Converge: {final['avg_epochs_to_converge']:.2f}\n")
    
    print(f"\n Results saved to: {folder_path}")

# ------------- Try Different Scenarios and Pick Best ------------- #
def select_best_config_hidden(scaled_data_options, y_train,patience):
    best_score, best_config, best_result = 0, None, None
    results_all = {True: {}, False: {}}
    summary_table = []

    for norm, (X_scaled, label) in scaled_data_options.items():
        print(f"\nRunning configuration: {label}")
        input_dim = X_scaled.shape[1]
        hidden_units_options = [int(input_dim * ratio) for ratio in HIDDEN_UNIT_RATIOS]

        for h_units in hidden_units_options:
            fold_res, final = k_fold_evaluation(X_scaled, y_train, hidden_units=h_units , patience = patience, norm=norm)
            results_all[norm][h_units] = (fold_res, final)

            if final['accuracy'] > best_score:
                best_score = final['accuracy']
                best_config = (norm, h_units)
                best_result = (fold_res, final)

        save_results_hidden(RESULTS_DIR, norm, summary_table, results_all[norm], hidden_units_options)

    print(f"\n Best Configuration: {'Normalization' if best_config[0] else 'Standardization'} with H1 = {best_config[1]} (Accuracy = {best_score:.4f})")
    return best_config, best_result


def select_best_config_hyper(X_scaled, y_train, hidden_units, patience,best_norm):
    best_score = 0
    best_config = None
    best_result = None
    results = []

    print(f"\n Searching best learning rate & momentum for H1 = {hidden_units}")

    for lr in LEARNING_RATES:
        for m in MOMENTUM_VALUES:
            print(f"Evaluating: Learning Rate = {lr}, Momentum = {m}")

            _, final = k_fold_evaluation(
                X_scaled, y_train,
                hidden_units=hidden_units,
                learning_rate=lr,
                momentum=m,
                patience=patience
            )

            result = {
                'learning_rate': lr,
                'momentum': m,
                'accuracy': final['accuracy'],
                'ce_loss': final['ce_loss'],
                'mse': final['mse'],
                'epoch_accuracy': final['epoch_accuracy'],
                'avg_training_time': final['avg_training_time'],
                'avg_epochs_to_converge': final['avg_epochs_to_converge']
            }

            results.append(result)

            if final['accuracy'] > best_score:
                best_score = final['accuracy']
                best_config = (lr, m)
                best_result = result
        save_results_hyper(RESULTS_DIR, best_norm, results) 
    print(f"\n Best Hyperparameter Combination: Learning Rate = {best_config[0]}, Momentum = {best_config[1]} (Accuracy = {best_score:.4f})")
    return best_config, best_result, results

def save_results_hyper(output_dir, norm, results):
    norm_type = 'Normalization' if norm else 'Standardization'
    folder_path = os.path.join(output_dir, norm_type)
    os.makedirs(folder_path, exist_ok=True)

    results_file = os.path.join(folder_path, 'hyperparameter_results.txt')
    plot_convergence_for_lr_momentum(results)

    with open(results_file, 'w') as f:
        f.write(f"Hyperparameter Tuning Results ({norm_type})\n{'='*80}\n\n")
        f.write(f"{'η':<8}{'m':<8}{'CE loss':<10}{'MSE':<10}{'Accuracy':<10}\n")
        for res in results:
            f.write(f"{res['learning_rate']:<8.4f}{res['momentum']:<8.2f}{res['ce_loss']:<10.4f}{res['mse']:<10.4f}{res['accuracy']:<10.4f}\n")
        f.write("\n" + "=" * 80 + "\n")

    print(f"\nResults saved to: {folder_path}")
'''    