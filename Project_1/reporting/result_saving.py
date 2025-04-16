import os
from visualization.training_plots import plot_convergence_for_hidden_units, plot_convergence_for_lr_momentum
from reporting.report_writer import write_fold_metrics, write_summary_table, write_final_stats
from helpers import get_norm_label, get_method_label


def create_results_folder(output_dir, norm):
    folder_path = os.path.join(output_dir, get_norm_label(norm))
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# ------------------ Save Hidden Units Results ------------------ #

def save_results_hidden(output_dir, norm, summary_table, results, hidden_units_options):
    """Save the results of different hidden units configurations."""
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
    """Save the results of hyperparameter tuning."""
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
