# this file saves results for hidden unit experiments and hyperparameter tuning, and writes performance summaries to text files

import os
from visualization.training_plots import plot_convergence_for_hidden_units, plot_convergence_for_lr_momentum,plot_regularization_accuracy,plot_regularization_ce_loss
from reporting.report_writer import write_fold_metrics, write_summary_table, write_final_stats, write_hyperparameter_summary, write_regularization_summary
from helpers import create_results_folder



# this function saves metrics and convergence plots for different hidden unit sizes
def save_results_hidden(output_dir, summary_table, results, hidden_units_options):
    folder_path = create_results_folder(output_dir)
    results_file = os.path.join(folder_path, "A2",'best_hidden_neurons.txt')

    with open(results_file, 'w') as f:
        f.write(f"Neural Network Results \n{'='*80}\n\n")

        for h_units in hidden_units_options:
            fold_res, final = results[h_units]
            write_fold_metrics(f, h_units, fold_res)

            # append summary row for this configuration
            summary_table.append({
                'hidden_units': h_units,
                'ce_loss': final['ce_loss'],
                'mse': final['mse'],
                'accuracy': final['accuracy']
            })

        # generate and save plot
        plot_convergence_for_hidden_units(results, folder_path, hidden_units_options)

        # write summary and final stats
        write_summary_table(f, summary_table)
        write_final_stats(f, final)

    print(f"\nResults saved to: {results_file}")


# this function saves metrics and plots from learning rate and momentum tuning
def save_results_hyper(output_dir, results):
    folder_path = create_results_folder(output_dir)
    results_file = os.path.join(folder_path, "A3",'hyperparameter_result.txt')
    plot_convergence_for_lr_momentum(results,output_dir)

    with open(results_file, 'w') as f:
        f.write(f"Hyperparameter Tuning Results \n{'='*80}\n\n")
        write_hyperparameter_summary(f, results)
        f.write("\n" + "=" * 80 + "\n")

    print(f"\nResults saved to: {results_file}")


# this function saves metrics and plots for regularization results
def save_results_regularization(output_dir, results):
    folder_path = create_results_folder(output_dir)
    results_file = os.path.join(folder_path,"Α4" ,'regularization_results.txt')

    # plot convergence curves for regularization values
    plot_regularization_ce_loss(results,output_dir)
    plot_regularization_accuracy(results,output_dir)

    with open(results_file, 'w') as f:
        f.write(f"Regularization Tuning Results \n{'='*80}\n\n")
        write_regularization_summary(f, results)
        f.write("\n" + "=" * 80 + "\n")

    print(f"\nResults saved to: {output_dir}")