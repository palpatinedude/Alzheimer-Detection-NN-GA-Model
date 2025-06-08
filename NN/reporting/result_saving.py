# this file saves results for hidden unit experiments and hyperparameter tuning, and writes performance summaries to text files

import os
import json
from ..visualization.training_plots import plot_convergence_for_hidden_units, plot_convergence_for_lr_momentum,plot_regularization_accuracy,plot_regularization_ce_loss
from ..reporting.report_writer import write_fold_metrics, write_summary_table, write_final_stats, write_hyperparameter_summary, write_regularization_summary
from ..helpers import create_results_folder
import numpy as np


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


def save_best_model_results(output_dir,model,best_h1,best_lr,best_momentum,best_reg,config_summary,X_val,y_val,X_test,y_test):

    # Save summary
    results_path = os.path.join(output_dir, "best_ann_model_summary.txt")
    with open(results_path, "w") as f:
        f.write(config_summary)
    print(f"\nSummary written to {results_path}")

    # Save hyperparameters
    best_params = {
        "hidden_units": best_h1,
        "learning_rate": best_lr,
        "momentum": best_momentum,
        "regularization_lambda": best_reg
    }
    params_path = os.path.join(output_dir, "best_ann_hyperparameters.json")
    with open(params_path, "w") as json_file:
        json.dump(best_params, json_file, indent=4)
    print(f"Hyperparameters saved to {params_path}")

    # Save full model (architecture + weights)
    model_path = os.path.join(output_dir, "best_ann_model.keras")
    model.save(model_path)
    print(f"Full model saved to {model_path}")

    # Save validation set for GA use
    val_data_path = os.path.join(output_dir, "val_data.npz")
    np.savez(val_data_path, X_val=X_val, y_val=y_val)
    print(f"Validation data saved to {val_data_path}")

    # Save test set for GA use
    test_data_path = os.path.join(output_dir, "test_data.npz")
    np.savez(test_data_path, X_test=X_test, y_test=y_test)
    print(f"Test data saved to {test_data_path}")

    weights_path = os.path.join(output_dir, "best_ann_model.weights.h5")
    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")




