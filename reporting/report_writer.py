# this file writes evaluation results, summary tables, and final statistics to a file in a readable format

import numpy as np

# this function writes per-fold averaged metrics for a given number of hidden units
def write_fold_metrics(f, h_units, fold_res):
    f.write(f"\nResults for H1 = {h_units}:\n")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        f.write(f"{metric}: {np.mean(fold_res[metric]):.4f}\n")
    f.write("=" * 80 + "\n")

# this function writes a formatted summary table 
def write_summary_table(f, summary_table):
    f.write("\nSummary Table:\n\tH1\tCE Loss\tMSE\tAccuracy\n")
    for row in summary_table:
        f.write(f"{row['hidden_units']}\t{row['ce_loss']:.4f}\t{row['mse']:.4f}\t{row['accuracy']:.4f}\n")

# this function writes final averaged statistics like time and epochs to converge
def write_final_stats(f, final):
    f.write("=" * 80 + "\n")
    f.write(f"Avg Time to Converge: {final['avg_training_time']:.2f} sec\n")
    f.write(f"Avg Epochs to Converge: {final['avg_epochs_to_converge']:.2f}\n")

# this function writes summary rows for hyperparameter tuning
def write_hyperparameter_summary(f, results):
    f.write(f"{'η':<8}{'m':<8}{'CE loss':<10}{'MSE':<10}{'Accuracy':<10}\n")
    for res in results:
        f.write(f"{res['learning_rate']:<8.4f}{res['momentum']:<8.2f}"
                f"{res['ce_loss']:<10.4f}{res['mse']:<10.4f}{res['accuracy']:<10.4f}\n")


# this function writes summary rows for regularization tuning
def write_regularization_summary(f, results):
    f.write(f"{'Reg':<8}{'CE loss':<10}{'MSE':<10}{'Accuracy':<10}\n")
    for res in results:
        f.write(f"{res['regularization']:<8.4f}{res['ce_loss']:<10.4f}{res['mse']:<10.4f}{res['accuracy']:<10.4f}\n")
