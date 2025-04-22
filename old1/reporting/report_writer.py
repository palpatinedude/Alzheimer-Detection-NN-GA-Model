import numpy as np

# ------------------ Helpers ------------------ #

def write_fold_metrics(f, h_units, fold_res):
    """Write fold metrics to the file."""
    f.write(f"\nResults for H1 = {h_units}:\n")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        f.write(f"{metric}: {np.mean(fold_res[metric]):.4f}\n")
    f.write("=" * 80 + "\n")

def write_summary_table(f, summary_table, method_label):
    """Write the summary table to the file."""
    f.write("\nSummary Table:\nMethod\tH1\tCE Loss\tMSE\tAccuracy\n")
    for row in summary_table:
        if row['method'] == method_label:
            f.write(f"{row['hidden_units']}\t{row['ce_loss']:.4f}\t{row['mse']:.4f}\t{row['accuracy']:.4f}\n")

def write_final_stats(f, final):
    """Write the final statistics to the file."""
    f.write("=" * 80 + "\n")
    f.write(f"Avg Time to Converge: {final['avg_training_time']:.2f} sec\n")
    f.write(f"Avg Epochs to Converge: {final['avg_epochs_to_converge']:.2f}\n")
