from visualization.plot_base import plot_line_chart
import os
from config import RESULTS_DIR_NN_BONUS

# this function plots convergence and early stopping for each fold
def plot_convergence_and_early_stopping(fold, history, patience):
    output_dir = os.path.join(RESULTS_DIR_NN_BONUS)  # set output directory
    os.makedirs(output_dir, exist_ok=True)  # create directory if it doesn't exist

    epochs = list(range(len(history.history['accuracy'])))  # create epoch range

    # plot accuracy for training and validation
    accuracy_path = os.path.join(output_dir, f'accuracy_plot_fold_{fold}.png')
    plot_line_chart(
        x=epochs,
        y_series=[history.history['accuracy'], history.history['val_accuracy']],
        title=f"Accuracy over Epochs - Fold {fold}",
        xlabel='Epochs',
        ylabel='Accuracy',
        legend_labels=['Training Accuracy', 'Validation Accuracy'],
        colors=['blue', 'green'],
        output_path=accuracy_path
    )

    # plot loss with early stopping indicator
    loss_path = os.path.join(output_dir, f'loss_plot_fold_{fold}_early_stopping.png')
    early_stop_line = [{
        'x': len(history.history['loss']) - patience,  # early stopping point
        'color': 'r',
        'linestyle': '--',
        'label': 'Early Stopping Point'
    }]
    plot_line_chart(
        x=epochs,
        y_series=[history.history['loss'], history.history['val_loss']],
        title=f"Loss over Epochs - Fold {fold} - Early Stopping)",
        xlabel='Epochs',
        ylabel='Loss',
        legend_labels=['Training Loss', 'Validation Loss'],
        colors=['blue', 'orange'],
        vlines=early_stop_line,  # add early stopping vertical line
        output_path=loss_path
    )
