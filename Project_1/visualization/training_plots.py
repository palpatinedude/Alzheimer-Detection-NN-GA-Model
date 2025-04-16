import os
import numpy as np
from config import RESULTS_DIR
from visualization.plot_base import plot_line_chart

def plot_convergence_for_lr_momentum(results):
    """
    Plots convergence of accuracy across different learning rate/momentum pairs.
    """
    x = list(range(len(results[0]['epoch_accuracy'])))
    y_series = [res['epoch_accuracy'] for res in results]
    labels = [f"LR={res['learning_rate']}, Momentum={res['momentum']}" for res in results]

    output_dir = os.path.join(RESULTS_DIR, "Standardization")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "convergence_lr_momentum.png")

    plot_line_chart(
        x=x,
        y_series=y_series,
        title="Convergence for Different Learning Rates and Momentums",
        xlabel="Epochs",
        ylabel="Mean Accuracy",
        legend_labels=labels,
        output_path=output_path,
        show=True
    )

def plot_convergence_for_hidden_units(results, output_dir, hidden_units_options, norm=False, method_label='Standardized'):
    """
    Plots convergence for different hidden unit configurations.
    """
    epochs_range = np.arange(1, 101)
    combined_y_series = []
    combined_labels = []

    for h_units in hidden_units_options:
        _, final = results[h_units]
        y = final['epoch_accuracy']
        combined_y_series.append(y)
        combined_labels.append(f"H1={h_units}")

    # Combined plot
    combined_path = os.path.join(output_dir, f"convergence_plot_{'minmax' if norm else 'standard'}.png")
    plot_line_chart(
        x=epochs_range,
        y_series=combined_y_series,
        title=f"Convergence of Accuracy over Epochs ({method_label})",
        xlabel="Epochs",
        ylabel="Accuracy",
        legend_labels=combined_labels,
        output_path=combined_path,
        show=True
    )

def plot_convergence_and_early_stopping(fold, history, patience, norm=False):
    """
    Plots training/validation accuracy and loss with early stopping indicator.
    """
    norm_type = 'Normalization' if norm else 'Standardization'
    output_dir = os.path.join(RESULTS_DIR, norm_type)
    os.makedirs(output_dir, exist_ok=True)

    epochs = list(range(len(history.history['accuracy'])))

    # Accuracy plot
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

    # Loss plot with early stopping line
    loss_path = os.path.join(output_dir, f'loss_plot_fold_{fold}_early_stopping.png')
    early_stop_line = [{
        'x': len(history.history['loss']) - patience,
        'color': 'r',
        'linestyle': '--',
        'label': 'Early Stopping Point'
    }]
    plot_line_chart(
        x=epochs,
        y_series=[history.history['loss'], history.history['val_loss']],
        title=f"Loss over Epochs - Fold {fold} (Early Stopping)",
        xlabel='Epochs',
        ylabel='Loss',
        legend_labels=['Training Loss', 'Validation Loss'],
        colors=['blue', 'orange'],
        vlines=early_stop_line,
        output_path=loss_path
    )
