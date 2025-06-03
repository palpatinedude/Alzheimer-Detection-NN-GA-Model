# this file contains functions for plotting convergence and early stopping during model training
#  and how accuracy and cross-entropy loss vary with different regularization values

import os
import numpy as np
from config import RESULTS_DIR_NN
from visualization.plot_base import plot_line_chart

# this function plots convergence of accuracy for different lr/momentum pairs
def plot_convergence_for_lr_momentum(results,output_dir):
    x = list(range(len(results[0]['epoch_accuracy'])))  # create x-axis values (epochs)
    y_series = [res['epoch_accuracy'] for res in results]  # gather accuracy histories for all results
    labels = [f"LR={res['learning_rate']}, Momentum={res['momentum']}" for res in results]  # create labels

    output_path = os.path.join(output_dir, "A3",  "convergence_lr_momentum.png")

    # plot the convergence graph
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

# this function plots convergence for different hidden unit configurations
def plot_convergence_for_hidden_units(results, output_dir, hidden_units_options):
    epochs_range = np.arange(1, 101)  # set range for epochs
    combined_y_series = []  # list to store accuracy for each configuration
    combined_labels = []  # list to store labels for hidden unit options

    for h_units in hidden_units_options:
        _, final = results[h_units]  # get the final result for each configuration
        y = final['epoch_accuracy']  # extract epoch accuracy
        combined_y_series.append(y)  # add accuracy to series
        combined_labels.append(f"H1={h_units}")  # create label for each configuration

    # plot combined convergence for all configurations
    combined_path = os.path.join(output_dir, "A2", "convergence_plot.png")
    plot_line_chart(
        x=epochs_range,
        y_series=combined_y_series,
        title=f"Convergence of Accuracy over Epochs ",
        xlabel="Epochs",
        ylabel="Accuracy",
        legend_labels=combined_labels,
        output_path=combined_path,
        show=True
    )

# this function plots convergence and early stopping for a given fold
def plot_convergence_and_early_stopping(fold, history, patience, hidden_units, learning_rate, momentum=None, regularization=None):
    if momentum is None and regularization is None:
        output_dir = os.path.join(RESULTS_DIR_NN, 'A2', f'hidden_units_{hidden_units}')  # directory based on hidden_units
    elif momentum is not None and regularization is None:
        output_dir = os.path.join(RESULTS_DIR_NN, 'A3', f'learning_rate_momentum_{learning_rate}_{momentum}')  # directory based on learning rate and momentum
    elif momentum is not None and regularization is not None:
        output_dir = os.path.join(RESULTS_DIR_NN, 'A4', f'regularization_{regularization}')  # directory based on all parameters
        
    # create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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
        title=f"Loss over Epochs - Fold {fold} (Early Stopping)",
        xlabel='Epochs',
        ylabel='Loss',
        legend_labels=['Training Loss', 'Validation Loss'],
        colors=['blue', 'orange'],
        vlines=early_stop_line,  # add early stopping vertical line
        output_path=loss_path
    )

# this function plots accuracy and cross-entropy loss against regularization coefficient
def plot_regularization_accuracy(results, output_dir):
    # extract regularization values (λ) and corresponding accuracies
    lambdas = [r['regularization'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    # define path to save the plot
    path = os.path.join(output_dir, "Α4", "accuracy_vs_regularization.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)  # ensure the directory exists

    # plot using the generic line chart function
    plot_line_chart(
        x=lambdas,
        y_series=[accuracies],
        title="Accuracy vs Regularization Coefficient",
        xlabel="Regularization Coefficient (λ)",
        ylabel="Accuracy",
        legend_labels=["Accuracy"],
        output_path=path,
        show=True,
        colors=["blue"]
    )

def plot_regularization_ce_loss(results, output_dir):
    # extract regularization values (λ) and corresponding CE losses
    lambdas = [r['regularization'] for r in results]
    ce_losses = [r['ce_loss'] for r in results]

    # define path to save the plot
    path = os.path.join(output_dir, "Α4","ce_loss_vs_regularization.png")

    # plot using the generic line chart function
    plot_line_chart(
        x=lambdas,
        y_series=[ce_losses],
        title="Cross-Entropy Loss vs Regularization Coefficient",
        xlabel="Regularization Coefficient (λ)",
        ylabel="Cross-Entropy Loss",
        legend_labels=["CE Loss"],
        output_path=path,
        show=True,
        colors=["orange"]
    )
