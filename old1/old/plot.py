'''
    Includes tools to plot confusion matrices (to compare predictions vs. actual results), feature distributions (to see what your data looks like), and training progress (like accuracy and loss over time). 
    It also compares different model setups, such as learning rates, momentum, and hidden layer sizes, and shows where early stopping happens to prevent overfitting. 
    All plots are saved to specific folders for later review.
'''

'''worked
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
from config import RESULTS_DIR



# -----------------  VISUALIZATION  -----------------

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots a confusion matrix from true and predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# Plot distributions of features
def plot_histograms(df, types):
    """
    Plots histograms to visualize feature distributions.
    """
    print("\nFeature Distributions:")
    for col, typ in types.items():
        if typ in ['Numeric', 'Discrete Integer']:
            plt.figure(figsize=(6, 3))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

# Create a plot for convergence per learning rate and momentum combination
def plot_convergence_for_lr_momentum(results):
    plt.figure(figsize=(10, 6))
    
    for result in results:
        plt.plot(result['epoch_accuracy'], label=f"LR={result['learning_rate']}, Momentum={result['momentum']}")
    
    plt.title("Convergence for Different Learning Rates and Momentums")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_dir = os.path.join(RESULTS_DIR, "Standardization")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "convergence_lr_momentum.png"))
    plt.show()

# Function to plot convergence for each hidden unit setup
def plot_convergence_for_hidden_units(results, output_dir, hidden_units_options, norm=False, method_label='Standardized'):
    epochs_range = np.arange(1, 101)
    plt.figure(figsize=(10, 6))

    # Loop through each hidden unit configuration and plot its convergence
    for h_units in hidden_units_options:
        fold_res, final = results[h_units]

        # Save individual convergence plot for each hidden unit setup
        plt_ind = plt.figure()
        plt.plot(epochs_range, final['epoch_accuracy'], label=f"H1={h_units}", color='tab:blue')
        plt.title(f"Convergence (H1={h_units}) - {method_label}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"convergence_H1_{h_units}_{'minmax' if norm else 'standard'}.png"))
        plt.close(plt_ind)

        plt.plot(epochs_range, final['epoch_accuracy'], label=f"H1={h_units}")

    # Combined plot for all hidden unit setups
    plt.title(f"Convergence of Accuracy over Epochs ({method_label})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"convergence_plot_{'minmax' if norm else 'standard'}.png"))
    plt.show()


# ------------------ Plotting Convergence and Early Stopping ------------------ #
def plot_convergence_and_early_stopping(fold, history, patience, norm=False):
    # Determine the directory based on normalization or standardization
    norm_type = 'Normalization' if norm else 'Standardization'
    output_dir = os.path.join(RESULTS_DIR, norm_type)
    os.makedirs(output_dir, exist_ok=True)

    # Plot accuracy over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
    plt.title(f"Accuracy over Epochs - Fold {fold}")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    accuracy_plot_path = os.path.join(output_dir, f'accuracy_plot_fold_{fold}.png')
    plt.savefig(accuracy_plot_path)  # Save each fold's plot
    plt.close()

    # Plot validation loss and training loss to see where early stopping kicks in
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.axvline(x=len(history.history['loss']) - patience, color='r', linestyle='--', label="Early Stopping Point")
    plt.title(f"Loss over Epochs - Fold {fold} (Early Stopping)")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, f'loss_plot_fold_{fold}_early_stopping.png')
    plt.savefig(loss_plot_path)  # Save each fold's plot
    plt.close()            
'''
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
from config import RESULTS_DIR


# -----------------  UTILITIES  -----------------

def save_and_show_plot(output_path=None, show=False):
    """
    Saves and/or shows the current plot, then closes it.
    """
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def plot_line_chart(
    x, y_series, title, xlabel, ylabel,
    legend_labels=None, vlines=None, output_path=None, show=False, colors=None
):
    plt.figure(figsize=(10, 6))

    for idx, y in enumerate(y_series):
        color = colors[idx] if colors and idx < len(colors) else None
        label = legend_labels[idx] if legend_labels else None

        # Adjust y to match the length of x
        if len(x) != len(y):
            if len(x) < len(y):
                y = y[:len(x)]  # Truncate y to match x
            else:
                x = x[:len(y)]  # Truncate x to match y

        plt.plot(x, y, label=label, color=color)

    if vlines:
        for line in vlines:
            plt.axvline(**line)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend_labels:
        plt.legend()
    plt.grid(True)
    save_and_show_plot(output_path, show)


# -----------------  VISUALIZATION  -----------------

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots a confusion matrix from true and predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()


def plot_histograms(df, types):
    """
    Plots histograms to visualize feature distributions.
    """
    print("\nFeature Distributions:")
    for col, typ in types.items():
        if typ in ['Numeric', 'Discrete Integer']:
            plt.figure(figsize=(6, 3))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            save_and_show_plot(show=True)


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
