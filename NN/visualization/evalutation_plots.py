# this file contains functions to visualize model evaluation and dataset distributions with simple plots 
# and compare ce loss and accuracy for different configurations

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from visualization.plot_base import save_and_show_plot
from visualization.plot_base import plot_line_chart
import numpy as np
import pandas as pd
import os
# this function plots a confusion matrix as a heatmap to visualize prediction performance
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# this function plots histograms for all numeric columns to check feature distributions
def plot_histograms(df, types):
    for col, typ in types.items():
        if typ in ['Numeric']:
            plt.figure(figsize=(6, 3))
            sns.histplot(df[col], kde=True, bins=30)  # histogram with KDE for smoother distribution line
            plt.title(f'distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('frequency')
            save_and_show_plot(show=True)


#  this function plots the comparison between accuracy and CE Loss based on the number of hidden neurons.
def plot_comparison_accuracy_ce_loss(results_all, hidden_units_options, output_dir):
    accuracies = []
    ce_losses = []

    # extract metrics
    for h_units in hidden_units_options:
        _, final_metrics = results_all[h_units]
        accuracies.append(final_metrics['accuracy'])
        ce_losses.append(final_metrics.get('ce_loss'))

    # generalized line chart plotter
    plot_line_chart(
        x=hidden_units_options,
        y_series=[accuracies, ce_losses],
        title='Comparison of Accuracy and CE Loss with Different Hidden Neurons',
        xlabel='Number of Hidden Neurons',
        ylabel='Score',
        legend_labels=['Accuracy', 'CE Loss'],
        output_path=os.path.join(output_dir, "A2","comparison_accuracy_celoss.png"),
        show=True,
        colors=['tab:blue', 'tab:red']
    )



# bar plot with accuracy and cross entropy loss
def plot_accuracy_and_ce_loss(df, output_dir):
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, df['accuracy'], width, label='Accuracy', color='skyblue')
    ax.bar(x + width / 2, df['ce_loss'], width, label='CE Loss', color='salmon')

    ax.set_xlabel('Hyperparameter Combinations')
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Accuracy and CE Loss')
    ax.set_xticks(x)
    ax.set_xticks(np.arange(len(df['label'])))
    ax.set_xticklabels(df['label'], rotation=45)
    ax.legend()
    ax.grid(axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"A3", "barplot_accuracy_ce_loss.png"))
    plt.show()


# bar plot for training time
def plot_avg_training_time(df, output_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df['label'], df['avg_training_time'], color='mediumpurple')

    ax.set_ylabel('Avg. Training Time (s)')
    ax.set_xlabel('Hyperparameter Combinations')
    ax.set_title('Average Training Time per Hyperparameter Combination')
    ax.set_xticklabels(df['label'], rotation=45)
    ax.grid(axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"A3", "barplot_avg_training_time.png"))
    plt.show()



def plot_hyper_comparison(results, output_dir):
    df = pd.DataFrame(results)
    df['label'] = df.apply(lambda row: f"LR={row['learning_rate']}, M={row['momentum']}", axis=1)

    plot_accuracy_and_ce_loss(df, output_dir)
    plot_avg_training_time(df, output_dir)
    #plot_accuracy_vs_epoch(df, output_dir)