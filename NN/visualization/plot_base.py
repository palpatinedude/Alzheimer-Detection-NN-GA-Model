# this file provides functions to save and display plots, especially line charts for metrics or trends

import matplotlib.pyplot as plt
import os
from helpers import create_results_folder

# this function saves and/or shows the current plot, then safely closes it
def save_and_show_plot(output_path=None, show=False):
    plt.tight_layout()  # adjust layout to prevent clipping
    if output_path:
        #  directory exists before saving the plot
        output_dir = os.path.dirname(output_path)
        if output_dir:  # if there's a directory path, create it
            create_results_folder(output_dir)
        plt.savefig(output_path)  # save plot 
    if show:
        plt.show()  # display plot 
    plt.close()  # close plot free memory


# this function plots one or more lines with optional vertical markers and legend
def plot_line_chart(
    x, y_series, title, xlabel, ylabel,
    legend_labels=None, vlines=None, output_path=None, show=False, colors=None
):
    plt.figure(figsize=(10, 6))  # set figure size

    for idx, y in enumerate(y_series):
        color = colors[idx] if colors and idx < len(colors) else None
        label = legend_labels[idx] if legend_labels else None

        # adjust lengths if x and y don't match
        if len(x) != len(y):
            if len(x) < len(y):
                y = y[:len(x)]  # truncate y to match x
            else:
                x = x[:len(y)]  # truncate x to match y

        plt.plot(x, y, label=label, color=color)  # plot line

    if vlines:
        for line in vlines:
            plt.axvline(**line)  # add vertical lines if any

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend_labels:
        plt.legend()  # show legend if labels are provided
    plt.grid(True)
    save_and_show_plot(output_path, show)  # save or show final plot
