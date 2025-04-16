import matplotlib.pyplot as plt

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
    """
    Plots a line chart with multiple series and optional vertical lines.
    """
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
