import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from visualization.plot_base import save_and_show_plot


# Confusion matrix plotting 
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


# Histogram plotting for detect feature distributions
def plot_histograms(df, types):
    for col, typ in types.items():
        if typ in ['Numeric']:
            plt.figure(figsize=(6, 3))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            save_and_show_plot(show=True)
