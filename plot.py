import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# this function plots a comparison of model performance metrics from a CSV file
def plot_comparison(output_csv_file, output_file='comparison_plot.png'):
    df = pd.read_csv(output_csv_file)
    metrics = ['Validation Accuracy', 'Test Accuracy', 'Overfitting Ratio Accuracy']
    df_filtered = df.loc[df.Metric.isin(metrics)]

    # required models to check for
    base_models = ['NN Model', 'GA Model']
    optional_model = 'GA Retrained Model'

    # check which models exist in the CSV
    models = [m for m in base_models if m in df_filtered.columns]
    if optional_model in df_filtered.columns:
        models.append(optional_model)

    if len(models) < 2:
        raise ValueError("CSV must contain at least 'NN Model' and 'GA Model' columns.")

    # adjust bar width based on number of models
    bar_width = 0.8 / len(models)  
    x = range(len(metrics))

    colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple'] 

    plt.figure(figsize=(10, 6))

    for i, model in enumerate(models):
        vals = df_filtered[model].astype(float).tolist()
        plt.bar([p + i * bar_width for p in x], vals, color=colors[i], width=bar_width, label=model)

    plt.xlabel('Metric')
    plt.xticks([p + bar_width * (len(models) / 2) for p in x], metrics)
    plt.legend()
    plt.title('Model Performance and Overfitting')
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot successfully saved to {output_file}")
