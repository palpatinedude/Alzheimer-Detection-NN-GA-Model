# A1 and A2
from A1_Preprocessing import load_data, detect_outliers, determine_attribute_type, print_feature_ranges, plot_histograms, evaluate_model_A1, plot_confusion_matrix, scale_data
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.metrics import MeanSquaredError, BinaryCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os


# Define neural network model
def create_model(input_dim, hidden_units):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_units, activation='relu'))  # Only relu
    model.add(Dense(1, activation='sigmoid'))  # Binary output
    optimizer = SGD(learning_rate=0.001, momentum=0.2, decay=0.0, nesterov=False)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[
            'accuracy',
            MeanSquaredError(name='mse'),
            BinaryCrossentropy(name='ce_loss')
        ]
    )
    return model

# Metrics calculation
def calculate_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        roc_auc_score(y_true, y_pred)
    )

# Evaluate ANN on validation
def evaluate_model_ann(model, X_val, y_val):
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    accuracy, precision, recall, f1, roc_auc = calculate_metrics(y_val, y_pred)
    confusion = confusion_matrix(y_val, y_pred)
    return accuracy, precision, recall, f1, roc_auc, confusion

# K-fold evaluation
def k_fold_evaluation(X, y, n_splits=5, hidden_units=64, epochs=100, batch_size=32):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrices']}
    fold_metrics = {'accuracy': [], 'ce_loss': [], 'mse': []}
    epoch_accuracies = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nEvaluating Fold {fold}/{n_splits}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = create_model(X_train.shape[1], hidden_units)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val), verbose=0)

        accuracy, precision, recall, f1, roc_auc, confusion = evaluate_model_ann(model, X_val, y_val)

        fold_results['accuracy'].append(accuracy)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1'].append(f1)
        fold_results['roc_auc'].append(roc_auc)
        fold_results['confusion_matrices'].append(confusion)

        eval_results = model.evaluate(X_val, y_val, verbose=0)
        fold_metrics['accuracy'].append(eval_results[1])
        fold_metrics['mse'].append(eval_results[2])
        fold_metrics['ce_loss'].append(eval_results[3])
        epoch_accuracies.append(history.history['accuracy'])

        all_y_true.extend(y_val)
        all_y_pred.extend((model.predict(X_val) > 0.5).astype(int))

    mean_epoch_accuracy = np.mean(epoch_accuracies, axis=0)
    plot_confusion_matrix(all_y_true, all_y_pred, title=f"Combined Confusion Matrix (Hidden Units: {hidden_units})")

    final_metrics = {
        'accuracy': np.mean(fold_metrics['accuracy']),
        'ce_loss': np.mean(fold_metrics['ce_loss']),
        'mse': np.mean(fold_metrics['mse']),
        'epoch_accuracy': mean_epoch_accuracy
    }

    return fold_results, final_metrics

# Save results
def save_results(output_dir, norm, summary_table, results, hidden_units_options):
    norm_folder = 'Normalization' if norm else 'Standarization'
    folder_path = os.path.join(output_dir, norm_folder)
    os.makedirs(folder_path, exist_ok=True)
    results_file = os.path.join(folder_path, 'neural_network_results.txt')

    with open(results_file, 'w') as file:
        scaling_method = "MinMax Scaled Data" if norm else "Standardized Data"
        file.write(f"Results for Neural Network Evaluation ({scaling_method})\n{'='*80}\n\n")
        plt.figure(figsize=(10, 6))  # Combined plot

        for hidden_units in hidden_units_options:
            print(f"\nEvaluating with {hidden_units} hidden units (ReLU)")
            fold_results, final_metrics = results[hidden_units]

            file.write(f"\nResults for {hidden_units} hidden units:\n")
            for metric in fold_results:
                if metric != 'confusion_matrices':
                    avg_metric = np.mean(fold_results[metric])
                    file.write(f"{metric}: {avg_metric:.4f}\n")
            file.write("=" * 80 + "\n\n")

            summary_table.append({
                'hidden_units': hidden_units,
                'ce_loss': final_metrics['ce_loss'],
                'mse': final_metrics['mse'],
                'accuracy': final_metrics['accuracy']
            })

            # Individual convergence plot
            epochs_range = np.arange(1, 101)
            plt_individual = plt.figure(figsize=(8, 5))
            plt.plot(epochs_range, final_metrics['epoch_accuracy'], label=f'H1 = {hidden_units}', color='tab:blue')
            method_label = "MinMax Scaled" if norm else "Standardized"
            plt.title(f"Convergence of Accuracy over Epochs ({method_label}, H1 = {hidden_units})")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"convergence_H1_{hidden_units}_{'minmax' if norm else 'standard'}.png"))
            plt.close(plt_individual)

            plt.plot(epochs_range, final_metrics['epoch_accuracy'], label=f'H1 = {hidden_units}')

        # Summary Table
        file.write("Number of neurons in the hidden layer\tCE loss\tMSE\tAccuracy\n")
        for row in summary_table:
            file.write(f"H1 = {row['hidden_units']}\t{row['ce_loss']:.4f}\t{row['mse']:.4f}\t{row['accuracy']:.4f}\n")
        file.write("=" * 80 + "\n\n")

        # Combined convergence plot
        overall_title = f"Convergence of Accuracy over Epochs ({method_label})"
        plt.title(overall_title)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        combined_filename = f"convergence_plot_{'minmax' if norm else 'standard'}.png"
        plt.savefig(os.path.join(output_dir, combined_filename))
        plt.show()

    print(f"\nResults and plots saved to {folder_path}")

# Hidden units evaluation
def evaluate_with_different_hidden_units(X_train, y_train, output_dir, norm):
    input_dim = X_train.shape[1]
    hidden_units_options = [input_dim // 2, (2 * input_dim) // 3, input_dim, 2 * input_dim]
    results = {}
    summary_table = []

    for hidden_units in hidden_units_options:
        fold_results, final_metrics = k_fold_evaluation(
            X_train, y_train,
            n_splits=5,
            hidden_units=hidden_units,
            epochs=100,
            batch_size=32
        )
        results[hidden_units] = (fold_results, final_metrics)

    save_results(output_dir, norm, summary_table, results, hidden_units_options)

# Main function
def main(file_path):
    print("############# A1: Preprocessing and Logistic Regression ###############")
    df = load_data(file_path)
    df.drop(columns=['PatientID', 'DoctorInCharge'], inplace=True, errors='ignore')
    output_dir = '/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Results'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Standardization
    X_train_std, std_scaler = scale_data(X_train, method='standard')
    X_test_std = pd.DataFrame(std_scaler.transform(X_test), columns=X_test.columns)
    print("\n🔹 Evaluating Model with Standardized Data:")
    metrics_std, y_true_std, y_pred_std = evaluate_model_A1(X_train_std, y_train)
    plot_confusion_matrix(y_true_std, y_pred_std, title="Standardized Data (Validation)")
    print("############# A2: Neural Network with Standardization ###############")
    evaluate_with_different_hidden_units(X_train_std, y_train, output_dir, norm=False)

    # Normalization
    X_train_norm, norm_scaler = scale_data(X_train, method='minmax')
    X_test_norm = pd.DataFrame(norm_scaler.transform(X_test), columns=X_test.columns)
    print("\n🔹 Evaluating Model with Normalized Data:")
    metrics_norm, y_true_norm, y_pred_norm = evaluate_model_A1(X_train_norm, y_train)
    plot_confusion_matrix(y_true_norm, y_pred_norm, title="Normalized Data (Validation)")
    print("############# A2: Neural Network with Normalization ###############")
    evaluate_with_different_hidden_units(X_train_norm, y_train, output_dir, norm=True)


if __name__ == "__main__":
    main("alzheimers_disease_data.csv")



'''
# ---------- Normalization ----------
    X_train_norm, norm_scaler = scale_data(X_train, method='minmax')
    X_test_norm = pd.DataFrame(norm_scaler.transform(X_test), columns=X_test.columns)
    print("\n🔹 Evaluating Model with Normalized Data:")
    metrics_norm, y_true_norm, y_pred_norm = evaluate_model(X_train_norm, y_train)
    plot_confusion_matrix(y_true_norm, y_pred_norm, title="Normalized Data (Validation)")

    print(" ############# A2 : Neural Network  Building Normalization  ###############" )
    evaluate_with_different_hidden_units(X_train_norm, y_train, X_test_norm, y_test, output_dir, True)
'''    