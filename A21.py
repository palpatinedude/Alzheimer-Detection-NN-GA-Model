# ============================================================================================
# Project: Alzheimer’s Disease Diagnosis using ANN
# Description: This script performs preprocessing, model training, evaluation, and 
#    hyperparameter tuning using a simple feedforward artificial neural network (ANN).
#  Goal: Classify Alzheimer’s diagnosis based on patient data with performance comparison 
#    between Standardization and Normalization techniques.
#  Author: Marianthi Thodi
#  AM: 1084576
# ============================================================================================

# ------------------------- Imports ------------------------- #
from A1_Preprocessing import (
    load_data, detect_outliers, determine_attribute_type, print_feature_ranges,
    plot_histograms, evaluate_model_A1, plot_confusion_matrix, scale_data
)
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.metrics import MeanSquaredError, BinaryCrossentropy

# --------------------- Build ANN Model --------------------- #
# Create a simple feedforward neural network with one hidden layer
def create_model(input_dim, hidden_units):
    model = Sequential([
        Input(shape=(input_dim,)),  # Input layer
        Dense(hidden_units, activation='relu'),  # Hidden layer
        Dense(1, activation='sigmoid')  # Output layer (binary classification)
    ])
    optimizer = SGD(learning_rate=0.001, momentum=0.2)  # Basic SGD optimizer
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', MeanSquaredError(name='mse'), BinaryCrossentropy(name='ce_loss')]
    )
    return model

# --------------------- Evaluation Metrics --------------------- #
# Calculate classification metrics
def calculate_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        roc_auc_score(y_true, y_pred)
    )

# Predict and evaluate the ANN model
def evaluate_model_ann(model, X_val, y_val):
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    metrics = calculate_metrics(y_val, y_pred)
    return (*metrics, confusion_matrix(y_val, y_pred))

# ---------------------- K-Fold Training ---------------------- #
# Perform Stratified K-Fold cross-validation
def k_fold_evaluation(X, y, n_splits=5, hidden_units=64, epochs=100, batch_size=32):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrices']}
    fold_metrics = {'accuracy': [], 'ce_loss': [], 'mse': []}
    epoch_accuracies, all_y_true, all_y_pred = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n Fold {fold}/{n_splits}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = create_model(X_train.shape[1], hidden_units)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val), verbose=0)

        accuracy, precision, recall, f1, roc_auc, confusion = evaluate_model_ann(model, X_val, y_val)
        # Store all fold metrics, how good the model is in classification
        for metric, value in zip(['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrices'],
                                 [accuracy, precision, recall, f1, roc_auc, confusion]):
            fold_results[metric].append(value)

        # Evaluate final metrics and accuracy over epochs, how well model learned
        eval_results = model.evaluate(X_val, y_val, verbose=0)
        fold_metrics['accuracy'].append(eval_results[1])
        fold_metrics['mse'].append(eval_results[2])
        fold_metrics['ce_loss'].append(eval_results[3])
        epoch_accuracies.append(history.history['accuracy'])

        all_y_true.extend(y_val)
        all_y_pred.extend((model.predict(X_val) > 0.5).astype(int))

    # Average accuracy per epoch across folds
    mean_epoch_accuracy = np.mean(epoch_accuracies, axis=0)
    plot_confusion_matrix(all_y_true, all_y_pred, title=f"Combined Confusion Matrix (H1={hidden_units})")

    final_metrics = {
        'accuracy': np.mean(fold_metrics['accuracy']),
        'ce_loss': np.mean(fold_metrics['ce_loss']),
        'mse': np.mean(fold_metrics['mse']),
        'epoch_accuracy': mean_epoch_accuracy
    }

    return fold_results, final_metrics

# ------------------ Save Output Results ------------------ #
def save_results(output_dir, norm, summary_table, results, hidden_units_options):
    norm_type = 'Normalization' if norm else 'Standardization'
    folder_path = os.path.join(output_dir, norm_type)
    os.makedirs(folder_path, exist_ok=True)

    results_file = os.path.join(folder_path, 'neural_network_results.txt')
    method_label = "MinMax Scaled" if norm else "Standardized"

    with open(results_file, 'w') as f:
        f.write(f"Neural Network Results ({method_label})\n{'='*80}\n\n")
        plt.figure(figsize=(10, 6))  # For summary plot

        for h_units in hidden_units_options:
            print(f"\n Hidden Units: {h_units}")
            fold_res, final = results[h_units]

            f.write(f"\nResults for H1 = {h_units}:\n")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                f.write(f"{metric}: {np.mean(fold_res[metric]):.4f}\n")
            f.write("=" * 80 + "\n")

            summary_table.append({
                'hidden_units': h_units,
                'ce_loss': final['ce_loss'],
                'mse': final['mse'],
                'accuracy': final['accuracy']
            })

            # Plot convergence for each hidden unit setup
            epochs_range = np.arange(1, 101)
            plt_ind = plt.figure()
            plt.plot(epochs_range, final['epoch_accuracy'], label=f"H1={h_units}", color='tab:blue')
            plt.title(f"Convergence (H1={h_units}) - {method_label}")
            plt.xlabel("Epochs"); plt.ylabel("Accuracy")
            plt.grid(); plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"convergence_H1_{h_units}_{'minmax' if norm else 'standard'}.png"))
            plt.close(plt_ind)

            plt.plot(epochs_range, final['epoch_accuracy'], label=f"H1={h_units}")

        # Combined plot for all H1
        plt.title(f"Convergence of Accuracy over Epochs ({method_label})")
        plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"convergence_plot_{'minmax' if norm else 'standard'}.png"))
        plt.show()

        # Write summary table
        f.write("\nSummary Table:\nH1\tCE Loss\tMSE\tAccuracy\n")
        for row in summary_table:
            f.write(f"{row['hidden_units']}\t{row['ce_loss']:.4f}\t{row['mse']:.4f}\t{row['accuracy']:.4f}\n")
        f.write("=" * 80 + "\n")

    print(f"\n Results saved to: {folder_path}")

# ------------- Try Different Scenarios and Pick Best ------------- #
def select_best_config(scaled_data_options, y_train):
    best_score, best_config, best_result = 0, None, None
    results_all = {True: {}, False: {}}
    summary_table = []

    for norm, (X_scaled, label) in scaled_data_options.items():
        print(f"\nRunning configuration: {label}")
        input_dim = X_scaled.shape[1]
        hidden_units_options = [input_dim // 2, (2 * input_dim) // 3, input_dim, 2 * input_dim]

        for h_units in hidden_units_options:
            fold_res, final = k_fold_evaluation(X_scaled, y_train, hidden_units=h_units)
            results_all[norm][h_units] = (fold_res, final)

            if final['accuracy'] > best_score:
                best_score = final['accuracy']
                best_config = (norm, h_units)
                best_result = (fold_res, final)

        save_results("/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Results", norm, summary_table, results_all[norm], hidden_units_options)

    print(f"\n Best Configuration: {'Normalization' if best_config[0] else 'Standardization'} with H1 = {best_config[1]} (Accuracy = {best_score:.4f})")
    return best_config, best_result

# ------------------------- Main Pipeline ------------------------- #
def main(file_path):
    print("##### A1: Preprocessing + Logistic Regression #####")
    df = load_data(file_path)
    df.drop(columns=['PatientID', 'DoctorInCharge'], inplace=True, errors='ignore')

    output_dir = '/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Results'
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    X, y = df.drop(columns='Diagnosis'), df['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # ----- Standardization ----- #
    X_train_std, std_scaler = scale_data(X_train, method='standard')
    X_test_std = pd.DataFrame(std_scaler.transform(X_test), columns=X_test.columns)
    print("\n   Logistic Regression (Standardized)")
    _, y_true_std, y_pred_std = evaluate_model_A1(X_train_std, y_train)
    plot_confusion_matrix(y_true_std, y_pred_std, title="Standardized Data")

    # ----- Normalization ----- #
    X_train_norm, norm_scaler = scale_data(X_train, method='minmax')
    X_test_norm = pd.DataFrame(norm_scaler.transform(X_test), columns=X_test.columns)
    print("\n Logistic Regression (Normalized)")
    _, y_true_norm, y_pred_norm = evaluate_model_A1(X_train_norm, y_train)
    plot_confusion_matrix(y_true_norm, y_pred_norm, title="Normalized Data")

    # --- Find Best Hyperparameters --- #
    scaled_data_options = {
        False: (X_train_std, "Standardization"),
        True: (X_train_norm, "Normalization")
    }
    (best_norm, best_h1), _ = select_best_config(scaled_data_options, y_train)

    # -------------- Final ANN on Test Set -------------- #
    print("\n############# Final Model Evaluation on Test Set ###############")
    final_X_train = X_train_norm if best_norm else X_train_std
    final_X_test = X_test_norm if best_norm else X_test_std

    input_dim = final_X_train.shape[1]
    final_model = create_model(input_dim, best_h1)
    final_model.fit(final_X_train, y_train, epochs=100, batch_size=32, verbose=0)

    y_test_pred_prob = final_model.predict(final_X_test)
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)

    accuracy, precision, recall, f1, roc_auc = calculate_metrics(y_test, y_test_pred)
    print(f" Test Set Evaluation ({'Normalization' if best_norm else 'Standardization'}, H1 = {best_h1}):")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")

    plot_confusion_matrix(y_test, y_test_pred, title="Final Model - Test Set Confusion Matrix")

# Run everything
if __name__ == "__main__":
    main("alzheimers_disease_data.csv")
