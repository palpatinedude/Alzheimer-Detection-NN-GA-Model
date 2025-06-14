

# ============================================================================================
# Goal:
# This script performs the following:
# 1. Loads the Alzheimer's Disease dataset.
# 2. Loads the selected_features previously found by the GA algorithm.
# 3. Loads the hyperparameters used by the best ANN after GA.
# 4. Retrains the ANN using the entire training set with the selected_features.
# 5. Evaluates the trained ANN on the separate test set.
# 6. Saves all the results (selected_features, hyperparameters, training history, 
#    test metrics, GA configuration) into a text report.

# Author: Marianthi Thodi
# AM: 1084576
# ============================================================================================

import json
from sklearn.model_selection import train_test_split
from NN.preprocessing.preprocessing import inspect_data
from NN.modeling.cross_validation import logistic_scaling
from NN.config import RESULTS_DIR_NN
from GA.config import RESULTS_DIR_GA
from NN.modeling.architecture import create_model_wrapper
from NN.modeling.training import train_model
from NN.modeling.evaluation import evaluate_performance
from NN.visualization.evalutation_plots import plot_confusion_matrix
from NN.visualization.training_plots import plot_training_history
from save import save_results_to_csv
from plot import plot_comparison



def main(file_path='alzheimers_disease_data.csv'):

    print(" Retraining ANN with GA's selected_features.\n")

    # Inspect data
    X, y = inspect_data(file_path)

    # Filter by GA's selected_features
    with open(f"{RESULTS_DIR_GA}/selected_features.txt", "r") as f:
        selected_features = [line.strip() for line in f if line.strip()]
        if selected_features[0] == "Selected Features:":
            selected_features = selected_features[1:]

    print(f"Selected Features from GA:\n{selected_features}\n")

    X = X[selected_features]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )
    
    # Standardization 
    X_train_std, X_test_std, fold_results_std, final_metrics_std = logistic_scaling(X_train, X_test, y_train, False)

    # Loading hyperparameters
    with open(f"{RESULTS_DIR_NN}/best_ann_hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)
    
    hidden_units = hyperparameters['hidden_units']
    learning_rate = hyperparameters['learning_rate']
    momentum = hyperparameters['momentum']
    regularization_lambda = hyperparameters['regularization_lambda']

    # Create and train the final ANN model
    final_model = create_model_wrapper(
        'ann',
        X_train_std.shape[1],
        hidden_units,
        learning_rate,
        momentum,
        regularization_lambda,
        simple_metrics=None
    )

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_std, y_train,
        stratify=y_train,
        test_size=0.2,
        random_state=42
    )


    train_stats = train_model(
        model=final_model,
        X_train=X_train_final,
        y_train=y_train_final,
        X_val=X_val,
        y_val=y_val,
        model_type='ann'
    )


    # Extract training history
    train_acc_history = train_stats['accuracy_history']
    val_acc_history = train_stats['val_accuracy_history']
    train_loss_history = train_stats['history'].history['loss'] if train_stats['history'] else None
    val_loss_history = train_stats['history'].history.get('val_loss', None) if train_stats['history'] else None
    epochs_ran = train_stats['epochs_ran']

    final_train_acc = train_acc_history[-1] if train_acc_history else None
    final_val_acc = val_acc_history[-1] if val_acc_history else None
    final_train_loss = train_loss_history[-1] if train_loss_history else None
    final_val_loss = val_loss_history[-1] if val_loss_history else None

    # Plot training curves
    plot_training_history(
        train_acc=train_acc_history,
        val_acc=val_acc_history,
        train_loss=train_loss_history,
        val_loss=val_loss_history,
        epochs_ran=epochs_ran
    )


    # Evaluate the final model on validation set
    y_pred_val, evalresults_val, metrics_val = evaluate_performance(
        model=final_model,
        X_val=X_val,
        y_val=y_val,
        model_type='ann'
    )
    val_loss = evalresults_val[0] if evalresults_val else None
    val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, _ = metrics_val

    # Evaluate the final model on test set
    y_pred_test, evalresults_test, metrics_test = evaluate_performance(
        model=final_model,
        X_val=X_test_std,
        y_val=y_test,
        model_type='ann'
    )
    test_loss = evalresults_test[0] if evalresults_test else None
    test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, _ = metrics_test

    # Calculate Overfitting Ratio 
    if val_accuracy and test_accuracy:
        overfitting_ratio = ((val_accuracy - test_accuracy) / val_accuracy) 
    else:
        overfitting_ratio = None

    # Plot confusion matrix for test set
    plot_confusion_matrix(y_test, y_pred_test, title='Confusion Matrix - Final Test Evaluation')

    # Save to CSV the metrics for the retrained NN model in new GA Retrained Model column
    output_csv = "results_comparison/comparison_results.csv"
    save_results_to_csv(
        selected_features,
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc,
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc,
        overfitting_ratio,
        output_csv=output_csv
    )

    plot_comparison(output_csv, output_file='results_comparison/comparison_plot_NN_GA_Retrained.png')


if __name__ == "__main__":
    data_file = "alzheimers_disease_data.csv"
    main(data_file)

