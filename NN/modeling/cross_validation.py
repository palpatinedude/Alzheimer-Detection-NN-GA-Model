# this script performs 5-fold cross-validation for ann and logistic models
# includes trainin.g, evaluation, metric logging, and optional scaling

from sklearn.model_selection import StratifiedKFold
from ..modeling.metrics import finalize_metrics
from ..helpers import is_ann
from ..visualization.training_plots import plot_convergence_and_early_stopping
from ..config import PATIENCE
from ..modeling.architecture import create_model_wrapper
from ..modeling.training import train_model
from ..modeling.evaluation import evaluate_performance
from ..preprocessing.preprocessing import scale_data
import pandas as pd


# this runs a single round of cross-validation, collecting metrics for each fold
def cross_validate_fold(skf, X, y, model_type, hidden_units, learning_rate, momentum,regularization,simple_metrics=None):
    # initialize structures for storing fold-wise metrics
    fold_results = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrices']}
    fold_metrics = {'accuracy': [], 'ce_loss': [], 'mse': []}
    epoch_accuracies, all_y_true, all_y_pred = [], [], []
    val_epoch_accuracies = []

    # iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n Fold {fold}/5")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # create the model for this fold
        model = create_model_wrapper(model_type, X_train.shape[1], hidden_units, learning_rate, momentum,regularization,simple_metrics=simple_metrics)

        # train and evaluate the model
        if is_ann(model_type):
            # train ann model with validation
            results = train_model(model, X_train, y_train, X_val, y_val, model_type='ann')
            all_y_pred_fold, eval_results, metrics = evaluate_performance(model, X_val, y_val, model_type)

            # store evaluation results
            fold_metrics['accuracy'].append(eval_results[1])
            fold_metrics['mse'].append(eval_results[2])
            fold_metrics['ce_loss'].append(eval_results[3])
            fold_metrics.setdefault('training_time', []).append(results['training_time'])
            fold_metrics.setdefault('epochs_to_converge', []).append(results['epochs_ran'])
            epoch_accuracies.append(results['accuracy_history']) # training accuracy
            val_epoch_accuracies.append(results['val_accuracy_history'])      # validation accuracy

            # plot training curve for this fold
            plot_convergence_and_early_stopping(fold, results['history'], PATIENCE,hidden_units, learning_rate, momentum,regularization)
        else:
            # train logistic regression model without validation
            results = train_model(model, X_train, y_train, model_type='logistic')
            all_y_pred_fold, eval_results, metrics = evaluate_performance(model, X_val, y_val, model_type)
            fold_metrics['accuracy'].append(metrics[0])
            fold_metrics.setdefault('training_time', []).append(results['training_time'])

        # save true and predicted labels
        all_y_true.extend(y_val)
        all_y_pred.extend(all_y_pred_fold)

        # store all computed metrics for this fold
        for metric, value in zip(fold_results.keys(), metrics):
            fold_results[metric].append(value)

    return fold_results, fold_metrics, all_y_true, all_y_pred, epoch_accuracies,val_epoch_accuracies


# this runs 5-fold cross-validation and returns detailed and summarized results
def cross_validate_model(X, y, model_type='ann', hidden_units=64, learning_rate=0.001, momentum=None,regularization=None):
    # set up stratified k-fold to preserve class distribution
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # perform cross-validation and collect results
    fold_results, fold_metrics, all_y_true, all_y_pred, epoch_accuracies,val_epoch_accuracies = cross_validate_fold(
        skf, X, y, model_type, hidden_units, learning_rate, momentum, regularization
    )

    # compute final averaged metrics
    final_metrics = finalize_metrics(
        fold_results, fold_metrics, all_y_true, all_y_pred, epoch_accuracies, val_epoch_accuracies,model_type, hidden_units, learning_rate,momentum,regularization)

    return fold_results, final_metrics

# this handles data scaling and evaluation pipeline for logistic regression
def logistic_scaling(X_train, X_test ,y_train,norm):
    # scale training and test data using standard scaler
    X_train_scaled, scaler = scale_data(X_train,norm)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # print section header 
    print(f"\n   Logistic Regression  - Final Metrics:")

    # run cross-validation on scaled data
    fold_results, final_metrics = cross_validate_model(
        X_train_scaled, y_train, model_type='logistic')

    return X_train_scaled, X_test_scaled, fold_results, final_metrics
