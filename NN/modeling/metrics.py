# this file provides functions to compute classification metrics and average epoch wise accuracy

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
import numpy as np

from ..visualization.evalutation_plots import plot_confusion_matrix
from ..helpers import predict_labels

#from helpers import predict_labels
#from visualization.evalutation_plots import plot_confusion_matrix


# this function calculates key classification metrics: accuracy, precision, recall, f1, and roc auc
def calculate_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred,zero_division=0),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        roc_auc_score(y_true, y_pred)
    )

# this function returns the confusion matrix for predicted vs actual labels
def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

# this function computes mean accuracy across multiple training runs (epochs aligned)
def compute_mean_epoch_accuracy(epoch_accuracies):
    if not epoch_accuracies:
        return None
    # find the shortest run to align epoch counts
    max_length = min(len(acc) for acc in epoch_accuracies)
    # trim all accuracy lists to the same length
    trimmed = [acc[:max_length] for acc in epoch_accuracies]
    # average accuracies across folds/runs
    return np.mean(trimmed, axis=0)

# this function evaluates a model and returns performance metrics and confusion matrix
def get_model_metrics(model, X_val, y_val, model_type='ann'):
    y_pred = predict_labels(model, X_val, model_type)

    metrics = calculate_metrics(y_val, y_pred)
    confusion = get_confusion_matrix(y_val, y_pred)
    return (*metrics, confusion)


# this processes all folds to compute final average metrics and plot the confusion matrix
def finalize_metrics(fold_results, fold_metrics, all_y_true, all_y_pred, epoch_accuracies, val_epoch_accuracies,model_type, hidden_units,learning_rate,momentum,regularization=None):
    if model_type == 'logistic':
        title = f"Confusion Matrix  logistic"
        print_title = f"Model = logistic :"
    else:  # model_type == 'ann'
        ann_parts = ["ann"]
        if hidden_units is not None:
            ann_parts.append(f"H1 = {hidden_units}")
        if learning_rate is not None:
            ann_parts.append(f"LR = {learning_rate}")
        if momentum is not None:
            ann_parts.append(f"Momentum = {momentum}")
        if regularization is not None:
            ann_parts.append(f"Reg = {regularization}")
        
        title = "Confusion Matrix " + ", ".join(ann_parts)
        print_title = ", ".join(ann_parts)

    # plot confusion matrix
    plot_confusion_matrix(all_y_true, all_y_pred, title=title)

    # calculate final averaged metrics
    mean_epoch_accuracy = compute_mean_epoch_accuracy(epoch_accuracies) if model_type == 'ann' else None
    mean_val_epoch_accuracy = compute_mean_epoch_accuracy(val_epoch_accuracies) if model_type == 'ann' else None
    final_metrics = {
        'accuracy': np.mean(fold_metrics['accuracy']),
        'ce_loss': np.mean(fold_metrics['ce_loss']) if model_type == 'ann' else None,
        'mse': np.mean(fold_metrics['mse']) if model_type == 'ann' else None,
        'epoch_accuracy': mean_epoch_accuracy,
        'val_epoch_accuracy': mean_val_epoch_accuracy,
        'avg_training_time': np.mean(fold_metrics['training_time']),
        'avg_epochs_to_converge': np.mean(fold_metrics['epochs_to_converge']) if model_type == 'ann' else None
    }

    return final_metrics

# this function to detect overfitting based on epoch accuracies
def score_epoch_accuracy(epoch_acc, val_epoch_acc):

    train_end = epoch_acc[-1]
    val_end = val_epoch_acc[-1]
    acc_gap = train_end - val_end

    # penalize based on how much overfitting occurred
    overfit_penalty = acc_gap ** 2  # strict penalty for overfitting

    # use score as a measure of performance in  overfitting
    score = val_end - overfit_penalty

    return max(0, min(1, score))  # clamp to [0,1]


# this function computes a composite score based on accuracy, overfitting, and epochs to converge
def composite_score(final_metrics, weights):
    overfit_gap_threshold = 0.1
    hard_penalty_cutoff = 0.3
    soft_penalty_factor = 0.2

    acc = final_metrics['accuracy']
    train_acc = final_metrics['epoch_accuracy'][-1]
    val_acc = final_metrics['val_epoch_accuracy'][-1]
    acc_gap = train_acc - val_acc

    # hard disqualify if overfitting is extreme
    if acc_gap > hard_penalty_cutoff:
        print(f"  Disqualified: Overfit gap = {acc_gap:.4f} (>{hard_penalty_cutoff})")
        return 0

    # normalize the epochs because it can be large cause of early stopping
    epochs_score = 1 / (final_metrics['avg_epochs_to_converge'] + 1)

    # calculate the overfitting score
    val_curve_score = score_epoch_accuracy(
        final_metrics['epoch_accuracy'], final_metrics['val_epoch_accuracy']
    )

    # apply soft penalty
    overfit_penalty = 0
    if acc_gap > overfit_gap_threshold:
        overfit_penalty = soft_penalty_factor * acc_gap

    score = (
        weights['accuracy'] * acc +
        weights['curve'] * val_curve_score +
        weights['epochs'] * epochs_score -
        overfit_penalty
    )

    print(f"Score breakdown:")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Curve Score     : {val_curve_score:.4f}")
    print(f"  Epochs Score    : {epochs_score:.4f}")
    print(f"  Overfit Penalty : {overfit_penalty:.4f} (Gap={acc_gap:.4f})")
    print(f"  → Total Score   : {score:.4f}")

    return max(0, min(1, score))
