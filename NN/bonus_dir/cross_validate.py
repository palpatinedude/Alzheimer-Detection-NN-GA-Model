from sklearn.model_selection import StratifiedKFold
from bonus_dir.model import create_model_bonus
from modeling.training  import train_model
from modeling.evaluation import evaluate_performance
from config import PATIENCE
from bonus_dir.visualize import plot_convergence_and_early_stopping


def cross_validate_model_bonus(X_train, y_train, hidden_units, learning_rate, momentum, regularization):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []  # List of dicts with final metrics per fold
    fold_metrics = {
        'accuracy': [],
        'mse': [],
        'ce_loss': [],
        'training_time': [],
        'epochs_to_converge': []
    }
    epoch_accuracies = []  # Track per-epoch accuracy history
    all_y_true = []
    all_y_pred = []
    val_epoch_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]

        print(f"\n--- Fold {fold_idx+1}/{5} ---")

        # Create and train model
        model = create_model_bonus(X_train_fold.shape[1], hidden_units, learning_rate, momentum, regularization)
        train_meta = train_model(model, X_train_fold, y_train_fold, X_val_fold, y_val_fold, model_type='ann')

        # Evaluate model
        y_pred_fold, eval_results, metrics = evaluate_performance(model, X_val_fold, y_val_fold, model_type='ann')

        # Store evaluation metrics
        fold_metrics['accuracy'].append(eval_results[1])     # index 1 = accuracy
        fold_metrics['mse'].append(eval_results[2])          # index 2 = MSE
        fold_metrics['ce_loss'].append(eval_results[3])      # index 3 = CE Loss
        fold_metrics['training_time'].append(train_meta['training_time'])
        fold_metrics['epochs_to_converge'].append(train_meta['epochs_ran'])
        epoch_accuracies.append(train_meta['accuracy_history'])
        val_epoch_accuracies.append(train_meta['val_accuracy_history'])      # validation accuracy

        # Save true & predicted values
        all_y_true.extend(y_val_fold)
        all_y_pred.extend(y_pred_fold)

        # Save all other metrics for this fold (from `get_model_metrics`)
        fold_results.append(metrics)

        plot_convergence_and_early_stopping(fold_idx, train_meta['history'], PATIENCE)

    return fold_results, fold_metrics, all_y_true, all_y_pred, epoch_accuracies,val_epoch_accuracies
