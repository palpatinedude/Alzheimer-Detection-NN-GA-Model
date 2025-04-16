import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from modeling.architecture import create_model
from modeling.evaluation import evaluate_model
from visualization.training_plots import plot_convergence_and_early_stopping
from visualization.evalutation_plots import plot_confusion_matrix
from config import EPOCHS, BATCH_SIZE, PATIENCE




'''
def k_fold_evaluation(X, y, hidden_units=64, norm=False, learning_rate=0.001, momentum=0.2):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrices']}
    fold_metrics = {'accuracy': [], 'ce_loss': [], 'mse': []}
    epoch_accuracies, all_y_true, all_y_pred = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n Fold {fold}/{5}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = create_model(X_train.shape[1], hidden_units, learning_rate, momentum)
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
        training_time = time.time() - start_time
        epochs_ran = len(history.history['val_loss'])

        print(f" Converged in {epochs_ran} epochs, Time taken: {training_time:.2f} seconds")

        fold_metrics.setdefault('training_time', []).append(training_time)
        fold_metrics.setdefault('epochs_to_converge', []).append(epochs_ran)

        accuracy, precision, recall, f1, roc_auc, confusion = evaluate_model_ann(model, X_val, y_val)
        for metric, value in zip(['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrices'],
                                 [accuracy, precision, recall, f1, roc_auc, confusion]):
            fold_results[metric].append(value)

        eval_results = model.evaluate(X_val, y_val, verbose=0)
        fold_metrics['accuracy'].append(eval_results[1])
        fold_metrics['mse'].append(eval_results[2])
        fold_metrics['ce_loss'].append(eval_results[3])
        epoch_accuracies.append(history.history['accuracy'])

        all_y_true.extend(y_val)
        all_y_pred.extend((model.predict(X_val) > 0.5).astype(int))

        plot_convergence_and_early_stopping(fold, history, PATIENCE, norm)

    max_length = min(len(acc) for acc in epoch_accuracies)
    epoch_accuracies = [acc[:max_length] for acc in epoch_accuracies]
    mean_epoch_accuracy = np.mean(epoch_accuracies, axis=0)

    plot_confusion_matrix(all_y_true, all_y_pred, title=f"Combined Confusion Matrix (H1={hidden_units})")

    final_metrics = {
        'accuracy': np.mean(fold_metrics['accuracy']),
        'ce_loss': np.mean(fold_metrics['ce_loss']),
        'mse': np.mean(fold_metrics['mse']),
        'epoch_accuracy': mean_epoch_accuracy,
        'avg_training_time': np.mean(fold_metrics['training_time']),
        'avg_epochs_to_converge': np.mean(fold_metrics['epochs_to_converge'])
    }

    return fold_results, final_metrics
'''


def k_fold_evaluation(X, y, model_type='ann', hidden_units=64, norm=False,
                      learning_rate=0.001, momentum=0.2,transformation='standardization'):
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrices']}
    fold_metrics = {'accuracy': [], 'ce_loss': [], 'mse': []}
    epoch_accuracies, all_y_true, all_y_pred = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n Fold {fold}/{5}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create model depending on type
        model = create_model(
            input_dim=X_train.shape[1] if model_type == 'ann' else None,
            hidden_units=hidden_units,
            learning_rate=learning_rate,
            momentum=momentum,
            model_type=model_type
        )

        if model_type == 'ann':
            early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_val, y_val),
                verbose=0,
                callbacks=[early_stopping]
            )
            training_time = time.time() - start_time
            epochs_ran = len(history.history['val_loss'])

            print(f" Converged in {epochs_ran} epochs, Time taken: {training_time:.2f} seconds")

            fold_metrics.setdefault('training_time', []).append(training_time)
            fold_metrics.setdefault('epochs_to_converge', []).append(epochs_ran)

            # Evaluate model
            accuracy, precision, recall, f1, roc_auc, confusion = evaluate_model(model, X_val, y_val, model_type)

            eval_results = model.evaluate(X_val, y_val, verbose=0)
            fold_metrics['accuracy'].append(eval_results[1])
            fold_metrics['mse'].append(eval_results[2])
            fold_metrics['ce_loss'].append(eval_results[3])
            epoch_accuracies.append(history.history['accuracy'])

            # Prediction collection
            all_y_true.extend(y_val)
            all_y_pred.extend((model.predict(X_val) > 0.5).astype(int))

            # Plot convergence
            plot_convergence_and_early_stopping(fold, history, PATIENCE, norm)

        elif model_type == 'logistic':
            model.fit(X_train, y_train)

            # Evaluate model using unified function
            accuracy, precision, recall, f1, roc_auc, confusion = evaluate_model(model, X_val, y_val, model_type)

            # Collect metrics
            fold_metrics['accuracy'].append(accuracy)

            # Prediction collection
            all_y_true.extend(y_val)
            all_y_pred.extend(model.predict(X_val))

        else:
            raise ValueError("Invalid model_type. Choose 'ann' or 'logistic'.")

        # Store per-fold results
        for metric, value in zip(
            ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrices'],
            [accuracy, precision, recall, f1, roc_auc, confusion]
        ):
            fold_results[metric].append(value)

    # Compute mean epoch-wise accuracy for ANN
    mean_epoch_accuracy = None
    if model_type == 'ann' and epoch_accuracies:
        max_length = min(len(acc) for acc in epoch_accuracies)
        epoch_accuracies = [acc[:max_length] for acc in epoch_accuracies]
        mean_epoch_accuracy = np.mean(epoch_accuracies, axis=0)

    # Plot combined confusion matrix
    transformation_type = "Normalization" if transformation == "normalization" else "Standardization"
    plot_confusion_matrix(
        all_y_true, 
        all_y_pred, 
        title=f"Combined Confusion Matrix ({model_type.upper()} - {transformation_type}{f', H1={hidden_units}' if model_type == 'ann' else ''})"
    )

    # Compile final metrics
    final_metrics = {
        'accuracy': np.mean(fold_metrics['accuracy']),
        'ce_loss': np.mean(fold_metrics['ce_loss']) if model_type == 'ann' else None,
        'mse': np.mean(fold_metrics['mse']) if model_type == 'ann' else None,
        'epoch_accuracy': mean_epoch_accuracy,
        'avg_training_time': np.mean(fold_metrics['training_time']) if model_type == 'ann' else None,
        'avg_epochs_to_converge': np.mean(fold_metrics['epochs_to_converge']) if model_type == 'ann' else None
    }

    return fold_results, final_metrics


def search_lr_momentum(X, y, hidden_units=64, learning_rates=None, momentum=None, norm=False):
    results = []
    for lr in learning_rates:
        for m in momentum:
            print(f"\nEvaluating: Learning Rate = {lr}, Momentum = {m}")
            fold_res, final = k_fold_evaluation(X, y,'ann',hidden_units, norm, learning_rate=lr, momentum=m)
            results.append({
                'learning_rate': lr,
                'momentum': m,
                'epoch_accuracy': final['epoch_accuracy'],
                'ce_loss': final['ce_loss'],
                'mse': final['mse'],
                'accuracy': final['accuracy'],
                'avg_training_time': final['avg_training_time'],
                'avg_epochs_to_converge': final['avg_epochs_to_converge']
            })
    return results