'''
  Defines how your Artificial Neural Network (ANN) is built — architecture, layers, activation functions, etc.
  Contains helper functions for training the model like setting up cross validation, early stopping, and recording metrics.
  Handles how you measure model performance — computing accuracy, precision, recall, F1 score, confusion matrix, etc.

'''


from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.metrics import MeanSquaredError, BinaryCrossentropy
from keras.callbacks import EarlyStopping
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix)
from sklearn.model_selection import StratifiedKFold
import time
import numpy as np
from visualization.plot import plot_convergence_and_early_stopping, plot_confusion_matrix
from config import EPOCHS, BATCH_SIZE, PATIENCE, LEARNING_RATES, MOMENTUM_VALUES


# --------------------- Build ANN Model --------------------- #
# Create a simple feedforward neural network with one hidden layer
def create_model(input_dim, hidden_units, learning_rate=0.001, momentum=0.2):
    model = Sequential([
        Input(shape=(input_dim,)),  # Input layer
        Dense(hidden_units, activation='relu'),  # Hidden layer
        Dense(1, activation='sigmoid')  # Output layer (binary classification)
    ])
    
    # Use SGD with specified learning rate and momentum
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)  
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
def k_fold_evaluation(X, y, hidden_units=64,norm=False, learning_rate=0.001, momentum=0.2,):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrices']}
    fold_metrics = {'accuracy': [], 'ce_loss': [], 'mse': []}
    epoch_accuracies, all_y_true, all_y_pred = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n Fold {fold}/{5}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = create_model(X_train.shape[1], hidden_units, learning_rate=learning_rate, momentum=momentum)

         # Early stopping callback is to prevent overfitting with patience of 10 epochs 
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
        end_time = time.time()
        training_time = end_time - start_time
        epochs_ran = len(history.history['val_loss'])

        print(f" Converged in {epochs_ran} epochs, Time taken: {training_time:.2f} seconds")

        # Save convergence speed details
        fold_metrics.setdefault('training_time', []).append(training_time)
        fold_metrics.setdefault('epochs_to_converge', []).append(epochs_ran)

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
        fold_metrics['training_time'].append(training_time)
        fold_metrics['epochs_to_converge'].append(epochs_ran)
        epoch_accuracies.append(history.history['accuracy'])

        all_y_true.extend(y_val)
        all_y_pred.extend((model.predict(X_val) > 0.5).astype(int))

        # Plot Convergence and Early Stopping
        plot_convergence_and_early_stopping(fold, history,PATIENCE,norm)

    # Average accuracy per epoch across folds
    # Ensure all sequences in epoch_accuracies are of the same length
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


# Search grid for optimized learning rate and momentum
def search_lr_momentum(X, y, hidden_units=64,learning_rates=LEARNING_RATES, momentum=MOMENTUM_VALUES):    
    results = []
    
    # Perform K-fold evaluation for each combination of learning rate and momentum
    for lr in LEARNING_RATES:
        for m in MOMENTUM_VALUES:
            print(f"\nEvaluating: Learning Rate = {lr}, Momentum = {m}")
            
            fold_res, final = k_fold_evaluation(X, y, hidden_units=hidden_units,norm=norm,learning_rate=lr, momentum=m )
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
    
    # Return all results for analysis
    return results