import keras 
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from A1_Preprocessing import plot_confusion_matrix
# Define neural network model
def create_model(input_dim, hidden_units, activation_function, output_layer='sigmoid'):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # Input layer
    model.add(Dense(hidden_units, activation=activation_function))  # Hidden layer
    model.add(Dense(1, activation=output_layer))  # Output layer for binary classification
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# K-Fold Evaluation
def k_fold_evaluation(X, y, n_splits=5, hidden_units=64, activation_function='relu', epochs=100, batch_size=32):
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = {
            'accuracy': [],
            'loss': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'confusion_matrices': []
    }

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nEvaluating Fold {fold}/{n_splits}")
            
        # Split data into train and validation sets
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
        # Create and train the model
        model = create_model(X_train.shape[1], hidden_units, activation_function, 'sigmoid')
            
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
            
        # Evaluate on validation data
        y_pred = (model.predict(X_val) > 0.5).astype(int)
            
        accuracy = accuracy_score(y_val, y_pred)
        loss = model.evaluate(X_val, y_val, verbose=0)[0]
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred)
            
        # Store results for the fold
        fold_results['accuracy'].append(accuracy)
        fold_results['loss'].append(loss)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1'].append(f1)
        fold_results['roc_auc'].append(roc_auc)
        fold_results['confusion_matrices'].append(confusion_matrix(y_val, y_pred))

        # Collect all true and predicted values for final confusion matrix
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

    # Plot a single confusion matrix for all folds combined
    plot_confusion_matrix(
        all_y_true, 
        all_y_pred, 
        title=f"Combined Confusion Matrix (Hidden Units: {hidden_units}, Activation: {activation_function})"
    )
        
    return fold_results

def evaluate_with_different_hidden_units(X_train, y_train, X_test, y_test):
    """
    Experiment with different hidden units and activation functions.
    """
    input_dim = X_train.shape[1]
   # hidden_units_options = [input_dim // 2, 2 * input_dim // 3, input_dim, 2 * input_dim]
    hidden_units_options = [input_dim // 2]
   # activation_functions = ['relu', 'tanh', 'sigmoid']  # List of activation functions to experiment with
    activation_functions = ['relu']
    results = {}

    for hidden_units in hidden_units_options:
        for activation_function in activation_functions:
            print(f"\nEvaluating with {hidden_units} hidden units and {activation_function} activation function:")
            fold_results = k_fold_evaluation(X_train, y_train, n_splits=5, hidden_units=hidden_units, activation_function=activation_function, epochs=100, batch_size=32)
            results[(hidden_units, activation_function)] = fold_results
    
    # Display results for each configuration
    for (hidden_units, activation_function), fold_results in results.items():
        print(f"\nResults for {hidden_units} hidden units and {activation_function} activation function:")
        for metric in fold_results:
            print(f"{metric}: {np.mean(fold_results[metric]):.4f}")
    
