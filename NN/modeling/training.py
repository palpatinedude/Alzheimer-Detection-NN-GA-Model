# this script handles training and evaluation of both ann and logistic regression models
# using cross-validation, with support for early stopping, plotting, and metric logging

import time
from keras.callbacks import EarlyStopping
from config import EPOCHS, BATCH_SIZE, PATIENCE
from helpers import is_ann

# this function trains an artificial neural network model with optional early stopping
def train_ann(model, X_train, y_train, X_val=None, y_val=None):
    # set up early stopping if validation data is provided
    if X_val is not None and y_val is not None:
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        validation_data = (X_val, y_val)
        callbacks = [early_stopping]
    else:
        validation_data = None
        callbacks = []

    # start training and track time
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=validation_data,
        verbose=0,
        callbacks=callbacks
    )
    training_time = time.time() - start_time

    # get the number of epochs actually run
    epochs_ran = len(history.history['loss'])
    if 'val_loss' in history.history:
        epochs_ran = len(history.history['val_loss'])

    # return training metadata and accuracy history
    return {
        'training_time': training_time,
        'epochs_ran': epochs_ran,
        'history': history,
        'accuracy_history': history.history['accuracy'],
        'val_accuracy_history': history.history.get('val_accuracy', [])
    }

# this function trains a non-ann model like logistic regression
def train_non_ann(model, X_train, y_train):
    # start training and track time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # return basic training metadata without epoch or history
    return {
        'training_time': training_time,
        'epochs_ran': None,
        'history': None,
        'accuracy_history': None
    }

# this function selects the right training function based on model type
def train_model(model, X_train, y_train, X_val=None, y_val=None, model_type='ann'):
    if is_ann(model_type):
        return train_ann(model, X_train, y_train, X_val, y_val)
    else:
        return train_non_ann(model, X_train, y_train)
