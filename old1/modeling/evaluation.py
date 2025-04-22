import numpy as np
from modeling.metrics import calculate_metrics, get_confusion_matrix

def evaluate_model(model, X_val, y_val, model_type='ann'):
    if model_type == 'ann':
        y_pred = (model.predict(X_val) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_val)

    metrics = calculate_metrics(y_val, y_pred)
    confusion = get_confusion_matrix(y_val, y_pred)
    return (*metrics, confusion)

