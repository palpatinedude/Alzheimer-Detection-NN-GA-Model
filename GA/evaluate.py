# this script evaluates a model using a masked input and returns metrics like loss, accuracy, precision, recall, and F1 score.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def evaluate_model(model, X, y, mask):
    X_masked = np.zeros_like(X)
    X_masked[:, mask] = X[:, mask]

    y = y.astype(np.float32).reshape(-1, 1) 
    
    evaluation_results = model.evaluate(X_masked, y, verbose=0)
    loss = evaluation_results[0]
    accuracy = evaluation_results[1] if len(evaluation_results) > 1 else None

    y_pred_probs = model.predict(X_masked)
    y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_pred)

    return {
        'loss': loss,
        'eval_accuracy': accuracy,
        'sklearn_accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc
    }
