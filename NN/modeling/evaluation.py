# this file evalu.ates model performance during validation using predictions, metrics, and confusion matrix
# also contains functions to update the best configuration based on composite score



from NN.modeling.metrics import get_model_metrics
from NN.helpers import is_ann, predict_labels
from NN.config import WEIGHTS
from NN.modeling.metrics import composite_score

'''
from modeling.metrics import get_model_metrics, composite_score
from helpers import is_ann, predict_labels
from config import WEIGHTS
'''
import numpy as np


# this function evaluates a model on the validation set and returns predictions, eval results, and metrics
def evaluate_performance(model, X_val, y_val, model_type):
    y_val = np.array(y_val).astype(np.float32).reshape(-1, 1)
    if is_ann(model_type):  # Check if the model is an ANN
        # get loss and metrics
        eval_results = model.evaluate(X_val, y_val, verbose=0)
        # get predicted labels (thresholded for binary classification)
        all_y_pred = predict_labels(model, X_val, model_type)
    else:
        eval_results = None  # no eval results for non ann models
        all_y_pred = predict_labels(model, X_val, model_type)

    metrics = get_model_metrics(model, X_val, y_val, model_type)
    
    return all_y_pred, eval_results, metrics

# this function updates the best configuration if the current score is better than the best score
def update_best_configuration(current_score, best_score, best_result, fold_res, final, best_config, 
                               lr=None, m=None, reg=None):
    # check if the current score is better than the best score 
    if current_score > best_score:
        best_score = current_score
        best_result = (fold_res, final)
        
        # update best configuration for learning rate and momentum
        if lr is not None and m is not None:
            if best_config.get('lr_m') != (lr, m):
                best_config['lr_m'] = (lr, m)
        
        # update best configuration for regularization 
        if reg is not None:
            if best_config.get('reg') != (reg,): 
                best_config['reg'] = (reg,)
    print(f"Updated Best Configuration: {best_config}")
    
    return best_score, best_result, best_config


# this function evaluates the model performance and updates the best configuration if needed
def evaluate_and_update(results, final, fold_res, best_score, best_result, best_config, 
                        lr=None, m=None, reg=None, verbose_info=""):
    # collect metrics
    result = {
        'learning_rate': lr,
        'momentum': m,
        'regularization': reg,
        'accuracy': final['accuracy'],
        'ce_loss': final['ce_loss'],
        'mse': final['mse'],
        'epoch_accuracy': final['epoch_accuracy'],
        'val_epoch_accuracy': final['val_epoch_accuracy'],
        'avg_training_time': final['avg_training_time'],
        'avg_epochs_to_converge': final['avg_epochs_to_converge']
    }
    results.append(result)

    # overfitting check
    train_acc = final['epoch_accuracy'][-1]
    val_acc = final['val_epoch_accuracy'][-1]
    acc_gap = train_acc - val_acc
    if acc_gap > 0.1:
        print(f" Overfitting detected (Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}) {verbose_info}")

    # composite score
    current_score = composite_score(final, WEIGHTS)

    # update best configuration
    best_score, best_result, best_config = update_best_configuration(
        current_score, best_score, best_result, fold_res, final, best_config,lr=lr, m=m, reg=reg
    )

    return best_score, best_result, best_config