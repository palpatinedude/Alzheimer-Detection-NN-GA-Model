from config import WEIGHTS
from modeling.metrics import composite_score



def update_best_configuration(current_score, best_score, best_result, fold_res, final, best_config, 
                              num_layers=None, neurons_per_layer=None):
    if current_score > best_score:
        best_score = current_score
        best_result = (fold_res, final)
        best_config = {'layers': num_layers, 'neurons': neurons_per_layer}
        print(f"New Best Configuration: {best_config} | Score: {current_score:.4f}")
    return best_score, best_result, best_config


def evaluate_and_update(results, final, fold_res, best_score, best_result, best_config, 
                        num_layers=None, neurons_per_layer=None, verbose_info=""):

    result = {
        'num_layers': num_layers,
        'neurons_per_layer': neurons_per_layer,
        'accuracy': final['accuracy'],
        'ce_loss': final['ce_loss'],
        'mse': final['mse'],
        'epoch_accuracy': final['epoch_accuracy'],
        'val_epoch_accuracy': final['val_epoch_accuracy'],
        'avg_training_time': final['avg_training_time'],
        'avg_epochs_to_converge': final['avg_epochs_to_converge']
    }
    results.append(result)

    # check for overfitting
    if isinstance(final['epoch_accuracy'], list) and isinstance(final['val_epoch_accuracy'], list):
        acc_gap = final['epoch_accuracy'][-1] - final['val_epoch_accuracy'][-1]
        if acc_gap > 0.1:
            print(f"⚠️ Overfitting detected (Train Acc: {final['epoch_accuracy'][-1]:.4f}, "
                  f"Val Acc: {final['val_epoch_accuracy'][-1]:.4f}) {verbose_info}")

    # composite score for model comparison
    current_score = composite_score(final, WEIGHTS)

    # update best if this is the best so far
    best_score, best_result, best_config = update_best_configuration(
        current_score, best_score, best_result, fold_res, final, best_config,
        num_layers=num_layers, neurons_per_layer=neurons_per_layer
    )

    return best_score, best_result, best_config