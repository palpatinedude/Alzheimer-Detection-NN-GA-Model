from config import HIDDEN_UNIT_RATIOS, NUM_LAYERS
from bonus_dir.cross_validate import cross_validate_model_bonus
from bonus_dir.save import save_results_deep_network
import numpy as np
from bonus_dir.evaluation import evaluate_and_update
from modeling.metrics import compute_mean_epoch_accuracy

def tuning_hidden_layers_neurons(X_train_std, y_train, input_dim, learning_rate, momentum, reg):    
    results = []
    best_score = 0
    best_config = None
    best_result = None

    for ratio in HIDDEN_UNIT_RATIOS:
        for num_layers in NUM_LAYERS:
            neurons_per_layer = int(input_dim * ratio)
            hidden_units = [neurons_per_layer] * num_layers
            
            print(f"Evaluating configuration: Hidden Neurons = {hidden_units}, Layers = {num_layers}")
            
            fold_results, final_metrics, _, _,epoch_accuracies,val_epoch_accuracies= cross_validate_model_bonus(
                X_train_std, y_train, 
                hidden_units=hidden_units, 
                learning_rate=learning_rate, 
                momentum=momentum, 
                regularization=reg
            )

            final = {
                'accuracy': np.mean(final_metrics['accuracy']),
                'mse': np.mean(final_metrics['mse']),
                'ce_loss': np.mean(final_metrics['ce_loss']),
                'epoch_accuracy': compute_mean_epoch_accuracy(epoch_accuracies),
                'val_epoch_accuracy': compute_mean_epoch_accuracy(val_epoch_accuracies),
                'avg_training_time': np.mean(final_metrics['training_time']),
                'avg_epochs_to_converge': np.mean(final_metrics['epochs_to_converge']),
            }
 

            verbose_info = f"(Hidden Units: {neurons_per_layer}, Layers: {num_layers})"
            best_score, best_result, best_config = evaluate_and_update(
                results=results,
                final=final,
                fold_res=fold_results,
                best_score=best_score,
                best_result=best_result,
                best_config=best_config,
                num_layers=num_layers,
                neurons_per_layer=neurons_per_layer,
                verbose_info=verbose_info
            )

    save_results_deep_network(results, 'bonus_dir/Results/deep_network_results.txt')

    return best_config['layers'], best_config['neurons'], best_score, best_result

'''
def tuning_hidden_layers_neurons(X_train_std, y_train, input_dim, learning_rate, momentum, reg):    
    results = []
    best_score = 0
    best_config = None
    best_result = None
    
    for ratio in HIDDEN_UNIT_RATIOS:
        for num_layers in NUM_LAYERS:
            neurons_per_layer = int(input_dim * ratio)
            hidden_units = [neurons_per_layer] * num_layers
            
            print(f"Evaluating configuration: Hidden Neurons = {hidden_units}, Layers = {num_layers}")
            
            fold_results, final_metrics, _, _, _ = cross_validate_model_bonus(
                X_train_std, y_train, 
                hidden_units=hidden_units, 
                learning_rate=learning_rate, 
                momentum=momentum, 
                regularization=reg
            )
            
            result = {
                'hidden_neurons': hidden_units,
                'hidden_layers': num_layers,
                'ce_loss':  np.mean(final_metrics['ce_loss']),
                'mse': np.mean(final_metrics['mse']),
                'accuracy': np.mean(final_metrics['accuracy'])
            }
            
            #mean_accuracy = np.mean(final_metrics['accuracy'])
            mean_accuracy = result['accuracy']
            results.append(result)

            if mean_accuracy > best_score:
                best_score = mean_accuracy
                best_config = hidden_units
                best_result = (fold_results, final_metrics)

    save_results_deep_network(results, 'bonus_dir/Results/deep_network_results.txt')
    
    best_num_layers = len(best_config)
    best_num_neurons = best_config[0]
    return best_num_layers, best_num_neurons, best_score, best_result

'''   