from modeling.training import k_fold_evaluation
from config import HIDDEN_UNIT_RATIOS, LEARNING_RATES, MOMENTUM_VALUES, RESULTS_DIR
from reporting.result_saving import save_results_hidden, save_results_hyper
from  helpers import get_norm_label


# ------------------ Hidden Units Optimization ------------------ #

def select_best_config_hidden(scaled_data_options, y_train):
    best_score = 0
    best_config = None
    best_result = None
    results_all = {True: {}, False: {}}
    summary_table = []

    for norm, (X_scaled, label) in scaled_data_options.items():
        print(f"\nRunning configuration: {label}")
        input_dim = X_scaled.shape[1]
        hidden_units_options = [int(input_dim * ratio) for ratio in HIDDEN_UNIT_RATIOS]

        for h_units in hidden_units_options:
            fold_res, final = k_fold_evaluation(X_scaled, y_train,'ann' ,hidden_units=h_units, norm=norm)
            results_all[norm][h_units] = (fold_res, final)

            if final['accuracy'] > best_score:
                best_score = final['accuracy']
                best_config = (norm, h_units)
                best_result = (fold_res, final)

        save_results_hidden(RESULTS_DIR, norm, summary_table, results_all[norm], hidden_units_options)

    print(f"\nBest Configuration: {get_norm_label(best_config[0])} with H1 = {best_config[1]} (Accuracy = {best_score:.4f})")
    return best_config, best_result  # Return only the best configuration and final metrics

# ------------------ Hyperparameter Optimization ------------------ #

def select_best_config_hyper(X_scaled, y_train, hidden_units, best_norm):
    best_score = 0
    best_config = None
    best_result = None
    results = []

    print(f"\nSearching best learning rate & momentum for H1 = {hidden_units}")

    for lr in LEARNING_RATES:
        for m in MOMENTUM_VALUES:
            print(f"Evaluating: Learning Rate = {lr}, Momentum = {m}")

            _, final = k_fold_evaluation(
                X_scaled, y_train,'ann',
                hidden_units=hidden_units,
                norm=best_norm,
                learning_rate=lr,
                momentum=m
            )

            result = {
                'learning_rate': lr,
                'momentum': m,
                'accuracy': final['accuracy'],
                'ce_loss': final['ce_loss'],
                'mse': final['mse'],
                'epoch_accuracy': final['epoch_accuracy'],
                'avg_training_time': final['avg_training_time'],
                'avg_epochs_to_converge': final['avg_epochs_to_converge']
            }

            results.append(result)

            if final['accuracy'] > best_score:
                best_score = final['accuracy']
                best_config = (lr, m)
                best_result = result

    save_results_hyper(RESULTS_DIR, best_norm, results)
    print(f"\nBest Hyperparameter Combination: Learning Rate = {best_config[0]}, Momentum = {best_config[1]} (Accuracy = {best_score:.4f})")
    return best_config[0], best_config[1], best_result[0], best_result[1] # Return only the best configuration and final metrics
