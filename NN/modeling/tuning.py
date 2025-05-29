# this script handles tuning of hidden units and hyperparameters for ann models
# it finds the best configuration using 5-fold cross-validation

from modeling.cross_validation import cross_validate_model
from config import HIDDEN_UNIT_RATIOS, LEARNING_RATES, MOMENTUM_VALUES, REGULARIZATION_VALUES,WEIGHTS
from modeling.metrics  import composite_score
from modeling.evaluation import update_best_configuration, evaluate_and_update

# this function tunes the number of hidden units based on input size and stores the best result
def tuning_hidden(X_scaled, y_train, results_all, best_score, best_config, best_result):
    print("\nRunning hidden layer tuning with Standardization")
    input_dim = X_scaled.shape[1]

    # calculate hidden unit sizes based on input dimension and predefined ratios
    hidden_units_options = [int(input_dim * ratio) for ratio in HIDDEN_UNIT_RATIOS]

    # evaluate each hidden unit configuration
    for h_units in hidden_units_options:
        print(f"Evaluating: Hidden Units = {h_units}")
        fold_res, final = cross_validate_model(
            X_scaled, y_train, model_type='ann',
            hidden_units=h_units
        )
        results_all[h_units] = (fold_res, final)

    
        # update best result if current accuracy is better
        if final['accuracy'] > best_score:
            best_score = final['accuracy']
            best_config = h_units
            best_result = (fold_res, final)

    return results_all, best_score, best_config, best_result, hidden_units_options


# this function tunes learning rate and momentum using cross validation to find best hyperparameters
def tuning_hyper(X_scaled, y_train, hidden_units, results, best_score, best_config, best_result):

    best_accuracy = -1  # initialize

    for lr in LEARNING_RATES:
        for m in MOMENTUM_VALUES:
            print(f"Evaluating: Learning Rate = {lr}, Momentum = {m}")

            # train and evaluate
            fold_res, final = cross_validate_model(
                X_scaled, y_train, 'ann',
                hidden_units=hidden_units,
                learning_rate=lr,
                momentum=m
            )

            # collect metrics, check overfitting, and update best
            best_score, best_result, best_config = evaluate_and_update(
                results, final, fold_res,
                best_score, best_result, best_config,
                lr=lr, m=m, reg=None,
                verbose_info=f"for LR={lr}, M={m}"
            )


    return results, best_score, best_config, best_result



# this function tunes regularization using cross-validation to find best lambda
def tuning_regularization(X_scaled, y_train, hidden_units, learning_rate, momentum, results, best_score, best_config, best_result):

    best_accuracy = -1  # initialize

    for reg in REGULARIZATION_VALUES:
        print(f"Evaluating: Regularization = {reg}")

        # train and evaluate
        fold_res, final = cross_validate_model(
            X_scaled, y_train, 'ann',
            hidden_units=hidden_units,
            learning_rate=learning_rate,
            momentum=momentum,
            regularization=reg
        )

        # collect metrics, check overfitting, and update best
        best_score, best_result, best_config = evaluate_and_update(
            results, final, fold_res,
            best_score, best_result, best_config,
            lr=learning_rate, m=momentum, reg=reg,
            verbose_info=f"for Reg={reg}"
        )


    return results, best_score, best_config, best_result



