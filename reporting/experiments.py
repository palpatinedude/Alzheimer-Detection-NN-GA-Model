# this script selects the best configuration for hidden layer size , hyperparameters and regularization coefficient


from config import RESULTS_DIR,REGULARIZATION_VALUES
from reporting.result_saving import save_results_hidden, save_results_hyper,save_results_regularization
from modeling.tuning import tuning_hidden, tuning_hyper, tuning_regularization
from visualization.evalutation_plots import plot_comparison_accuracy_ce_loss, plot_hyper_comparison


# select the best hidden layer size based on cross validation results and save results
def select_best_config_hidden(X_scaled, y_train):
    # initialize tracking variables
    best_score = 0
    best_config = None
    best_result = None
    results_all = {True: {}, False: {}}
    summary_table = []

    # evaluate with different hidden unit configurations
    results_all, best_score, best_config, best_result, hidden_units_options = tuning_hidden(
        X_scaled, y_train, results_all, best_score, best_config, best_result
    )

    # compare accuracy and cross entropy loss for different configurations
    plot_comparison_accuracy_ce_loss(results_all, hidden_units_options,RESULTS_DIR)

    # save results for each configuration
    save_results_hidden(RESULTS_DIR,  summary_table, results_all, hidden_units_options)

    # print the best found configuration
    print(f"\nBest Configuration: H1 = {best_config} (Accuracy = {best_score:.4f})")
    return best_config, best_result, results_all


# select the best learning rate and momentum for the given hidden units
def select_best_config_hyper(X_scaled, y_train, hidden_units):
    # initialize tracking variables
    best_score = 0
    best_config = {'lr_m': (None, None), 'reg': (None,)} 
    best_result = None
    results = []

    # display what hyperparameter range is being searched
    print(f"\nSearching best learning rate & momentum for H1 = {hidden_units}")
    
    # perform tuning over learning rate and momentum
    results, best_score, best_config, best_result = tuning_hyper(
        X_scaled, y_train, hidden_units,
        results, best_score, best_config, best_result
    )

    # compare accuracy and cross entropy loss for different combinations
    plot_hyper_comparison(results, RESULTS_DIR)

    # save all hyperparameter results
    save_results_hyper(RESULTS_DIR, results)

    # print the best combination found
    best_accuracy = best_result[1]['accuracy']  # validation accuracy
    best_composite_score = best_score  # sum score

    print(f"Best Hyperparameter Combination: Learning Rate = {best_config['lr_m'][0]}, Momentum = {best_config['lr_m'][1]}")
    print(f"  Validation Accuracy = {best_accuracy:.4f}, Composite Score = {best_composite_score:.4f}")
    return best_config['lr_m'][0], best_config['lr_m'][1], best_result[0], best_result[1] # return best hyperparameter config and results
    


# select the best regularization coef for the given hidden units, learning rate, momentum, and transformation method
def select_best_config_regularization(X_scaled, y_train, hidden_units,  best_lr, best_momentum):
    # initialize tracking variables
    best_score = 0
    best_config = {'lr_m': (None, None), 'reg': (None,)} 
    best_result = None
    results = []

    # display what regularization range is being searched
    print(f"\nSearching best regularization for H1 = {hidden_units}, LR = {best_lr}, Momentum = {best_momentum}")

    # perform tuning over regularization
    results, best_score, best_config, best_result = tuning_regularization(
        X_scaled, y_train, hidden_units,  best_lr, best_momentum,
        results, best_score, best_config, best_result
    )

    # save all regularization results
    save_results_regularization(RESULTS_DIR,  results)

    # print the best combination found
    best_accuracy = best_result[1]['accuracy']  # validation accuracy
    best_composite_score = best_score  # sum score

    # print the best regularization combination found
    print(f"Best Regularization: Lambda = {best_config['reg'][0]}")
    print(f" Validation Accuracy = {best_accuracy:.4f}, Composite Score = {best_composite_score:.4f}")
    return best_config['reg'][0], best_result[0], best_result[1]
    