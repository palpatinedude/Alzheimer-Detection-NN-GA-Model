# ============================================================================================
# Project: Alzheimer’s Disease Diagnosis using Genetic Algorithm for Feature Selection
# Description: This script sets up and runs Genetic Algorithm (GA) experiments using
#    different combinations of population size, crossover probability, and mutation probability.
# Goal: Optimize feature subset selection (from 32 input features) to improve Alzheimer's 
#    prediction performance with a fixed ANN, by maximizing validation accuracy while reducing input dimensionality.
#    The script evaluates GA behavior across different parameter sets and prints summary results.
# Author: Marianthi Thodi
# AM: 1084576
# ============================================================================================

import numpy as np
from .reporting import save_best_set_config,save_metrics_to_file
from .config import VAL_DATA_PATH, BEST_PARAM, WEIGHTS, RESULTS_DIR_GA, MAX_GENERATIONS, ELITISM,TEST_DATA_PATH, MODEL,DATA
from .experiment import Experiment
import os
from tensorflow.keras.models import load_model
from .evaluate import evaluate_model
import pandas as pd


'''

    # define sets of GA parameters to test (population size, crossover, mutation)
    param_sets = [
        {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.00},
        {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.01},
       
        
    ]
'''

def main():
    val_data = np.load(VAL_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)
    
    X_val = val_data["X_val"]
    y_val = val_data["y_val"]
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    full_model_path = MODEL
    if full_model_path and os.path.exists(full_model_path):
        model = load_model(full_model_path)
        print("Loaded full keras model successfully.")
    else:
        raise FileNotFoundError("Model path not found.")

    param_sets = [
        {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.00},
        {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.01},
        {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
        {'pop_size': 20, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
        {'pop_size': 20, 'crossover_prob': 0.1, 'mutation_prob': 0.01},
        {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.00},
        {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.01},
        {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
        {'pop_size': 200, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
        {'pop_size': 200, 'crossover_prob': 0.1, 'mutation_prob': 0.01} 
    ]

    experiment = Experiment(
        X_val=X_val,
        y_val=y_val,
        param_sets=param_sets,
        best_params_path=BEST_PARAM,
        weights_path=WEIGHTS,
        results_dir=RESULTS_DIR_GA,
        n_trials=10,
        max_generations=MAX_GENERATIONS,
        elitism=ELITISM,
    )

    results, best_set = experiment.run(model)
    save_best_set_config(best_set, RESULTS_DIR_GA)
    print("\nBest parameter set info:")
    for k, v in best_set.items():
        print(f"{k}: {v}")

    mask = np.array(best_set['Best Individual Mask'], dtype=bool)

    print("\nEvaluating on Validation Data:")
    val_metrics = evaluate_model(model, X_val, y_val, mask)
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nEvaluating on Test Data:")
    test_metrics = evaluate_model(model, X_test, y_test, mask)
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # save all metrics to text file
    save_metrics_to_file(val_metrics, test_metrics, RESULTS_DIR_GA)

     # load original CSV dataset to get feature names
    df = pd.read_csv(DATA)
    df.drop(columns=['PatientID', 'DoctorInCharge'], inplace=True, errors='ignore')
    feature_names = df.columns[:-1]  # Exclude label column (assumed last)
    selected_feature_names = feature_names[mask]

    #  save selected feature names
    selected_features_path = os.path.join(RESULTS_DIR_GA, "selected_features.txt")
    with open(selected_features_path, "w") as f:
        f.write("Selected Features:\n")
        for name in selected_feature_names:
            f.write(name + "\n") 
    
    # save results to pass to the final united code
    best_set_summary = {
        'Best Individual Mask': best_set['Best Individual Mask'],
        'Average Selected Features': best_set['Average Selected Features'],
        'Population Size': best_set['Population Size'],
        'Mutation Probability': best_set['Mutation Probability'],
        'Crossover Probability': best_set['Crossover Probability'],
        'Validation Metrics': val_metrics,
        'Test Metrics': test_metrics
    }
    return  best_set_summary



if __name__ == "__main__":
    main()      