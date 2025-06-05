import numpy as np
import os
from config import VAL_DATA_PATH, BEST_PARAM, WEIGHTS, RESULTS_DIR_GA, MAX_GENERATIONS, ELITISM
from experiment import Experiment

def main():
    val_data = np.load(VAL_DATA_PATH)
    X_val = val_data["X_val"]
    y_val = val_data["y_val"]

    param_sets = [
        {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.00},
        {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.01}
       # {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
       # {'pop_size': 20, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
       # {'pop_size': 20, 'crossover_prob': 0.1, 'mutation_prob': 0.01},
        #{'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.00},
        #{'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.01},
        #{'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
        #{'pop_size': 200, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
        #{'pop_size': 200, 'crossover_prob': 0.1, 'mutation_prob': 0.01} 
    ]

    '''
    
    {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
    {'pop_size': 20, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
    {'pop_size': 20, 'crossover_prob': 0.1, 'mutation_prob': 0.01},
    {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.00},
    {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.01},
    {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
    {'pop_size': 200, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
    {'pop_size': 200, 'crossover_prob': 0.1, 'mutation_prob': 0.01},
    '''
    experiment = Experiment(
        X_val=X_val,
        y_val=y_val,
        param_sets=param_sets,
        n_trials=10,
        max_generations=MAX_GENERATIONS,
        elitism=ELITISM,
        best_params_path=BEST_PARAM,
        weights_path=WEIGHTS,
        results_dir=RESULTS_DIR_GA
    )

    results = experiment.run()

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
