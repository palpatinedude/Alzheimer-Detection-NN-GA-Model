import numpy as np
import pandas as pd
from GA import GeneticAlgorithm
from config import VAL_DATA_PATH, BEST_PARAM, WEIGHTS, RESULTS_DIR_GA,MAX_GENERATIONS, ELITISM, TOURNAMENT_SIZE
from reporting import save_results

def evaluate_ga_parameters(X_val, y_val, param_sets, n_trials=10):
    results = []
    

    for idx, params in enumerate(param_sets, 1):
        print(f"\n=== Running parameter set {idx} ===")
        pop_size, cross_prob, mut_prob = params['pop_size'], params['crossover_prob'], params['mutation_prob']

        best_fitnesses = []
        generations_to_converge = []
        
        for trial in range(n_trials):
            print(f" Trial {trial + 1}/{n_trials}")
            np.random.seed(trial)
            ga = GeneticAlgorithm(
                n_features=X_val.shape[1],
                num_generations=MAX_GENERATIONS,
                elitism=ELITISM,
                tournament_size=TOURNAMENT_SIZE,
                best_params_path=BEST_PARAM,
                weights_path=WEIGHTS,
                results_dir=RESULTS_DIR_GA,
                pop_size=pop_size,
                mutation_rate=mut_prob,
                crossover_prob=cross_prob,
                alpha=0.05,
                plots=True
            )

            best_ind, generations = ga.run(X_val, y_val)
            best_fitnesses.append(best_ind.fitness)
            generations_to_converge.append(generations)

        avg_best_fitness = np.mean(best_fitnesses)
        avg_generations = np.mean(generations_to_converge)

        results.append({
            'Population Size': pop_size,
            'Crossover Probability': cross_prob,
            'Mutation Probability': mut_prob,
            'Average Best Fitness': avg_best_fitness,
            'Average Generations': avg_generations
        })
        # Save results to a file
        print(f"Trial {trial+1}: Best fitness = {best_ind.fitness:.4f}, Generations = {generations}")
        save_results(params, avg_best_fitness, avg_generations, output_dir=RESULTS_DIR_GA)

    return results






# Load validation data
val_data = np.load(VAL_DATA_PATH)
X_val = val_data["X_val"]
y_val = val_data["y_val"]


# Parameter grid from the table
param_sets = [
    {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.00}
]
'''
    {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.01},
    {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
    {'pop_size': 20, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
    {'pop_size': 20, 'crossover_prob': 0.1, 'mutation_prob': 0.01},
    {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.00},
    {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.01},
    {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
    {'pop_size': 200, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
    {'pop_size': 200, 'crossover_prob': 0.1, 'mutation_prob': 0.01},
'''



# assuming you have validation data X_val, y_val loaded somewhere
results = evaluate_ga_parameters(X_val, y_val, param_sets, n_trials=10)

for r in results:
    print(r)