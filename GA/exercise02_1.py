
'''
import numpy as np
import pandas as pd
from GA import GeneticAlgorithm
from config import VAL_DATA_PATH, BEST_PARAM, WEIGHTS, RESULTS_DIR_GA,MAX_GENERATIONS, ELITISM
from reporting import save_trial_details
from plotting import plot_fitness_over_generations, plot_feature_counts_over_generations, plot_tournament_selection_bar
import os

def evaluate_ga_parameters(X_val, y_val, param_sets, n_trials=10):
     results = []
     for idx, params in enumerate(param_sets, 1):
        print(f"\n=== Running parameter set {idx} ===")

        # Create a unique folder per parameter set
        param_results_dir = os.path.join(RESULTS_DIR_GA, f"SET{idx}")
        os.makedirs(param_results_dir, exist_ok=True)

        pop_size, cross_prob, mut_prob = params['pop_size'], params['crossover_prob'], params['mutation_prob']

        best_fitnesses = []
        generations_to_converge = []
        trial_results = []
        best_trial = None
        best_trial_index = -1
        best_trial_fitness = -np.inf
        best_trial_population_history = None

        for trial in range(n_trials):
            print(f" Trial {trial + 1}/{n_trials}")
            np.random.seed(trial)
            ga = GeneticAlgorithm(
                n_features=X_val.shape[1],
                num_generations=MAX_GENERATIONS,
                elitism=ELITISM,
                best_params_path=BEST_PARAM,
                weights_path=WEIGHTS,
                pop_size=pop_size,
                mutation_rate=mut_prob,
                crossover_prob=cross_prob,
                alpha=0.05,
            )

            best_ind, generations ,pop_history= ga.run(X_val, y_val)
            best_fitnesses.append(best_ind.fitness)
            generations_to_converge.append(generations)

            trial_results.append({
                'fitness': best_ind.fitness,
                'generations': generations,
                'mask': best_ind,
                'population_history': pop_history  # add for later plot
            })

            # Track best trial
            if best_ind.fitness > best_trial_fitness:
                best_trial_fitness = best_ind.fitness
                best_trial = best_ind
                best_trial_index = trial
                best_trial_population_history = ga.population_history

        # Save trial log + stats inside the parameter set folder
        save_trial_details(idx, params, trial_results, param_results_dir)
        selection_counts = ga.selection_counts 

        # Save best trial plots
        fitness_plot_path = os.path.join(param_results_dir, f"params_set_{idx}_fitness_plot.png")
        features_plot_path = os.path.join(param_results_dir, f"params_set_{idx}_feature_counts_plot.png")

        plot_fitness_over_generations(len(best_trial_population_history), best_trial_population_history, fitness_plot_path)
        plot_feature_counts_over_generations(len(best_trial_population_history), best_trial_population_history, features_plot_path)
        plot_tournament_selection_bar(selection_counts, save_path=os.path.join(param_results_dir, f"params_set_{idx}_tournament_selection_bar.png"))

        avg_best_fitness = np.mean(best_fitnesses)
        avg_generations = np.mean(generations_to_converge)


        results.append({
            'Set': idx,
            'Population Size': pop_size,
            'Crossover Probability': cross_prob,
            'Mutation Probability': mut_prob,
            'Average Best Fitness': avg_best_fitness,
            'Average Generations': avg_generations,
            'Best Trial Index': best_trial_index + 1,
            'Best Trial Fitness': best_trial_fitness,
            'Best Individual Mask': best_trial.chromosome.tolist() 
        })

     return results





# Load validation data
val_data = np.load(VAL_DATA_PATH)
X_val = val_data["X_val"]
y_val = val_data["y_val"]


# Parameter grid from the table
param_sets = [
    {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.00}
]

    {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.01},
    {'pop_size': 20, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
    {'pop_size': 20, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
    {'pop_size': 20, 'crossover_prob': 0.1, 'mutation_prob': 0.01},
    {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.00},
    {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.01},
    {'pop_size': 200, 'crossover_prob': 0.6, 'mutation_prob': 0.10},
    {'pop_size': 200, 'crossover_prob': 0.9, 'mutation_prob': 0.01},
    {'pop_size': 200, 'crossover_prob': 0.1, 'mutation_prob': 0.01},

# assuming you have validation data X_val, y_val loaded somewhere
results = evaluate_ga_parameters(X_val, y_val, param_sets, n_trials=10)

for r in results:
    print(r)
'''

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
        # Add other parameter sets here as needed
    ]

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
