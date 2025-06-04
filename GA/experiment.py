import os
import numpy as np
from GA import GeneticAlgorithm
from reporting import save_trial_details
from plotting import plot_fitness_over_generations, plot_feature_counts_over_generations, plot_tournament_selection_bar


class Experiment:
    def __init__(self, X_val, y_val, param_sets, n_trials=10, max_generations=100, elitism=1,
                 best_params_path=None, weights_path=None, results_dir=None):
        self.X_val = X_val
        self.y_val = y_val
        self.param_sets = param_sets
        self.n_trials = n_trials
        self.max_generations = max_generations
        self.elitism = elitism
        self.best_params_path = best_params_path
        self.weights_path = weights_path
        self.results_dir = results_dir

    def run(self):
        all_results = []

        for idx, params in enumerate(self.param_sets, 1):
            print(f"\n=== Running parameter set {idx} ===")
            param_results_dir = os.path.join(self.results_dir, f"SET{idx}")
            os.makedirs(param_results_dir, exist_ok=True)

            pop_size = params['pop_size']
            cross_prob = params['crossover_prob']
            mut_prob = params['mutation_prob']

            best_fitnesses = []
            generations_to_converge = []
            trial_results = []
            best_trial = None
            best_trial_index = -1
            best_trial_fitness = -np.inf
            best_trial_population_history = None

            for trial in range(self.n_trials):
                print(f" Trial {trial + 1}/{self.n_trials}")
                np.random.seed(trial)

                ga = GeneticAlgorithm(
                    n_features=self.X_val.shape[1],
                    num_generations=self.max_generations,
                    elitism=self.elitism,
                    best_params_path=self.best_params_path,
                    weights_path=self.weights_path,
                    pop_size=pop_size,
                    mutation_rate=mut_prob,
                    crossover_prob=cross_prob,
                    alpha=0.05,
                )

                best_ind, generations, pop_history = ga.run(self.X_val, self.y_val)

                best_fitnesses.append(best_ind.fitness)
                generations_to_converge.append(generations)

                trial_results.append({
                    'fitness': best_ind.fitness,
                    'generations': generations,
                    'mask': best_ind,
                    'population_history': pop_history
                })

                if best_ind.fitness > best_trial_fitness:
                    best_trial_fitness = best_ind.fitness
                    best_trial = best_ind
                    best_trial_index = trial
                    best_trial_population_history = ga.population_history

            # Save trial logs and stats
            save_trial_details(idx, params, trial_results, param_results_dir)
            selection_counts = ga.selection_counts

            plot_fitness_over_generations(len(best_trial_population_history), best_trial_population_history,
                                          os.path.join(param_results_dir, f"params_set_{idx}_fitness_plot.png"))
            plot_feature_counts_over_generations(len(best_trial_population_history), best_trial_population_history,
                                                 os.path.join(param_results_dir, f"params_set_{idx}_feature_counts_plot.png"))
            plot_tournament_selection_bar(selection_counts,
                                          save_path=os.path.join(param_results_dir, f"params_set_{idx}_tournament_selection_bar.png"))

            avg_best_fitness = np.mean(best_fitnesses)
            avg_generations = np.mean(generations_to_converge)

            all_results.append({
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

        return all_results
