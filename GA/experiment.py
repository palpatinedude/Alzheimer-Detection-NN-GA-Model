# this class manages running multiple genetic algorithm experiments with different parameter sets
# it performs multiple trials per parameter set, collects results, saves data, and generates plots
# goal is to evaluate and compare genetic algorithm configurations on validation data

import os
import numpy as np
from .ga import GeneticAlgorithm
from .reporting import save_trial_details
from .plotting import plot_fitness_over_generations, plot_feature_counts_over_generations, plot_tournament_selection_bar, plot_avg_best_fitness_over_generations

class Experiment:
    # initialize experiment with validation data, parameter sets, and configuration options
    def __init__(self, X_val, y_val,param_sets,best_params_path, weights_path, results_dir, n_trials=10, max_generations=100, elitism=1,
                 ):
        self.X_val = X_val
        self.y_val = y_val
        self.param_sets = param_sets
        self.best_params_path = best_params_path
        self.weights_path = weights_path
        self.results_dir = results_dir
        self.n_trials = n_trials
        self.max_generations = max_generations
        self.elitism = elitism
    

    # run one trial of the genetic algorithm with specified parameters and random seed
    # returns best individual, population history, and other trial statistics
    def run_single_trial(self, params, trial_seed,model):
        # set seed for reproducibility
        np.random.seed(trial_seed)

        # initialize genetic algorithm with trial-specific parameters
        ga = GeneticAlgorithm(
            n_features=self.X_val.shape[1],
            num_generations=self.max_generations,
            elitism=self.elitism,
            best_params_path=self.best_params_path,
            weights_path=self.weights_path,
            pop_size=params['pop_size'],
            mutation_rate=params['mutation_prob'],
            crossover_prob=params['crossover_prob'],
            alpha=0.05
            
        )

        # run genetic algorithm to get best individual and population history
        best_ind, generations, pop_history = ga.run(model,self.X_val, self.y_val)

        # calculate average number of features selected per generation
        avg_features_by_generation = [
            np.mean([ind.num_selected_features() for ind in population.individuals])
            for population in pop_history
        ]

        # return dictionary containing key results and data from this trial
        return {
            'fitness': best_ind.fitness,
            'generations': generations,
            'mask': best_ind,
            'population_history': pop_history,
            'avg_features_by_generation': avg_features_by_generation,
            'selection_counts': ga.selection_counts
        }

    # aggregate average feature selection counts across all trials of a parameter set
    def aggregate_trial_results(self, trial_results):
        # collect feature counts from each trial
        all_avg_features = [trial['avg_features_by_generation'] for trial in trial_results]
        # combine and compute overall average selected features
        concatenated = np.concatenate(all_avg_features)
        overall_average = np.mean(concatenated)
        return overall_average

    # generate various plots to visualize results for a parameter set
    def plot_results(self, best_trial_population_history, all_histories, selection_counts, param_results_dir, idx):
        # plot average best fitness across trials over generations
        plot_avg_best_fitness_over_generations(
            all_histories,
            save_path=os.path.join(param_results_dir, f"params_set_{idx}_avg_best_fitness_across_trials.png")
        )
        # plot fitness progression for best single trial
        if best_trial_population_history is not None:
            plot_fitness_over_generations(
            len(best_trial_population_history), best_trial_population_history,
            os.path.join(param_results_dir, f"params_set_{idx}_fitness_plot.png")
            )
        else:
            print(f"Warning: No population history available for parameter set {idx}. Skipping fitness plot.")
        # plot feature counts over generations for best trial
        plot_feature_counts_over_generations(
            len(best_trial_population_history), best_trial_population_history,
            os.path.join(param_results_dir, f"params_set_{idx}_feature_counts_plot.png")
        )
        # plot tournament selection counts as bar chart
        plot_tournament_selection_bar(
            selection_counts,
            save_path=os.path.join(param_results_dir, f"params_set_{idx}_tournament_selection_bar.png")
        )

    # save detailed trial results and summary to disk for a parameter set
    def save_results(self, idx, params, trial_results, param_results_dir, avg_selected_features_across_trials):
        save_trial_details(idx, params, trial_results, param_results_dir, avg_selected_features_across_trials)

    # select the best parameter set based on average fitness, generations, and selected features
    def select_best_param_set(self, results):

        # convert results to numpy arrays for processing
        fitnesses = np.array([r['Average Best Fitness'] for r in results])
        generations = np.array([r['Average Generations'] for r in results])
        selected_features = np.array([r['Average Selected Features'] for r in results])

        # find index with highest fitness
        max_fitness = fitnesses.max()
        candidate_indices = np.where(fitnesses == max_fitness)[0]

        # among candidates with max fitness, pick those with minimal avg generations
        min_gen = generations[candidate_indices].min()
        candidate_indices = candidate_indices[generations[candidate_indices] == min_gen]

        # among those, pick with minimal selected features
        min_features = selected_features[candidate_indices].min()
        final_index = candidate_indices[selected_features[candidate_indices] == min_features][0]

        return results[final_index]

    # main method to run experiments for all parameter sets and aggregate results
    def run(self,model):
        all_results = []

        for idx, params in enumerate(self.param_sets, 1):
            print(f"\n=== running parameter set {idx} ===")
            param_results_dir = os.path.join(self.results_dir, f"SET{idx}")
            os.makedirs(param_results_dir, exist_ok=True)

            trial_results = []
            best_trial = None
            best_trial_fitness = -np.inf
            best_trial_index = -1
            best_trial_population_history = None

            # run multiple trials for current parameter set
            for trial in range(self.n_trials):
                print(f" trial {trial + 1}/{self.n_trials}")
                result = self.run_single_trial(params, trial,model)
                trial_results.append(result)

                # update best trial info if current trial has better fitness
                if result['fitness'] > best_trial_fitness:
                    best_trial_fitness = result['fitness']
                    best_trial = result['mask']
                    best_trial_index = trial
                    best_trial_population_history = result['population_history']

            # calculate average selected features across all trials
            avg_selected_features_across_trials = self.aggregate_trial_results(trial_results)

            # save results to disk for current parameter set
            self.save_results(idx, params, trial_results, param_results_dir, avg_selected_features_across_trials)

            # assume selection counts are consistent, take from first trial
            selection_counts = trial_results[0]['selection_counts']

            # gather population histories from all trials for plotting
            all_histories = [trial['population_history'] for trial in trial_results]

            # generate and save plots for current parameter set
            self.plot_results(best_trial_population_history, all_histories, selection_counts, param_results_dir, idx)

            # compute average fitness and generation count across trials
            avg_best_fitness = np.mean([trial['fitness'] for trial in trial_results])
            avg_generations = np.mean([trial['generations'] for trial in trial_results])

            # append summarized results for reporting
            all_results.append({
                'Set': idx,
                'Population Size': params['pop_size'],
                'Crossover Probability': params['crossover_prob'],
                'Mutation Probability': params['mutation_prob'],
                'Average Best Fitness': avg_best_fitness,
                'Average Generations': avg_generations,
                'Best Trial Index': best_trial_index + 1,
                'Best Trial Fitness': best_trial_fitness,
                'Average Selected Features': avg_selected_features_across_trials,
                'Best Individual Mask': best_trial.chromosome.tolist()

            })


        best_set = self.select_best_param_set(all_results)
        print("\n=== Best Parameter Set ===")
        print(best_set)    

        # return summary of results for all parameter sets
        return all_results, best_set
