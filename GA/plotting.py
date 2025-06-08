# this code provides plotting utilities for visualizing the progress and behavior of a genetic algorithm
# it includes functions for plotting fitness trends, feature selection counts, tournament selection stats, and average fitness across trials

import matplotlib.pyplot as plt
import numpy as np

# this function plots the best and average fitness across generations
def plot_fitness_over_generations(num_generations, population_history, save_path=None):
    best_fitnesses = []
    avg_fitnesses = []

    for pop in population_history:
        best = pop.get_best_individual()  # get the best individual in the population
        best_fitnesses.append(best.fitness)
        # compute average fitness of the population
        avg_fitnesses.append(
            sum(ind.fitness for ind in pop.individuals) / len(pop.individuals)
        )

    generations = list(range(1, len(population_history) + 1))  # generation indices

    # create the plot
    plt.figure(figsize=(7, 5))
    plt.plot(generations, best_fitnesses, label='Best Fitness', marker='o')
    plt.plot(generations, avg_fitnesses, label='Average Fitness', linestyle='--')
    plt.title("Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# this function plots the number of selected features for best individual and average population across generations
def plot_feature_counts_over_generations(num_generations, population_history, save_path=None):
    best_counts = []
    avg_counts = []

    for pop in population_history:
        best = pop.get_best_individual()
        best_counts.append(best.num_selected_features())
        # compute average number of selected features
        avg_counts.append(
            sum(ind.num_selected_features() for ind in pop.individuals) / len(pop.individuals)
        )

    generations = list(range(1, len(population_history) + 1))  # generation indices

    # create the plot
    plt.figure(figsize=(7, 5))
    plt.plot(generations, best_counts, label='Best Individual', marker='o')
    plt.plot(generations, avg_counts, label='Population Average', linestyle='--')
    plt.title("Selected Features Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Number of Selected Features")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# this function plots a bar chart of how often each individual was selected during tournament selection
def plot_tournament_selection_bar(selection_counts, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(selection_counts)), selection_counts, color='skyblue')
    plt.title("Tournament Selection Counts")
    plt.xlabel("Individual Index")
    plt.ylabel("Selection Count")
    plt.xticks(range(len(selection_counts)))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# this function plots average best fitness across multiple runs over generations
def plot_avg_best_fitness_over_generations(all_histories, save_path=None):
    max_gens = max(len(h) for h in all_histories)  # find the max number of generations across all trials
    avg_best_per_gen = []

    for gen_idx in range(max_gens):
        gen_fitnesses = []
        for hist in all_histories:
            if gen_idx < len(hist):
                best_ind = hist[gen_idx].get_best_individual()
                gen_fitnesses.append(best_ind.fitness)
        # compute average best fitness for this generation across trials
        avg_best = np.mean(gen_fitnesses)
        avg_best_per_gen.append(avg_best)

    generations = list(range(1, len(avg_best_per_gen) + 1))  # generation indices

    # create the plot
    plt.figure(figsize=(7, 5))
    plt.plot(generations, avg_best_per_gen, label='Average Best Fitness Across Trials', color='darkorange', marker='o')
    plt.title("Average Best Fitness Over Generations (10 Trials)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    
    # save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()     
