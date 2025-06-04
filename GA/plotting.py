import matplotlib.pyplot as plt

def plot_fitness_over_generations(num_generations, population_history, save_path=None):
    best_fitnesses = []
    avg_fitnesses = []

    for pop in population_history:
        best = pop.get_best_individual()
        best_fitnesses.append(best.fitness)
        avg_fitnesses.append(
            sum(ind.fitness for ind in pop.individuals) / len(pop.individuals)
        )
    generations = list(range(1, len(population_history) + 1))

    plt.figure(figsize=(7, 5))
    plt.plot(generations, best_fitnesses, label='Best Fitness', marker='o')
    plt.plot(generations, avg_fitnesses, label='Average Fitness', linestyle='--')
    plt.title("Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_feature_counts_over_generations(num_generations, population_history, save_path=None):
    best_counts = []
    avg_counts = []

    for pop in population_history:
        best = pop.get_best_individual()
        best_counts.append(best.num_selected_features())
        avg_counts.append(
            sum(ind.num_selected_features() for ind in pop.individuals) / len(pop.individuals)
        )

    generations = list(range(1, len(population_history) + 1))

    plt.figure(figsize=(7, 5))
    plt.plot(generations, best_counts, label='Best Individual', marker='o')
    plt.plot(generations, avg_counts, label='Population Average', linestyle='--')
    plt.title("Selected Features Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Number of Selected Features")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



def plot_tournament_selection_bar(selection_counts, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(selection_counts)), selection_counts, color='skyblue')
    plt.title("Tournament Selection Counts")
    plt.xlabel("Individual Index")
    plt.ylabel("Selection Count")
    plt.xticks(range(len(selection_counts)))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()