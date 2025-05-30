import numpy as np
import random
from copy import deepcopy
from individual import Individual

class Population:
    """
    Population of individuals for the genetic algorithm.
    Includes elitism in replacement.
    """

    def __init__(self, size, n_features=32, elitism=1):
        self.size = size
        self.n_features = n_features
        self.elitism = elitism  # number of top individuals to keep
        self.individuals = [Individual(n_features=n_features) for _ in range(size)]

    def evaluate(self, X, y, best_params_path=None, weights_path=None, alpha=0.05):
        for i, individual in enumerate(self.individuals):
            print(f"\nEvaluating individual {i+1}/{len(self.individuals)}")
            individual.evaluate_fitness(X, y, best_params_path, weights_path, alpha)

    def get_best_individual(self):
        return max(self.individuals, key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf)

    def sort_by_fitness(self, reverse=True):
        self.individuals.sort(key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf, reverse=reverse)

    def select_parents(self, k=2, tournament_size=3):
        parents = []
        for _ in range(k):
            tournament = random.sample(self.individuals, tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf)
            parents.append(winner.copy())
        return parents

    def generate_offspring(self, num_offspring, mutation_rate=0.1):
        offspring = []
        while len(offspring) < num_offspring:
            parent1, parent2 = self.select_parents(2)
            child1, child2 = parent1.crossover(parent2)
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            offspring.extend([child1, child2])
        return offspring[:num_offspring]

    def replace_weakest(self, new_individuals):
        """
        Replace weakest individuals with new individuals,
        preserving the top `elitism` individuals.
        """
        self.sort_by_fitness()

        # Keep elites unchanged
        elites = self.individuals[:self.elitism]

        # Replace from weakest upwards
        self.individuals[-len(new_individuals):] = new_individuals

        # Restore elites at front
        self.individuals[:self.elitism] = elites

    def __str__(self):
        return '\n\n'.join([f"Individual {i+1}:\n{str(ind)}" for i, ind in enumerate(self.individuals)])
