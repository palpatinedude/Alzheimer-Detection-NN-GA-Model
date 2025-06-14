# this class manages a population of individuals for use in a genetic algorithm
# it handles evaluation, selection, crossover, mutation, and elitism-based replacement

import numpy as np
import warnings

from .individual import Individual
from .config import TOURNAMENT_SIZE 

'''
from individual import Individual
from config import TOURNAMENT_SIZE  #  select individuals randomly from the population and choose the best one among them
'''
# represents a population of individuals in a genetic algorithm
class Population:

    # initializes the population with a given size and number of features
    def __init__(self, size, n_features=32, elitism=1):
        self.size = size
        self.n_features = n_features
        self.elitism = elitism  # number of top individuals to retain (elitism)
        self.individuals = [Individual(n_features=n_features) for _ in range(size)]  # initialize individuals

    # evaluates the fitness of every individual in the population using the provided model and validation data
    def evaluate(self, model, X_val, y_val, alpha=0.05):

        # evaluate all individuals using the same pretrained model
        for i, individual in enumerate(self.individuals):
            individual.evaluate_fitness(model, X_val, y_val, alpha)


    # returns the individual with the highest fitness in the population
    def get_best_individual(self):
        if any(ind.fitness is None for ind in self.individuals):
            warnings.warn("Some individuals have undefined fitness when calling get_best_individual.")
        return max(self.individuals, key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf)


    # sorts the population in descending order of fitness
    def sort_by_fitness(self, reverse=True):
        if any(ind.fitness is None for ind in self.individuals):
            warnings.warn("Some individuals have undefined fitness when calling sort_by_fitness.")
        self.individuals.sort(key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf, reverse=reverse)


    # selects parent individuals using tournament selection
    def select_parents(self, k=2, selection_counts=None):
        parents = []
        for _ in range(k):
            # randomly select individuals for tournament
            tournament_indices = np.random.choice(len(self.individuals), TOURNAMENT_SIZE, replace=False)
            tournament = [self.individuals[i] for i in tournament_indices]

            # choose the best from the tournament
            winner = max(tournament, key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf)

            # record index before copying to avoid object identity issue
            if selection_counts is not None:
                winner_idx = self.individuals.index(winner)
                selection_counts[winner_idx] += 1

            parents.append(winner.copy())

        return parents


    # generates new individuals through crossover and mutation
    def generate_offspring(self, num_offspring, mutation_rate=0.1, crossover_prob=0.6, selection_counts=None):
        offspring = []
        while len(offspring) < num_offspring:
            parent1, parent2 = self.select_parents(2, selection_counts=selection_counts) # select two parents

            if np.random.rand() < crossover_prob: # perform crossover with a given probability
                child1, child2 = parent1.crossover(parent2) # create two children from parents
            else:
                child1, child2 = parent1.copy(), parent2.copy() # if no crossover, just copy parents

            child1.mutate(mutation_rate) # mutate each child with the mutation rate
            child2.mutate(mutation_rate)

            for child in (child1, child2): # ensure both children are valid individuals
                if len(offspring) < num_offspring:
                    offspring.append(child)
                else:
                    break

        return offspring


    # replaces the weakest individuals in the population with new ones, preserving top elites
    def replace_weakest(self, new_individuals):
        self.sort_by_fitness()  # sort individuals by fitness

        # retain top elites
        elites = self.individuals[:self.elitism]

        # replace weakest individuals with new offspring
        self.individuals[-len(new_individuals):] = new_individuals

        # reinsert elites at the top of the population
        self.individuals[:self.elitism] = elites