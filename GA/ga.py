# this class implements a genetic algorithm to optimize feature selection
# the goal is to find a subset of features that maximizes model performance while minimizing complexity
# it evolves a population of candidate feature subsets through selection, crossover, and mutation
# using a neural network model's validation performance as fitness guidance
# early stopping is supported based on improvement thresholds to avoid overfitting 


import numpy as np
import copy

from .population import Population
from .config import PATIENCE, IMPROVE_THRESHOLD

'''
from population import Population
from config import PATIENCE, IMPROVE_THRESHOLD
'''
class GeneticAlgorithm:
    # initializes the genetic algorithm with all required parameters and configuration
    def __init__(self, n_features, num_generations, elitism, best_params_path, weights_path,
                 pop_size=30, mutation_rate=0.1, crossover_prob=0.6, alpha=0.05):
        self.n_features = n_features 
        self.num_generations = num_generations  
        self.elitism = elitism  
        self.best_params_path = best_params_path  
        self.weights_path = weights_path 
        self.pop_size = pop_size  
        self.mutation_rate = mutation_rate  
        self.alpha = alpha 
        self.crossover_prob = crossover_prob  
        self.population_history = []  
        self.selection_counts = [0] * pop_size  

    # runs the genetic algorithm on the provided validation dataset
    def run(self, model,X_val, y_val, patience=PATIENCE, improve_threshold=IMPROVE_THRESHOLD):
        population = self.initialize_population()
        population.evaluate(model, X_val, y_val, self.alpha)
        self.population_history = [copy.deepcopy(population)]
        

        best_fitness = -np.inf  # initialize best fitness
        no_improve_count = 0  # counter for early stopping

        # iterate over generations
        for gen in range(1, self.num_generations + 1):
            print(f"\n===== Generation {gen} =====")

            # generate offspring and evaluate their fitness
            offspring = self.generate_and_evaluate_offspring(population, model, X_val, y_val)
            population.replace_weakest(offspring)  # apply elitism and replace weakest individuals

            # record the current population snapshot
            self.population_history.append(copy.deepcopy(population))

            # check for improvement
            current_best = population.get_best_individual().fitness
            improvement = self.calculate_improvement(best_fitness, current_best)

            if improvement > improve_threshold:
                best_fitness = current_best
                no_improve_count = 0
            else:
                no_improve_count += 1

            # stop early if no improvement for given patience
            if no_improve_count >= patience:
                print(f"stopping early at generation {gen} due to no improvement.")
                break

        # return the best individual found, number of generations run, and the population history
        best_individual = self.population_history[-1].get_best_individual()
        return best_individual, gen, self.population_history

    # creates and returns the initial population
    def initialize_population(self):
        population = Population(size=self.pop_size, n_features=self.n_features, elitism=self.elitism)
        self.population_history = [population]
        return population


    # generates offspring, evaluates their fitness, and returns them
    def generate_and_evaluate_offspring(self, population, model, X_val, y_val):
        offspring = population.generate_offspring(
            num_offspring=self.pop_size - self.elitism,
            mutation_rate=self.mutation_rate,
            crossover_prob=self.crossover_prob,
            selection_counts=self.selection_counts
        )

        # evaluate each new offspring individual
        for individual in offspring:
            individual.evaluate_fitness(model, X_val=X_val, y_val=y_val, alpha=self.alpha)

        return offspring

    # calculates relative improvement in fitness
    def calculate_improvement(self, best_fitness, current_best):
        if best_fitness == -np.inf:
            return float('inf')  # assume max improvement for the first generation
        return (current_best - best_fitness) / abs(best_fitness)



