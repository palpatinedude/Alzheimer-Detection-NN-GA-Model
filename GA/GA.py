import numpy as np
import os
import copy
from population import Population
from config import PATIENCE,IMPROVE_THRESHOLD



class GeneticAlgorithm:
    def __init__(self, n_features, num_generations,elitism,best_params_path, weights_path, pop_size=30,  mutation_rate=0.1, crossover_prob=0.6,  alpha=0.05):
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


    def run(self, X_val, y_val, patience=PATIENCE, improve_threshold=IMPROVE_THRESHOLD):
        population = Population(size=self.pop_size, n_features=self.n_features, elitism=self.elitism)
        model = population.evaluate(self.best_params_path, self.weights_path, X_val, y_val, self.alpha)
        self.population_history = [population]

        best_fitness = -np.inf
        no_improve_count = 0

        for gen in range(1, self.num_generations + 1):
            print(f"\n===== Generation {gen} =====")

            offspring = population.generate_offspring(num_offspring=self.pop_size - self.elitism,mutation_rate=self.mutation_rate,crossover_prob=self.crossover_prob,selection_counts=self.selection_counts )

            for ind in offspring:
                ind.evaluate_fitness(model=model,X_val=X_val,y_val=y_val,alpha=self.alpha)

            population.replace_weakest(offspring)
            self.population_history.append(copy.deepcopy(population))

            # Check best fitness this generation
            current_best = population.get_best_individual().fitness

            # Check for improvement
            if best_fitness == -np.inf:
                improvement = float('inf')  # First comparison always improves
            else:
                improvement = (current_best - best_fitness) / abs(best_fitness)

            if improvement > improve_threshold:
                best_fitness = current_best
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"Stopping early at generation {gen} due to no improvement.")
                break

        best_ind = self.population_history[-1].get_best_individual()

        return best_ind, gen,self.population_history  
