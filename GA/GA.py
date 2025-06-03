import numpy as np
import os
import copy
from population import Population
from reporting import save_generation_report, save_best_individual_report
from plotting import plot_fitness_over_generations,plot_feature_counts_over_generations,plot_tournament_selection_pie
from config import PATIENCE,IMPROVE_THRESHOLD



class GeneticAlgorithm:
    def __init__(self, n_features, num_generations,elitism,tournament_size,best_params_path, weights_path, results_dir, pop_size=30,  mutation_rate=0.1, crossover_prob=0.6,  alpha=0.05,  plots=True):
        self.n_features = n_features
        self.num_generations = num_generations
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.best_params_path = best_params_path
        self.weights_path = weights_path
        self.results_dir = results_dir
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.crossover_prob = crossover_prob 
        self.population_history = []
        self.selection_counts = [0] * pop_size
        self.plots = plots


        os.makedirs(self.results_dir, exist_ok=True)



    def run(self, X_val, y_val, patience=30, improve_threshold=IMPROVE_THRESHOLD):
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

        # Save reports
        save_generation_report(self.population_history, os.path.join(self.results_dir, "generation_report.csv"))
        save_best_individual_report(best_ind, os.path.join(self.results_dir, "best_individual.txt"))

        # Plot
        num_generations = len(self.population_history)
        if self.plots:
            plot_fitness_over_generations(num_generations, self.population_history,
                                          save_path=os.path.join(self.results_dir, "fitness_plot.png"))

            plot_feature_counts_over_generations(num_generations, self.population_history,
                                                 save_path=os.path.join(self.results_dir, "feature_counts_plot.png"))

            plot_tournament_selection_pie(self.selection_counts)

        return best_ind, gen  # Also return generations run    
    '''
    def run(self, X_val, y_val):
        # Initialize population
        population = Population(size=self.pop_size, n_features=self.n_features, elitism=self.elitism)

        # Evaluate initial population and get model
        model = population.evaluate(self.best_params_path, self.weights_path, X_val, y_val, self.alpha)
        self.population_history.append(population)

        for gen in range(1, self.num_generations + 1):
            print(f"\n===== Generation {gen} =====")

            # Generate offspring
            offspring = population.generate_offspring(
                num_offspring=self.pop_size - self.elitism,
                mutation_rate=self.mutation_rate
            )

            # Evaluate offspring
            for ind in offspring:
                ind.evaluate_fitness(
                model=model,
                X_val=X_val,
                y_val=y_val,
                alpha=self.alpha
            )

            # Replace weakest individuals
            population.replace_weakest(offspring)
            #self.population_history.append(population)
            self.population_history.append(copy.deepcopy(population))

            # Track tournament selection stats
            population.select_parents(
                k=2,
                tournament_size=self.tournament_size,
                selection_counts=self.selection_counts
            )

        # Final best individual
        best_ind = self.population_history[-1].get_best_individual()

        # Save reports
        save_generation_report(self.population_history, os.path.join(self.results_dir, "generation_report.csv"))
        save_best_individual_report(best_ind, os.path.join(self.results_dir, "best_individual.txt"))

        # Plot
        if self.plots:
            plot_fitness_over_generations(self.num_generations + 1, self.population_history,
                                          save_path=os.path.join(self.results_dir, "fitness_plot.png"))

            plot_feature_counts_over_generations(self.num_generations + 1, self.population_history,
                                                 save_path=os.path.join(self.results_dir, "feature_counts_plot.png"))

            plot_tournament_selection_pie(self.selection_counts)

        print("\n===== Genetic Algorithm Completed =====")
        print("Best Individual:")
        print(best_ind)
        return best_ind
        '''