import config
import numpy as np
import random
from individual import Individual
import json
from modeling.architecture import create_model_wrapper
import warnings
import os

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

    def load_model(self, best_params_path, weights_path):
        """
        Loads best hyperparameters, creates the model, loads weights.
        Returns the compiled model ready for evaluation.
        """
        # Load hyperparameters from JSON
        if best_params_path and os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            hidden_units = best_params.get("hidden_units", 64)
            learning_rate = best_params.get("learning_rate", 0.001)
            momentum = best_params.get("momentum", 0.0)
            regularization = best_params.get("regularization_lambda", 0.0)
        else:
            # Default params if JSON missing
            hidden_units = 64
            learning_rate = 0.001
            momentum = 0.0
            regularization = 0.0
            print("Warning: Using default hyperparameters.")

        # Create model
        model = create_model_wrapper(
            model_type='ann',
            input_dim=self.n_features,
            hidden_units=hidden_units,
            learning_rate=learning_rate,
            momentum=momentum,
            regularization=regularization,
            simple_metrics=True
        )

        # Load weights
        if weights_path and os.path.exists(weights_path):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Skipping variable loading for optimizer.*")
                    model.load_weights(weights_path)
            except Exception as e:
                print(f"Warning: Failed to load weights: {e}")
        else:
            print(f"Warning: Weights not found at '{weights_path}', using untrained model.")


        return model

    def evaluate(self, best_params_path=None, weights_path=None, X_val=None, y_val=None, alpha=0.05):

        if X_val is None or y_val is None:
         raise ValueError("Validation data must be provided for evaluation.")

        # Load pretrained model once
        model = self.load_model(best_params_path, weights_path)

        # Evaluate each individual
        for i, individual in enumerate(self.individuals):
            individual.evaluate_fitness(model, X_val, y_val, alpha)

        return model    



    def get_best_individual(self):
        return max(self.individuals, key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf)

    def sort_by_fitness(self, reverse=True):
        self.individuals.sort(key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf, reverse=reverse)

    def select_parents(self, k=2, tournament_size=3, selection_counts=None):
        parents = []
        for _ in range(k):
            tournament = random.sample(self.individuals, tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness if ind.fitness is not None else -np.inf)
            parents.append(winner.copy())

            # Track winner's selection index
            if selection_counts is not None:
                winner_idx = self.individuals.index(winner)
                selection_counts[winner_idx] += 1

        return parents


    def generate_offspring(self, num_offspring, mutation_rate=0.1, crossover_prob=0.6, selection_counts=None):
        offspring = []
        while len(offspring) < num_offspring:
            parent1, parent2 = self.select_parents(2, selection_counts=selection_counts)
            if np.random.rand() < crossover_prob:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
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
