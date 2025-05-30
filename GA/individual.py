import numpy as np
import os
import json
from modeling.architecture import create_model_wrapper

class Individual:
    """
    Represents an individual in a genetic algorithm for feature selection.
    The chromosome is a binary vector encoding selected (1) or unselected (0) features.
    Fitness is evaluated based on model performance and feature usage penalty.
    """

    def __init__(self, n_features=32, chromosome=None):
        """
        Initialize individual with a chromosome representing feature selection.
        If chromosome not provided, generate a random valid one.
        """
        self.n_features = n_features
        if chromosome is not None:
            self.chromosome = chromosome
        else:
            self.chromosome = self.encode()
        self.fitness = None

    def encode(self):
        """
        Generate a random chromosome (list of 0/1) ensuring at least one feature is selected.
        """
        while True:
            chromosome = np.random.randint(0, 2, size=self.n_features)
            if np.any(chromosome):  # Ensure at least one feature is included
                return chromosome.tolist()

    def decode(self, X):
        """
        Apply zero-masking to input data X based on chromosome.
        Features not selected (chromosome=0) are zeroed out, but all features are retained.
        
        Returns a copy of X with masked features.
        """
        mask = np.array(self.chromosome, dtype=bool)  # Convert chromosome to boolean mask
        X_masked = X.copy()

        if hasattr(X_masked, 'loc'):  # Handle pandas DataFrame
            X_masked.loc[:, ~mask] = 0  # Zero out unselected features
        else:  # Handle numpy arrays
            X_masked[:, ~mask] = 0

        return X_masked

    def evaluate_fitness(self, X, y, best_params_path=None, weights_path=None, alpha=0.05):
        """
        Evaluate fitness of the individual using:
        - Cross-entropy loss from the pretrained model on masked input.
        - A penalty proportional to the fraction of selected features.
        
        Returns the fitness score as: 1 - cross_entropy_loss - alpha * (num_features_used / total_features).
        """
        # Load best hyperparameters if path provided; otherwise use defaults
        if best_params_path and os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            hidden_units = best_params.get("hidden_units", 64)
            learning_rate = best_params.get("learning_rate", 0.001)
            momentum = best_params.get("momentum", 0.0)
            regularization = best_params.get("regularization_lambda", 0.0)
        else:
            print("Warning: best_params_path not found or not provided, using default hyperparameters.")
            hidden_units = 64
            learning_rate = 0.001
            momentum = 0.0
            regularization = 0.0

        # Create the ANN model with given hyperparameters
        model = create_model_wrapper(
            'ann',
            input_dim=self.n_features,
            hidden_units=hidden_units,
            learning_rate=learning_rate,
            momentum=momentum,
            regularization=regularization
        )

        # Load pretrained weights if available
        if weights_path and os.path.exists(weights_path):
            model.load_weights(weights_path)
        else:
            print(f"Warning: pretrained weights not found at '{weights_path}', using untrained model.")

        # Mask input features according to chromosome
        X_masked = self.decode(X)

        # Evaluate model on masked data to get loss and metrics
        loss_values = model.evaluate(X_masked, y, verbose=0)

        # Extract cross-entropy loss from evaluation results (fallback to 1.0 if missing)
        ce_loss = dict(zip(model.metrics_names, loss_values)).get('ce_loss', 1.0)

        # Calculate penalty based on fraction of features selected
        mask = np.array(self.chromosome, dtype=bool)
        penalty = alpha * (np.sum(mask) / self.n_features)

        # Fitness combines accuracy and penalty; higher is better
        self.fitness = 1 - ce_loss - penalty

        return self.fitness

    def mutate(self, mutation_rate=0.1):
        """
        Apply mutation by randomly flipping bits in the chromosome based on mutation_rate.
        Ensure at least one feature remains selected after mutation.
        """
        for i in range(len(self.chromosome)):
            if np.random.rand() < mutation_rate:
                self.chromosome[i] = 1 - self.chromosome[i]  # Flip bit

        # Make sure chromosome has at least one selected feature
        if np.sum(self.chromosome) == 0:
            self.chromosome[np.random.randint(len(self.chromosome))] = 1

    def crossover(self, other):
        """
        Perform one-point crossover with another individual.
        Returns two offspring individuals.
        """
        point = np.random.randint(1, len(self.chromosome))  # Crossover point
        child1_chrom = self.chromosome[:point] + other.chromosome[point:]
        child2_chrom = other.chromosome[:point] + self.chromosome[point:]
        return Individual(chromosome=child1_chrom), Individual(chromosome=child2_chrom)

    def __str__(self):
        """
        String representation showing chromosome and fitness.
        """
        return f"Chromosome: {self.chromosome}\nFitness: {self.fitness:.4f}" if self.fitness is not None else f"Chromosome: {self.chromosome}\nFitness: Not evaluated"
