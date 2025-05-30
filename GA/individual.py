import numpy as np
import os
import json
import warnings
from modeling.architecture import create_model_wrapper

class Individual:
    """
    Represents an individual in a genetic algorithm for feature selection.
    Chromosome: binary vector encoding presence(1)/absence(0) of features.
    Fitness: based on NN validation loss + penalty for feature count.
    """

    def __init__(self, n_features=32, chromosome=None):
        self.n_features = n_features
        if chromosome is not None:
            self.chromosome = np.array(chromosome, dtype=np.int8)
        else:
            self.chromosome = self.encode()
        self.fitness = None

    def encode(self):
        """Randomly generate chromosome with at least one feature selected."""
        while True:
            chrom = np.random.randint(0, 2, size=self.n_features, dtype=np.int8)
            if np.any(chrom):
                return chrom

    def decode(self, X):
        """
        Mask input X based on chromosome.
        Zero-out unselected features, return float32 array.
        """
        assert X.shape[1] == self.n_features, \
            f"Expected {self.n_features} features, got {X.shape[1]}"
        X_array = np.asarray(X, dtype=np.float32)
        mask = self.chromosome.astype(bool)
        X_array[:, ~mask] = 0.0
        return X_array

    def evaluate_fitness(self, X, y, best_params_path=None, weights_path=None, alpha=0.05):
        """
        Evaluate fitness: combine cross-entropy loss and penalty for feature count.
        Uses pretrained NN weights and masks input features.
        """
        # Load hyperparameters
        if best_params_path and os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            hidden_units = best_params.get("hidden_units", 64)
            learning_rate = best_params.get("learning_rate", 0.001)
            momentum = best_params.get("momentum", 0.0)
            regularization = best_params.get("regularization_lambda", 0.0)
        else:
            hidden_units = 64
            learning_rate = 0.001
            momentum = 0.0
            regularization = 0.0
            print("Warning: Using default hyperparameters.")

        # Create model
        model = create_model_wrapper(
            'ann',
            input_dim=self.n_features,
            hidden_units=hidden_units,
            learning_rate=learning_rate,
            momentum=momentum,
            regularization=regularization,
            simple_metrics=True
        )

        # Load pretrained weights if available
        if weights_path and os.path.exists(weights_path):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Skipping variable loading for optimizer.*")
                    model.load_weights(weights_path)
            except Exception as e:
                print(f"Warning: Failed to load weights: {e}")
        else:
            print(f"Warning: Weights not found at '{weights_path}', using untrained model.")

        # Mask input features
        X_masked = self.decode(X)

        # Evaluate model
        loss_values = model.evaluate(X_masked, y, verbose=0)
        ce_loss = loss_values[0] if loss_values else 1.0  # cross-entropy loss

        # Penalty based on feature count
        penalty = alpha * (np.sum(self.chromosome) / self.n_features)

        # Fitness (higher is better)
        self.fitness = 1 / (1 + ce_loss + penalty)
        return self.fitness

    def mutate(self, mutation_rate=0.1):
        """
        Flip bits randomly with probability mutation_rate.
        Ensure at least one feature is selected after mutation.
        """
        for i in range(len(self.chromosome)):
            if np.random.rand() < mutation_rate:
                self.chromosome[i] = 1 - self.chromosome[i]

        # Ensure at least one selected feature
        if np.sum(self.chromosome) == 0:
            self.chromosome[np.random.randint(len(self.chromosome))] = 1

    def crossover(self, other):
        """
        Single-point crossover with another individual.
        Returns two offspring.
        """
        point = np.random.randint(1, len(self.chromosome))
        child1_chrom = np.concatenate([self.chromosome[:point], other.chromosome[point:]])
        child2_chrom = np.concatenate([other.chromosome[:point], self.chromosome[point:]])

        if not np.any(child1_chrom):
            child1_chrom[np.random.randint(len(child1_chrom))] = 1
        if not np.any(child2_chrom):
            child2_chrom[np.random.randint(len(child2_chrom))] = 1

        return Individual(n_features=self.n_features, chromosome=child1_chrom), \
               Individual(n_features=self.n_features, chromosome=child2_chrom)

    def num_selected_features(self):
        return int(np.sum(self.chromosome))

    def copy(self):
        return Individual(n_features=self.n_features, chromosome=self.chromosome.copy())

    def __str__(self):
        chrom_list = self.chromosome.tolist()
        if self.fitness is not None:
            return f"Chromosome: {chrom_list}\nFitness: {self.fitness:.4f}"
        else:
            return f"Chromosome: {chrom_list}\nFitness: Not evaluated"