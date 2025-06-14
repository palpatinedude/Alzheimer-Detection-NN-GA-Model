# this class defines an individual solution for a genetic algorithm used in feature selection
# each individual has a binary chromosome representing selected features, and a fitness score based on pretrained nn performance

import numpy as np

# represents a single individual (solution) in the genetic algorithm
class Individual:
    
    # initializes the individual with a given chromosome or randomly generates one
    def __init__(self, n_features=32, chromosome=None):
        self.n_features = n_features
        if chromosome is not None:
            self.chromosome = np.array(chromosome, dtype=np.int8)  # ensure correct type
        else:
            self.chromosome = self.encode()  # generate a valid chromosome
        self.fitness = None  # fitness will be assigned after evaluation

    # randomly generates a binary chromosome with at least one selected feature
    def encode(self):
        while True:
            chrom = np.random.randint(0, 2, size=self.n_features, dtype=np.int8)
            if np.any(chrom):  # ensure at least one feature is selected
                return chrom

    # masks the input data X by zeroing out unselected features based on the chromosome
    def decode(self, X):
        assert X.shape[1] == self.n_features, \
            f"Expected {self.n_features} features, got {X.shape[1]}"
        X_array = np.asarray(X, dtype=np.float32)
        mask = self.chromosome.astype(bool)  # convert binary chromosome to boolean mask
        X_array[:, ~mask] = 0.0  # zero out columns corresponding to unselected features
        return X_array

    # evaluates the fitness of the individual based on validation loss and a feature penalty
    def evaluate_fitness(self, model, X_val, y_val, alpha=0.05):
        X_masked = self.decode(X_val)  # apply feature selection to input
        
        y_val_reshaped = y_val.reshape(-1, 1)  # reshape target to match model output shape
     
        loss_values = model.evaluate(X_masked, y_val_reshaped, verbose=0)  # get validation loss
        ce_loss = loss_values[0] if loss_values else 1.0  # use default loss if not returned

        num_features = np.sum(self.chromosome)  # number of selected features
        penalty = alpha * (num_features / self.n_features) ** 2  # quadratic penalty for complexity

        self.fitness = 1 / (1 + ce_loss + penalty)  # higher fitness for lower loss & fewer features
        return self.fitness

    # randomly mutates the chromosome bits based on the mutation rate
    def mutate(self, mutation_rate=0.1):
        for i in range(len(self.chromosome)):
            if np.random.rand() < mutation_rate:
                self.chromosome[i] = 1 - self.chromosome[i]  # flip bit (0→1 or 1→0)

        # ensures at least one feature remains selected after mutation
        if np.sum(self.chromosome) == 0:
            self.chromosome[np.random.randint(len(self.chromosome))] = 1

    # performs single-point crossover with another individual and returns two offspring
    def crossover(self, other):
        point = np.random.randint(1, len(self.chromosome))  # pick crossover point
        child1_chrom = np.concatenate([self.chromosome[:point], other.chromosome[point:]])
        child2_chrom = np.concatenate([other.chromosome[:point], self.chromosome[point:]])

        # ensure both children have at least one selected feature
        if not np.any(child1_chrom):
            child1_chrom[np.random.randint(len(child1_chrom))] = 1
        if not np.any(child2_chrom):
            child2_chrom[np.random.randint(len(child2_chrom))] = 1

        return Individual(n_features=self.n_features, chromosome=child1_chrom), \
               Individual(n_features=self.n_features, chromosome=child2_chrom)

    # returns the total number of features selected in the chromosome
    def num_selected_features(self):
        return int(np.sum(self.chromosome))

    # returns a copy of the current individual (used during selection and reproduction)
    def copy(self):
        return Individual(n_features=self.n_features, chromosome=self.chromosome.copy())

    # returns a formatted string showing the chromosome and its fitness
    def __str__(self):
        chrom_list = self.chromosome.tolist()
        if self.fitness is not None:
            return f"Chromosome: {chrom_list}\nFitness: {self.fitness:.4f}"
        else:
            return f"Chromosome: {chrom_list}\nFitness: Not evaluated"
