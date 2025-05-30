from individual import Individual
from config import params_path, weights_path, data_path
from preprocessing.preprocessing import inspect_data

def main():
    # Load and inspect data
    X, y = inspect_data(data_path)

    # Initialize population
    population = [Individual() for _ in range(10)]

    # Evaluate fitness of each individual in the population
    for i, ind in enumerate(population):
        fitness = ind.evaluate_fitness(X, y, best_params_path=params_path, weights_path=weights_path)
        print(f"Individual {i}: Fitness = {fitness:.4f}")

    # Create and evaluate two parents
    parent1 = Individual()
    parent2 = Individual()
    print("Parent 1:", parent1)
    print("Parent 2:", parent2)

    # Perform crossover to create children
    child1, child2 = parent1.crossover(parent2)
    print("Child 1 before mutation:", child1)
    print("Child 2 before mutation:", child2)

    # Mutate children
    mutation_rate = 0.2
    child1.mutate(mutation_rate=mutation_rate)
    child2.mutate(mutation_rate=mutation_rate)
    print("Child 1 after mutation:", child1)
    print("Child 2 after mutation:", child2)

    # Demonstrate decoding and masking of data using an individual
    ind = Individual()
    print("Original chromosome:", ind.chromosome)
    X_masked = ind.decode(X)
    print("Original X shape:", X.shape)
    print("Masked X shape:", X_masked.shape)
    print("Sum of masked features per sample:", X_masked.sum(axis=1)[:5])  # Check masking effect for first 5 samples

if __name__ == "__main__":
    main()


