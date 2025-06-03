import numpy as np
import os
import config
from config import  params_path, weights_path , VAL_DATA_PATH, RESULTS_DIR_GA
from population import Population

# Load validation data saved by the NN training script
val_data = np.load(VAL_DATA_PATH)
X_val = val_data["X_val"]
y_val = val_data["y_val"]


# Set random seed for reproducibility
np.random.seed(42)

# Create population with number of features from validation data
pop_size = 30
population = Population(size=pop_size, n_features=X_val.shape[1], elitism=1)

# Evaluate initial population
print("\nEvaluating initial population fitness:")
population.evaluate(
    best_params_path=params_path,
    weights_path=weights_path,
    X_val=X_val,
    y_val=y_val,
    alpha=0.05
)

# Print fitness of initial population
for i, ind in enumerate(population.individuals):
    print(f"Individual {i+1} fitness: {ind.fitness:.4f}, features selected: {ind.num_selected_features()}")

# Select parents for reproduction
parents = population.select_parents(k=2, tournament_size=3)
print("\nSelected parents (chromosomes):")
for i, parent in enumerate(parents):
    print(f"Parent {i+1}: {parent.chromosome}")

# Generate offspring with mutation
offspring = population.generate_offspring(num_offspring=2, mutation_rate=0.1)
print("\nGenerated offspring (chromosomes):")
for i, child in enumerate(offspring):
    print(f"Child {i+1}: {child.chromosome}")

# Replace weakest individuals with offspring
population.replace_weakest(offspring)

# Evaluate population after replacement
print("\nEvaluating population after replacement:")
population.evaluate(
    best_params_path=params_path,
    weights_path=weights_path,
    X_val=X_val,
    y_val=y_val,
    alpha=0.05
)

# Print final population fitness
for i, ind in enumerate(population.individuals):
    print(f"Individual {i+1} fitness: {ind.fitness:.4f}, features selected: {ind.num_selected_features()}")

# Print best individual
best = population.get_best_individual()
print("\nBest Individual:")
print(best)