import numpy as np
import config
from config import data_path, params_path, weights_path
from preprocessing.preprocessing import inspect_data
from population import Population

# --- Load real data ---
X, y = inspect_data(data_path)

# --- Optional: set random seed for reproducibility ---
np.random.seed(42)

# --- Create population ---
pop_size = 30
population = Population(size=pop_size, n_features=X.shape[1], elitism=1)

# --- Evaluate population ---
print("Evaluating initial population fitness:")
population.evaluate(X, y, best_params_path=params_path, weights_path=weights_path, alpha=0.05)

# Print initial population fitness
for i, ind in enumerate(population.individuals):
    print(f"Individual {i+1} fitness: {ind.fitness:.4f}, features selected: {ind.num_selected_features()}")

# --- Select parents ---
parents = population.select_parents(k=2, tournament_size=3)
print("\nSelected parents (chromosomes):")
for i, parent in enumerate(parents):
    print(f"Parent {i+1}: {parent.chromosome}")

# --- Generate offspring ---
offspring = population.generate_offspring(num_offspring=4, mutation_rate=0.1)
print("\nGenerated offspring (chromosomes):")
for i, child in enumerate(offspring):
    print(f"Child {i+1}: {child.chromosome}")

# --- Replace weakest in population with offspring ---
population.replace_weakest(offspring)

# --- Evaluate new population ---
print("\nEvaluating population after replacement:")
population.evaluate(X, y, best_params_path=params_path, weights_path=weights_path, alpha=0.05)

# Print final population fitness
for i, ind in enumerate(population.individuals):
    print(f"Individual {i+1} fitness: {ind.fitness:.4f}, features selected: {ind.num_selected_features()}")

# --- Print best individual ---
best = population.get_best_individual()
print("\nBest Individual:")
print(best)
