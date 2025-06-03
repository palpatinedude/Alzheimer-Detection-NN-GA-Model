import numpy as np
from GA import GeneticAlgorithm
from config import VAL_DATA_PATH, params_path, weights_path, RESULTS_DIR_GA

# Load validation data
val_data = np.load(VAL_DATA_PATH)
X_val = val_data["X_val"]
y_val = val_data["y_val"]

# Run genetic algorithm
ga = GeneticAlgorithm(
    n_features=X_val.shape[1],
    pop_size=30,
    num_generations=15,
    elitism=1,
    mutation_rate=0.1,
    tournament_size=3,
    alpha=0.05,
    best_params_path=params_path,
    weights_path=weights_path,
    results_dir=RESULTS_DIR_GA,
    plots=True
)

best_individual = ga.run(X_val=X_val, y_val=y_val)
