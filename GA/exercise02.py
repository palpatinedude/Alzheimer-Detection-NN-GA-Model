from config import data_path, params_path, weights_path
import pandas as pd
import numpy as np
from preprocessing.preprocessing import inspect_data
from individual import Individual


# Load and inspect data
X, y = inspect_data(data_path)


# Create and evaluate individual using best model weights and hyperparams
ind = Individual()
fitness = ind.evaluate_fitness(X, y, best_params_path=params_path, weights_path=weights_path)

# Print fitness or individual details
print(ind)
