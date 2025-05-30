import os
import sys

# Get the current script directory (GA/)
CURRENT_DIR = os.path.dirname(__file__)

# Add NN and NN/reporting directories to sys.path for imports
NN_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'NN'))
NN_REPORTING_DIR = os.path.join(NN_DIR, 'reporting')
sys.path.append(NN_DIR)
sys.path.append(NN_REPORTING_DIR)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.preprocessing import inspect_data
from individual import Individual

# Paths
data_path = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'alzheimers_disease_data.csv'))
params_path = os.path.abspath(os.path.join(NN_DIR, 'Results', 'best_ann_hyperparameters.json'))
weights_path = os.path.abspath(os.path.join(NN_DIR, 'Results', 'best_ann_model.weights.h5'))

# Load and inspect data
X, y = inspect_data(data_path)

# Split data into training and test sets (keeping 20% for final evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Create and evaluate individual using best model weights and hyperparams
ind = Individual()
fitness = ind.evaluate_fitness(X, y, best_params_path=params_path, weights_path=weights_path)

# Print fitness or individual details
print(ind)
