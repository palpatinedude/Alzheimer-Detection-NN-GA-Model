import os
import sys

# Get the current script directory (GA/)
CURRENT_DIR = os.path.dirname(__file__)

# Add NN and NN/reporting directories to sys.path for imports
NN_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'NN'))
NN_REPORTING_DIR = os.path.join(NN_DIR, 'reporting')
sys.path.append(NN_DIR)
sys.path.append(NN_REPORTING_DIR)

# Paths
data_path = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'alzheimers_disease_data.csv'))
params_path = os.path.abspath(os.path.join(NN_DIR, 'Results', 'best_ann_hyperparameters.json'))
weights_path = os.path.abspath(os.path.join(NN_DIR, 'Results', 'best_ann_model.weights.h5'))