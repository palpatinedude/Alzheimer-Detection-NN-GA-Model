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
BEST_PARAM = os.path.abspath(os.path.join(NN_DIR, 'Results', 'best_ann_hyperparameters.json'))
WEIGHTS = os.path.abspath(os.path.join(NN_DIR, 'Results', 'best_ann_model.weights.h5'))


RESULTS_DIR_GA = "/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Alzheimer-Detection-NN-GA-Model/GA/Results"
if not os.path.exists(RESULTS_DIR_GA):
    os.makedirs(RESULTS_DIR_GA)

# Path where the NN training saved validation data
VAL_DATA_PATH = "/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Alzheimer-Detection-NN-GA-Model/NN/Results/val_data.npz"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


MAX_GENERATIONS = 1000
PATIENCE = 10
IMPROVE_THRESHOLD = 0.01
ELITISM = 1
TOURNAMENT_SIZE = 3
