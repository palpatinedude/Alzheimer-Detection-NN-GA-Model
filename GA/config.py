# this script sets up directory paths, imports, and constants for running the genetic algorithm (ga) module
# it prepares access to neural network components, data files, and result directories

import os
import sys

# get the current script directory (GA/)
CURRENT_DIR = os.path.dirname(__file__)

# add NN and NN/reporting directories to sys.path for module imports
NN_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'NN'))
NN_REPORTING_DIR = os.path.join(NN_DIR, 'reporting')
sys.path.append(NN_DIR)
sys.path.append(NN_REPORTING_DIR)

# define important data and model paths
DATA = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'alzheimers_disease_data.csv'))  # dataset path
BEST_PARAM = os.path.abspath(os.path.join(NN_DIR, 'Results', 'best_ann_hyperparameters.json'))  # best hyperparameters
WEIGHTS = os.path.abspath(os.path.join(NN_DIR, 'Results', 'best_ann_model.weights.h5'))  # best trained weights
MODEL = os.path.abspath(os.path.join(NN_DIR, 'Results', 'best_ann_model.keras'))  # full model file


# define path to save ga results
RESULTS_DIR_GA = "/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Alzheimer-Detection-NN-GA-Model/GA/Results"
if not os.path.exists(RESULTS_DIR_GA):
    os.makedirs(RESULTS_DIR_GA)  # create results directory if it doesn't exist

# define path to validation data saved by the nn training process
VAL_DATA_PATH = "/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Alzheimer-Detection-NN-GA-Model/NN/Results/val_data.npz"

# define path to test data for evaluation
TEST_DATA_PATH = "/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Alzheimer-Detection-NN-GA-Model/NN/Results/test_data.npz"

# disable gpu usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ga configuration constants
MAX_GENERATIONS = 1000  # maximum number of generations to run
PATIENCE = 10  # early stopping patience
IMPROVE_THRESHOLD = 0.01  # threshold for improvement to reset patience
ELITISM = 1  # number of top individuals to carry over
TOURNAMENT_SIZE = 3  # number of individuals in tournament selection
