# this file defines configuration constants for training and saving model results

RESULTS_DIR = "/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Project_1/Results"  # this is where all result files will be saved for first part
RESULTS_DIR_BONUS = "/home/marianthi/Desktop/ceid/semester_10/υπολογιστικη/Project_1/bonus_dir/Results"  # this is where all result files will be saved for bonus part
EPOCHS = 100  # this is the maximum number of training epochs
BATCH_SIZE = 32  # this is the batch size used during training ,relative small dataset
PATIENCE = 10  # this is the number of epochs to wait for early stopping
HIDDEN_UNIT_RATIOS = [0.5, 0.66, 1.0, 2.0]  # this is how we scale hidden units relative to input size
LEARNING_RATES = [0.001, 0.05, 0.1]  # this is the list of learning rates for tuning
MOMENTUM_VALUES = [0.2, 0.6]  # this is the list of momentum values for tuning
REGULARIZATION_VALUES = [0.0001, 0.001, 0.01]  # this is the list of regularization values for tuning
NUM_LAYERS = [2, 3]  # this is the list of number of layers for tuning
WEIGHTS = {'accuracy': 0.4,'curve': 0.4,'epochs': 0.2} # weights for composite score