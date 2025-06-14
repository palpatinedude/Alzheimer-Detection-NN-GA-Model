# 🧠 Alzheimer's Disease Prediction with Neural Networks and Genetic Algorithms

This project aims to predict the likelihood of Alzheimer's disease in patients using machine learning, specifically a Neural Network (NN), and optimize its input features using a Genetic Algorithm (GA). It is divided into two main parts:

- Part A: Train and optimize a neural network model for Alzheimer's prediction.
- Part B: Use a genetic algorithm for feature selection based on the fixed, pre-trained NN from Part A.

The goal is to build a robust model for early Alzheimer's detection, reduce overfitting, and improve generalization by selecting the most relevant features.

---

# Key Features

Part A – Neural Network-Based Prediction

- Data Preprocessing
  Handles missing values, scaling (standardization or normalization), and outlier detection.

- Model Training & Architecture
  - Feedforward neural network with tunable hyperparameters.
  - Modular implementation for experimentation.
  - Note: Future versions will transition to an object-oriented approach for better modularity and extensibility.

- Hyperparameter Tuning
  - Hidden layer size
  - Learning rate
  - Momentum
  - Regularization
  - Data transformation method

- Cross-Validation & Evaluation
  - K-fold cross-validation
  - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

- Visualization & Reporting
  Training/validation curves, confusion matrices, hyperparameter convergence plots.

---

Part B – Genetic Algorithm for Feature Selection

Goal:
Select the most informative subset of 34 features to reduce dimensionality while preserving (or improving) predictive performance.

Key Components:
- Encoding: Each individual is a binary vector (length 34), where 1 = keep feature, 0 = drop.
- Population Initialization: Randomly generate individuals representing different feature subsets.

Fitness Function:
- Based on validation performance (cross-entropy loss or accuracy) of the fixed NN from Part A.
- Penalizes solutions using too many features to balance model simplicity and performance.

Genetic Operators:
- Selection: Tournament, rank-based, or roulette wheel.
- Crossover: Single-point, multi-point, uniform crossover.
- Mutation & Elitism: Introduce diversity and preserve best solutions.

Evaluation:
- Tested with multiple configurations of population size, crossover/mutation probabilities.
- Termination criteria:
  - No improvement over N generations
  - Less than 1% change in best fitness
  - Max generations reached

---

# Project Structure

```plaintext
Alzheimer-Detection-NN-GA-Model/
├── alzheimers_disease_data.csv
├── GA
│   ├── config.py
│   ├── exercise02.py
│   ├── experiment.py
│   ├── ga.py
│   ├── individual.py
│   ├── plotting.py
│   ├── population.py
│   ├── Project_ΥΝ_2024-25_Μέρος-Β.pdf
│   ├── reporting.py
│   ├── Results
│   │   ├── best_ga_model_summary.txt
│   │   ├── best_set.txt
│   │   ├── selected_features.txt
│   │   ├── SET1/
│   │   ├── SET2/
│   │   ├── SET3/
│   │   ├── SET4/
│   │   ├── SET5/
│   │   ├── SET6/
│   │   ├── SET7/
│   │   ├── SET8/
│   │   └── SET9/
│   │   ├── SET10/
│   ├── results_comparison
│   ├── sometests
│   │   ├── individualTest.py
│   │   └── populationTest.py
│   └── Εργαστηριακή_Άσκηση_Μέρος_Β_.pdf
├── main.py
├── NN
│   ├── bonus_dir
│   │   ├── cross_validate.py
│   │   ├── evaluation.py
│   │   ├── model.py
│   │   ├── Results
│   │   ├── save.py
│   │   ├── tuning_hidden.py
│   │   └── visualize.py
│   ├── bonus.py
│   ├── config.py
│   ├── exercise01.py
│   ├── helpers.py
│   ├── modeling
│   │   ├── architecture.py
│   │   ├── cross_validation.py
│   │   ├── evaluation.py
│   │   ├── metrics.py
│   │   ├── training.py
│   │   └── tuning.py
│   ├── old.py
│   ├── preprocessing
│   │   ├── preprocessing.py
│   ├── Project_ΥΝ_2024-25_Μέρος-Α.pdf
│   ├── reporting
│   │   ├── experiments.py
│   │   ├── report_writer.py
│   │   └── result_saving.py
│   ├── Results
│   │   ├── A1/
│   │   ├── A2/
│   │   ├── A3/
│   │   ├── best_ann_hyperparameters.json
│   │   ├── best_ann_model.keras
│   │   ├── best_ann_model_summary.txt
│   │   ├── best_ann_model.weights.h5
│   │   ├── test_data.npz
│   │   ├── val_data.npz
│   │   └── Α4
│   ├── visualization
│   │   ├── evalutation_plots.py
│   │   ├── plot_base.py
│   │   └── training_plots.py
│   └── Εργαστηριακή_Άσκηση_Μέρος_Α_.pdf
├── plot.py

├── README.md
├── requirements.txt
├── results_comparison/
├── retrain.py
└── save.py  
```

---

# How It Works

Part A – Neural Network

- Preprocessing
  Cleans and scales data, handles outliers.

- Model Definition
  Fully connected NN defined in architecture.py.

- Hyperparameter Optimization
  Explored via grid/random search; results saved in Results/A2/ and A3/.

- Cross-Validation & Evaluation
  Evaluates generalization performance using k-fold cross-validation.

- Visualization & Logging
  Accuracy/loss curves and performance summaries saved under Results/.

- Future Work:
  Refactor NN pipeline using object-oriented programming for better structure and flexibility.

---

Part B – Genetic Algorithm

- Feature Encoding
  Individuals are binary masks of features (length = 34).

- Fitness Evaluation
  Applies fixed NN weights and computes validation accuracy or loss plus a feature penalty.

- GA Workflow:
  1. Initialize population
  2. Evaluate fitness
  3. Select, crossover, mutate
  4. Track best individuals


- Convergence Curves & Plots
  Average fitness vs. generation plots for convergence analysis.

- Final Evaluation
  Compare NN (GA-selected features) vs. full NN on:
  - Accuracy
  - Generalization
  - Overfitting
  - Retraining effect using the entire dataset

---

# Running the Project

# Install dependencies
pip install -r requirements.txt

# Run NN pipeline (Part A)
python3 NN/exercise01.py

# Run optimized NN (bonus)
python3 NN/bonus.py

# Run GA for feature selection (Part B)
python3 GA/exercise02.py

# Run entire pipeline  both the Neural Network training/optimization and Genetic Algorithm feature selection in one workflow
python3 main.py 

# Run retrains the neural network model after GA has selected the optimal subset of features and the NN hyperparameters have been optimized.
python3 retrain.py
---

# Notes

- The NN uses weights obtained from full training (no retraining per GA individual) to speed up GA execution.
- All results, including performance metrics, convergence plots, and tuning logs, are saved under the appropriate Results/ subdirectories.
- Code is modular and organized by functionality for easy modification and testing.

---

# To Do (Future Improvements)

- Transition entire codebase, especially NN pipeline, to an Object-Oriented Programming (OOP) structure.


---





