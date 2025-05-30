🧠 Alzheimer's Disease Prediction with Neural Networks and Genetic Algorithms

This project aims to predict the likelihood of Alzheimer's disease in patients using machine learning, specifically a Neural Network (NN), and optimize its input features using a Genetic Algorithm (GA). It is divided into two main parts:

    Part A: Train and optimize a neural network model for Alzheimer's prediction.
    Part B: Use a genetic algorithm for feature selection based on the fixed, pre-trained NN from Part A.

The goal is to build a robust model for early Alzheimer's detection, reduce overfitting, and improve generalization by selecting the most relevant features.

🔍 Key Features

✅ Part A – Neural Network-Based Prediction

    Data Preprocessing
        - Handles missing values, scaling (standardization or normalization), and outlier detection.

    Model Training & Architecture
        - A feedforward neural network with tunable hyperparameters.
        - Implemented in a modular fashion for experimentation.
        - NOTE: Future versions will transition this to an object-oriented approach for better modularity and extensibility.

    Hyperparameter Tuning
        - Hidden layer size
        - Learning rate
        - Momentum
        - Regularization
        - Data transformation method

    Cross-Validation & Evaluation
        - K-fold cross-validation
        - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

    Visualization & Reporting
        - Training/validation curves, confusion matrices, hyperparameter convergence plots

🧬 Part B – Genetic Algorithm for Feature Selection

Goal:
    Select the most informative subset of 34 features to reduce dimensionality while preserving (or improving) predictive performance.

Key Components:
    - Encoding: Each individual is a binary vector (length 34), where 1 = keep feature, 0 = drop.
    - Population Initialization: Randomly generate individuals representing different feature subsets.

    Fitness Function:
        - Based on the validation performance (cross-entropy loss or accuracy) of the fixed NN from Part A.
        - Penalizes solutions using too many features to balance model simplicity and performance.

    Selection Methods: Tournament selection, rank-based, or roulette wheel.
    Crossover Strategies: Single-point, multi-point, and uniform crossover.
    Mutation & Elitism: Introduces diversity and preserves best solutions.

Evaluation:
    - Tested over multiple configurations of population size, crossover/mutation probabilities.
    - Termination Criteria:
        • No improvement over N generations
        • <1% change in best fitness
        • Max generations reached

📁 Project Structure

Alzheimer-Detection-NN-GA-Model/
├── alzheimers_disease_data.csv
├── GA/
│   ├── config.py
│   ├── exercise02.py
│   ├── individual*.py
│   ├── population*.py
│   └── Project_ΥΝ_2024-25_Μέρος-Β.pdf
├── NN/
│   ├── exercise01.py
│   ├── bonus.py
│   ├── config.py
│   ├── helpers.py
│   ├── Project_ΥΝ_2024-25_Μέρος-Α.pdf
│   ├── bonus_dir/
│   │   ├── cross_validate.py
│   │   ├── model.py
│   │   ├── save.py
│   │   ├── visualize.py
│   │   └── Results/
│   ├── modeling/
│   │   ├── architecture.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   ├── tuning.py
│   │   └── etc.
│   ├── preprocessing/
│   │   └── preprocessing.py
│   ├── reporting/
│   │   ├── experiments.py
│   │   ├── report_writer.py
│   │   └── result_saving.py
│   └── Results/
│       ├── A2/
│       └── A3/
├── requirements.txt


⚙️ How It Works

🧠 Part A – Neural Network

    Preprocessing
        - Cleans and scales data, handles outliers.

    Model Definition
        - Fully connected NN defined in architecture.py.

    Hyperparameter Optimization
        - Explored via grid/random search, results saved in Results/A2/ and A3/.

    Cross-Validation & Evaluation
        - Evaluates generalization performance using k-fold CV.

    Visualization & Logging
        - Accuracy/loss curves and performance summaries saved under Results/.

    ⚠️ Future Work:
        - Refactor NN pipeline using object-oriented programming principles for better structure and flexibility.

🧬 Part B – Genetic Algorithm

    Feature Encoding
        - Individuals are binary masks of features (length = 34).

    Fitness Evaluation
        - Applies fixed NN weights and computes validation accuracy or loss + feature penalty.

    GA Workflow
        - Initialize population
        - Evaluate fitness
        - Select, crossover, mutate
        - Track best individuals

    Convergence Curves & Plots
        - Average fitness vs. generation plots used for convergence analysis.

    Final Evaluation
        - Compare NN (GA-selected features) vs. full NN:
            • Accuracy
            • Generalization
            • Overfitting
            • Retraining effect using the entire dataset

▶️ Running the Project

Install dependencies:

    pip install -r requirements.txt

Run NN pipeline (Part A):

    python3 NN/exercise01.py

Run optimized NN (bonus):

    python3 NN/bonus.py

Run GA for feature selection (Part B):

    python3 GA/exercise02.py

📝 Notes

- The NN uses weights obtained from full training (no retraining per GA individual) to speed up GA execution.
- All results, including performance metrics, convergence plots, and tuning logs, are saved under the appropriate Results/ subdirectories.
- Code is modular and organized by functionality for easy modification and testing.

🛠️ To Do (Future Improvements)

- Transition entire codebase, especially NN pipeline, to an **Object-Oriented Programming (OOP)** structure.



