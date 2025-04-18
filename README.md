Alzheimer's Disease Prediction
Project Overview

This project aims to predict the likelihood of Alzheimer's disease in patients using machine learning. The model classifies whether a patient is likely to have Alzheimer's disease based on various patient data. The pipeline includes data preprocessing, model training, hyperparameter tuning, evaluation, and visualization of results. The goal is to create an effective predictive model for early detection of Alzheimer's disease.
Key Features

    Data Preprocessing: Cleans the dataset by handling missing values, scaling features, and detecting outliers.

    Modeling: A neural network model is trained to classify Alzheimer’s disease status based on patient data.

    Hyperparameter Tuning: Optimizes the model’s hyperparameters for best performance. This includes:

        Number of Hidden Neurons

        Learning Rate

        Momentum

        Transformation Method (Standardization vs Normalization)

        Regularization Coefficient

    Evaluation: The model is evaluated using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

    Visualization: Visualizes model performance and training progress with confusion matrices, accuracy plots, and loss curves.

Project Structure

.
├── alzheimers_disease_data.csv         # The dataset used for training the model.
├── config.py                           # Configuration file for model parameters.
├── exercise01.py                       # Main script to execute the project pipeline.
├── helpers.py                          # Helper functions for data handling and utilities.
├── modeling/                           # Contains model architecture, training, evaluation, etc.
│   ├── architecture.py                 # Defines the neural network architecture.
│   ├── cross_validation.py             # Implements cross-validation techniques.
│   ├── evaluation.py                   # Contains evaluation metrics and result logging.
│   ├── metrics.py                      # Functions to calculate various performance metrics.
│   ├── training.py                     # Implements model training logic.
│   └── tuning.py                       # Tuning of hyperparameters like learning rate, momentum, etc.
├── preprocessing/                      # Data preprocessing functions.
│   ├── preprocessing.py                # Handles data cleaning, scaling, and outlier detection.
├── reporting/                          # Logs and saves experiment results and metrics.
│   ├── experiments.py                  # Tracks the experiments and configuration.
│   ├── report_writer.py                # Writes results and reports.
│   └── result_saving.py                # Saves model results, metrics, and plots.
├── visualization/                      # Plots and charts for model evaluation and training progress.
│   ├── evalutation_plots.py            # Plots related to model evaluation metrics.
│   ├── plot_base.py                    # Functions for saving and showing plots.
│   └── training_plots.py               # Plots the training and validation performance over epochs.
├── requirements.txt                   # Required dependencies for the project.
└── Results/                            # Directory for storing results and output plots.
    └── Standardization/
        ├── accuracy_plot_fold_1.png
        └── neural_network_results.txt

How It Works
1. Data Preprocessing

    Cleaning: Missing values are handled, and unnecessary columns are removed from the dataset.

    Feature Scaling: Features are scaled using standardization or normalization based on the selected transformation method.

    Outlier Detection: Outliers are detected using Z-scores and addressed accordingly.

2. Model Architecture

    The neural network model is defined in architecture.py, including input, hidden, and output layers for classification.

    The number of hidden neurons is tuned during the hyperparameter optimization phase.

3. Hyperparameter Tuning

    The model’s key parameters (number of hidden neurons, learning rate, momentum, and regularization coefficient) are fine-tuned for optimal performance.

    The transformation method (standardization vs. normalization) is selected based on performance in cross-validation.

4. Cross-Validation

    The model is evaluated using cross-validation to assess performance on different data splits and ensure generalization.

5. Evaluation

    After training, the model is evaluated using various metrics, including accuracy, precision, recall, F1-score, and ROC-AUC.

    Evaluation results are logged and saved for further analysis.

6. Result Visualization

    Model performance is visualized with plots, such as confusion matrices, accuracy plots, and loss curves.

    Training progress is also visualized to assess how well the model is learning over time.

7. Results Reporting

    Final results, including hyperparameter tuning outcomes, performance metrics, and visualizations, are saved in the Results/ directory.

Requirements

To run this project, ensure you have the required dependencies installed. You can install them using the following command:

pip install -r requirements.txt

Running the Project

To execute the project, run the main script:

python exercise01.py

This will trigger the entire pipeline, including preprocessing, training, hyperparameter tuning, evaluation, and result visualization.
Results

After running the project, the results will be saved in the Results/ directory. You will find:

    Accuracy and Loss Plots: Visualizations of accuracy and loss for each fold during cross-validation.

    Evaluation Metrics: A summary of the model’s performance, including accuracy, precision, recall, F1-score, and ROC-AUC.

    Hyperparameter Tuning Results: Insights into the effect of different values of hidden neurons, learning rate, momentum, and regularization coefficient on model performance.
