# 🧠 Alzheimer's Disease Prediction

This project aims to predict the likelihood of Alzheimer's disease in patients using machine learning. The model classifies whether a patient is likely to have Alzheimer's disease based on various patient data. The pipeline includes data preprocessing, model training, hyperparameter tuning, evaluation, and visualization of results — with the goal of building an effective predictive model for early detection of Alzheimer’s disease.

---

## 🔍 Key Features

- **Data Preprocessing**  
  - Cleans the dataset by handling missing values, scaling features, and detecting outliers.

- **Modeling**  
  - A neural network model is trained to classify Alzheimer’s disease status based on patient data.

- **Hyperparameter Tuning**  
  - Number of Hidden Neurons  
  - Learning Rate  
  - Momentum  
  - Transformation Method (Standardization vs Normalization)  
  - Regularization Coefficient  

- **Evaluation**  
  - Metrics include Accuracy, Precision, Recall, F1-score, and ROC-AUC.

- **Visualization**  
  - Confusion matrices, training/validation accuracy and loss curves.

---

## 📁 Project Structure

. ├── alzheimers_disease_data.csv # The dataset used for training the model. ├── config.py # Configuration file for model parameters. ├── exercise01.py # Main script to execute the project pipeline. ├── helpers.py # Helper functions for data handling and utilities. ├── modeling/ # Contains model architecture, training, evaluation, etc. │ ├── architecture.py # Defines the neural network architecture. │ ├── cross_validation.py # Implements cross-validation techniques. │ ├── evaluation.py # Contains evaluation metrics and result logging. │ ├── metrics.py # Functions to calculate various performance metrics. │ ├── training.py # Implements model training logic. │ └── tuning.py # Tuning of hyperparameters like learning rate, momentum, etc. ├── preprocessing/ # Data preprocessing functions. │ ├── preprocessing.py # Handles data cleaning, scaling, and outlier detection. ├── reporting/ # Logs and saves experiment results and metrics. │ ├── experiments.py # Tracks the experiments and configuration. │ ├── report_writer.py # Writes results and reports. │ └── result_saving.py # Saves model results, metrics, and plots. ├── visualization/ # Plots and charts for model evaluation and training progress. │ ├── evalutation_plots.py # Plots related to model evaluation metrics. │ ├── plot_base.py # Functions for saving and showing plots. │ └── training_plots.py # Plots the training and validation performance over epochs. ├── requirements.txt # Required dependencies for the project. └── Results/ # Directory for storing results and output plots. └── Standardization/ ├── accuracy_plot_fold_1.png └── neural_network_results.txt


---

## ⚙️ How It Works

### 1. Data Preprocessing

- **Cleaning**: Handles missing values and removes unnecessary columns.  
- **Feature Scaling**: Applies either standardization or normalization depending on the selected method.  
- **Outlier Detection**: Uses Z-scores to detect and optionally handle outliers.

### 2. Model Architecture

- Defined in `architecture.py` with input, hidden, and output layers.
- The number of hidden neurons is a tunable hyperparameter.

### 3. Hyperparameter Tuning

- Optimizes the model for:
  - Number of hidden neurons
  - Learning rate
  - Momentum
  - Regularization coefficient
  - Transformation method

### 4. Cross-Validation

- Evaluates model generalization across multiple data splits.

### 5. Evaluation

- Uses Accuracy, Precision, Recall, F1-Score, and ROC-AUC to evaluate performance.
- Results are saved and logged for review.

### 6. Result Visualization

- Training progress and evaluation are visualized using:
  - Accuracy and loss plots
  - Confusion matrices
  - ROC curves (if implemented)

### 7. Results Reporting

- Final performance metrics, plots, and tuning logs are stored in the `Results/` directory.

---

## 📦 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt

▶️ Running the Project

To execute the entire pipeline (preprocessing, training, evaluation, and visualization), run:

python exercise01.py

📊 Output & Results

After execution, the Results/ directory will contain:

    Accuracy and loss plots for each fold during cross-validation.

    Final evaluation metrics summary.

    Hyperparameter tuning logs and model configuration.

    Plots for model training and evaluation performance.
