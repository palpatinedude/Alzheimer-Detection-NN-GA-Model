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

. ├── alzheimers_disease_data.csv # The dataset used for training the model 
├── config.py # Configuration file for model parameters 
├── exercise01.py # Main script to execute the project pipeline 
├── helpers.py # Helper functions for data handling and utilities

├── modeling/ # Model-related components
│ ├── architecture.py # Defines the neural network architecture 
│ ├── cross_validation.py # Implements cross-validation techniques 
│ ├── evaluation.py # Evaluation metrics and result logging
│ ├── metrics.py # Functions to calculate various performance metrics
│ ├── training.py # Implements model training logic 
│ └── tuning.py # Hyperparameter tuning logic

├── preprocessing/ # Data preprocessing components 
│ └── preprocessing.py # Data cleaning, scaling, and outlier detection

├── reporting/ # Experiment logging and reporting 
│ ├── experiments.py # Tracks experiments and configuration
│ ├── report_writer.py # Writes experiment reports
│ └── result_saving.py # Saves results, metrics, and plots

├── visualization/ # Plotting and visualization scripts 
│ ├── evalutation_plots.py # Plots for evaluation metrics
│ ├── plot_base.py # Helper functions for displaying/saving plots
│ └── training_plots.py # Visualizes training and validation performance

├── requirements.txt # List of required dependencies

└── Results/ # Output results and visualizations └── Standardization/ ├── accuracy_plot_fold_1.png └── neural_network_results.txt


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

python3 exercise01.py


