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

. ├── alzheimers_disease_data.csv # The dataset used for training the model <br>
├── config.py  <br>
├── exercise01.py<br>
├── helpers.py <br>
|── bonus_dir <br>
| ├── cross_validate.py <br>
| ├── model.py <br>
| ├── Results/ <br>
| ├── save.py <br>
| ├── tuning_hidden.py <br>
| ├── visualize.py <br>
├── modeling/ <br>
│ ├── architecture.py  <br>
│ ├── cross_validation.py  <br>
│ ├── evaluation.py<br>
│ ├── metrics.py metrics<br>
│ ├── training.py <br>
│ └── tuning.py <br>

├── preprocessing/  <br>
│ └── preprocessing.py<br>

├── reporting/  <br>
│ ├── experiments.py <br>
│ ├── report_writer.py <br>
│ └── result_saving.py <br>

├── visualization/  <br>
│ ├── evalutation_plots.py<br>
│ ├── plot_base.py <br>
│ └── training_plots.py <br>

├── requirements.txt<br>

|── Results/ <br>
│ ├── Standardization/<br>
│ ├── Normalization/<br>




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


