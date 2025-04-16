'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Function to handle constant columns and apply transformations
def evaluate_model_with_transforms(df):
    # 1. Load and inspect data
    print("Dataset Info:")
    print(df.info())

    # Check for constant columns (zero variance columns)
    constant_columns = df.columns[df.nunique() == 1]
    print(f"\nConstant Columns (Zero Variance): {list(constant_columns)}")
    
    # Remove constant columns from the features
    df = df.drop(columns=constant_columns)

    print(f"\nDataset Info after removing constant columns:")
    print(df.info())

    # 2. Preprocess data: Remove non-numeric columns (if any)
    df = df.select_dtypes(include=[np.number])  # Keep only numerical columns

    # 3. Split data into features and target
    X = df.drop('Diagnosis', axis=1)  # Assuming 'Diagnosis' is the target column
    y = df['Diagnosis']

    # 4. Apply transformations and check performance

    # 4.1 Apply Standardization
    print("\nApplying Standardization to the data...")
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Display the first few rows of the standardized data
    print("\nStandardized Data (First 5 rows):")
    print(pd.DataFrame(X_standardized, columns=X.columns).head())

    # Check for NaN or infinite values after standardization
    if np.any(np.isnan(X_standardized)) or np.any(np.isinf(X_standardized)):
        print("Warning: NaN or Infinite values found after Standardization")
        X_standardized = np.nan_to_num(X_standardized)  # Replace NaN/Inf with 0

    # Perform cross-validation on standardized data
    print("\nCross-validation performance for Standardization:")
    evaluate_model(X_standardized, y)

    # 4.2 Apply Normalization
    print("\nApplying Normalization to the data...")
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Display the first few rows of the normalized data
    print("\nNormalized Data (First 5 rows):")
    print(pd.DataFrame(X_normalized, columns=X.columns).head())

    # Check for NaN or infinite values after normalization
    if np.any(np.isnan(X_normalized)) or np.any(np.isinf(X_normalized)):
        print("Warning: NaN or Infinite values found after Normalization")
        X_normalized = np.nan_to_num(X_normalized)  # Replace NaN/Inf with 0

    # Perform cross-validation on normalized data
    print("\nCross-validation performance for Normalization:")
    evaluate_model(X_normalized, y)

# Function to evaluate model performance using cross-validation
def evaluate_model(X, y):
    # Define a logistic regression model
    model = LogisticRegression(solver='liblinear')

    # Define the metrics for evaluation
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score)
    }

    # 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for metric in scoring:
        cv_results = cross_val_score(model, X, y, cv=cv, scoring=scoring[metric])
        results[metric] = cv_results

    # Display cross-validation results
    print("Cross-validation results:")
    for metric, values in results.items():
        print(f"{metric.capitalize()} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")

    # Finally, print the results of the model's performance
    print("\nFinal Model Performance (Mean of 5-fold CV):")
    for metric, values in results.items():
        print(f"{metric.capitalize()}: {np.mean(values):.4f}")

# Main entry point
def main():
    # Load your dataset
    # Example: df = pd.read_csv('your_dataset.csv')
    # In your case, replace this with the path to your CSV or other data source
    file_path = "../alzheimers_disease_data.csv"  # Replace with your actual path
    df = pd.read_csv(file_path)

    # Evaluate model with transformations
    evaluate_model_with_transforms(df)

if __name__ == "__main__":
    main()
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import zscore


# ----------------------- HELPER FUNCTIONS -----------------------

# Function to detect outliers using Z-score
def detect_outliers(df, numerical_cols, threshold=3):
    """Detect outliers using Z-scores. Returns a dictionary of outliers."""
    outliers = {}
    for col in numerical_cols:
        z_scores = zscore(df[col])  # Calculate z-scores
        outliers[col] = np.where(np.abs(z_scores) > threshold)  # Store indices of outliers
    return outliers


# Function to remove outliers based on Z-scores
def remove_outliers(df, outliers):
    """Remove rows with outliers based on the indices."""
    for col, indices in outliers.items():
        df = df.drop(indices[0], axis=0)  # Drop rows corresponding to outliers
    return df

# ----------------------- DATA PREPROCESSING -----------------------

# Function to preprocess data
def preprocess_data(df):
    """Preprocess data: one-hot encode categorical columns, 
    and standardize/normalize numerical columns."""
    
    # Drop irrelevant columns
    df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')

    # Identify column types
    categorical_cols = ['Ethnicity', 'EducationLevel']
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col != 'Diagnosis']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + binary_cols + ['Diagnosis']]

    print(f"\nCategorical Columns (One-Hot Encoding): {categorical_cols}")
    print(f"Binary Columns (No Transformation): {binary_cols}")
    print(f"Numerical Columns (Standardization/Normalization): {numerical_cols}")

    # One-Hot Encoding for categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Standardization and Normalization
    df_standardized = standardize_data(df, numerical_cols)
    df_normalized = normalize_data(df, numerical_cols)

    return df, df_standardized, df_normalized, numerical_cols, binary_cols

# Function to standardize numerical columns
def standardize_data(df, numerical_cols):
    """Standardize numerical columns using StandardScaler."""
    scaler = StandardScaler()
    df_standardized = df.copy()
    df_standardized[numerical_cols] = scaler.fit_transform(df_standardized[numerical_cols])
    return df_standardized

# Function to normalize numerical columns
def normalize_data(df, numerical_cols):
    """Normalize numerical columns using MinMaxScaler."""
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[numerical_cols] = scaler.fit_transform(df_normalized[numerical_cols])
    return df_normalized

# ----------------------- MODEL TRAINING & EVALUATION -----------------------

# Function to evaluate the model using cross-validation
def evaluate_model(X, y):
    """Evaluate model performance using cross-validation."""
    model = LogisticRegression(solver='liblinear')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validation predictions
    y_pred = cross_val_predict(model, X, y, cv=cv)

    # Calculate and display performance metrics
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred),
        "ROC AUC": roc_auc_score(y, y_pred)
    }
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# ----------------------- MAIN EXECUTION -----------------------

def main():
    """Main function to load, preprocess data, and evaluate the model."""
    
    # Load dataset
    file_path = "../alzheimers_disease_data.csv"  # Update path if needed
    df = pd.read_csv(file_path)

    # Preprocess the data
    df, df_standardized, df_normalized, numerical_cols, binary_cols = preprocess_data(df)

    # Check and visualize outliers before preprocessing
    outliers = detect_outliers(df, numerical_cols)
    print("\nDetecting outliers using Z-scores:")
    print(outliers)

    # Remove outliers based on detected Z-scores
    df_no_outliers = remove_outliers(df, outliers)

    # Reapply preprocessing steps after removing outliers
    df_standardized_no_outliers = standardize_data(df_no_outliers, numerical_cols)
    df_normalized_no_outliers = normalize_data(df_no_outliers, numerical_cols)

    # Define features (X) and target (y)
    X_standardized = df_standardized_no_outliers.drop(columns=['Diagnosis'])
    X_normalized = df_normalized_no_outliers.drop(columns=['Diagnosis'])
    y = df_no_outliers['Diagnosis']

    # Evaluate models with and without outliers
    print("\nEvaluating model with Standardized Data (No Outliers):")
    evaluate_model(X_standardized, y)

    print("\nEvaluating model with Normalized Data (No Outliers):")
    evaluate_model(X_normalized, y)

if __name__ == "__main__":
    main()
