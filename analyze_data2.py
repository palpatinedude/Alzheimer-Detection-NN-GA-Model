import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import zscore

# ----------------------- HELPER FUNCTIONS -----------------------

# Function to load the dataset
def load_data(file_path):
    """Loads the dataset into a pandas DataFrame"""
    return pd.read_csv(file_path)

# Function to determine the type of attribute for each column
def determine_attribute_type(df):
    """Determines the attribute type for each column in the DataFrame"""
    attribute_types = {}

    for column in df.columns:
        # Check if the column is of type int64 (Integer columns)
        if df[column].dtype == 'int64':
            # If it contains only 2 unique values (binary 0/1)
            if df[column].nunique() == 2:
                attribute_types[column] = 'Binary (0/1)'
            # If it contains more than 2 unique values (Discrete Integer)
            else:
                attribute_types[column] = 'Discrete Integer'
        # Check if the column is of type object (Categorical columns)
        elif df[column].dtype == 'object':
            attribute_types[column] = 'Categorical'
        # Check if the column is of type float64 (Numeric columns)
        elif df[column].dtype == 'float64':
            attribute_types[column] = 'Numeric'
        else:
            attribute_types[column] = 'Unknown'

    return attribute_types

# Function to print the min and max values for Discrete Integer and Numeric attributes
def print_min_max_values(df, attribute_types):
    """Prints the min and max values for Discrete Integer and Numeric columns"""
    for column, attr_type in attribute_types.items():
        print(f"{column}: {attr_type}")
        # If the column is Discrete Integer or Numeric, print min and max values
        if attr_type in ['Discrete Integer', 'Numeric']:
            min_val = df[column].min()
            max_val = df[column].max()
            print(f"  Min: {min_val}, Max: {max_val}")

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

def main(file_path):
    """Main function to load, preprocess data, and evaluate the model."""
    
    # Load dataset
    df = load_data(file_path)

    # Determine the attribute types
    attribute_types = determine_attribute_type(df)
    
    # Print attribute types and their min/max values for Discrete Integer and Numeric attributes
    print("Attribute Types:")
    for column, attr_type in attribute_types.items():
        print(f"{column}: {attr_type}")
    
    print("\nMin and Max Values for Discrete Integer and Numeric attributes:")
    print_min_max_values(df, attribute_types)

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
    # Update the path with your dataset location
    main("../alzheimers_disease_data.csv")
