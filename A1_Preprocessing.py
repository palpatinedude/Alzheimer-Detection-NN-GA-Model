import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import zscore


def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def determine_attribute_type(df):
    """
    Identifies attribute types: binary, discrete integer, or numeric.
    """
    types = {}
    for col in df.columns:
        if col == 'Diagnosis':
            continue
        if df[col].dtype == 'int64':
            types[col] = 'Binary (0/1)' if df[col].nunique() == 2 else 'Discrete Integer'
        elif df[col].dtype == 'float64':
            types[col] = 'Numeric'
        else:
            types[col] = 'Unknown'
    return types

def print_feature_ranges(df, types):
    """
    Prints min/max for all numerical or discrete features.
    """
    print("\nFeature Ranges:")
    for col, typ in types.items():
        print(f"{col} ({typ}): Min = {df[col].min()}, Max = {df[col].max()}")

def plot_histograms(df, types):
    """
    Plots histograms to visualize feature distributions.
    """
    print("\nFeature Distributions:")
    for col, typ in types.items():
        if typ in ['Numeric', 'Discrete Integer']:
            plt.figure(figsize=(6, 3))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

# ----------------- STEP 2: OUTLIER DETECTION -----------------

def detect_outliers(df, threshold=3):
    """
    Detects outliers using Z-score.
    A data point is considered an outlier if its Z-score is greater than the threshold.
    """
    outliers = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        z_scores = zscore(df[col].dropna())  # Calculate Z-scores
        outliers[col] = np.where(np.abs(z_scores) > threshold)[0]  # Indices of outliers

    return outliers


# ----------------- STEP 3: SCALING METHODS -----------------

def scale_data(X, method='standard'):
    """
    Scales data using either standardization or normalization.
    """
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), scaler

# ----------------- STEP 4: MODEL EVALUATION -----------------

def evaluate_model(X, y):
    """
    Performs 5-fold stratified cross-validation and computes average metrics.
    """
    model = LogisticRegression(solver='liblinear')
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    metrics = {
        'Accuracy': [], 'Precision': [], 'Recall': [],
        'F1 Score': [], 'ROC AUC': []
    }

    y_true_all, y_pred_all = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)

        metrics['Accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['Precision'].append(precision_score(y_val, y_pred))
        metrics['Recall'].append(recall_score(y_val, y_pred))
        metrics['F1 Score'].append(f1_score(y_val, y_pred))
        metrics['ROC AUC'].append(roc_auc_score(y_val, y_pred))

    print("\nAverage 5-Fold Cross-Validation Metrics:")
    for k in metrics:
        print(f"{k}: {np.mean(metrics[k]):.4f}")

    return metrics, y_true_all, y_pred_all

# ----------------- STEP 5: VISUALIZATION -----------------

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots a confusion matrix from true and predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

