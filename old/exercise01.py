'''
THEORETICAL NOTES:
a.Metrics :
  Accuracy : It represents the proportion of correct predictions (both true positives and true negatives) out of the total number of predictions.
  Precision : It represents the proportion of true positive predictions out of the total positive predictions made.
  Recall : It represents the proportion of true positive predictions out of the actual positives in the dataset.
  F1 Score : It is the harmonic mean of precision and recall, providing a balance between the two metrics.
  ROC AUC : It represents the area under ROC curve, which is a plot of ability to discriminate between the positive and negative classes at various thresholds.




'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import zscore

# ----------------- STEP 1: LOAD & INSPECT DATA -----------------

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

# ----------------- MAIN FUNCTION -----------------

def main(file_path):
    # Load and clean data
    df = load_data(file_path)
    df.drop(columns=['PatientID', 'DoctorInCharge'], inplace=True, errors='ignore')

    # Identify feature types
    attribute_types = determine_attribute_type(df)
    print_feature_ranges(df, attribute_types)
    plot_histograms(df, attribute_types)

    # Detect outliers using Z-score
    outliers = detect_outliers(df, threshold=3)
    
    # Print outliers information
    print("\nOutlier Indices (Z-score > 3):")
    for col, indices in outliers.items():
        print(f"{col}: {len(indices)} outliers detected")
    

    # Split features and labels
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']

    # 80% train, 20% hold-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # ---------- Standardization ----------
    X_train_std, std_scaler = scale_data(X_train, method='standard')
    X_test_std = pd.DataFrame(std_scaler.transform(X_test), columns=X_test.columns)

    print("\n🔹 Evaluating Model with Standardized Data:")
    metrics_std, y_true_std, y_pred_std = evaluate_model(X_train_std, y_train)
    plot_confusion_matrix(y_true_std, y_pred_std, title="Standardized Data (Validation)")

    # ---------- Normalization ----------
    X_train_norm, norm_scaler = scale_data(X_train, method='minmax')
    X_test_norm = pd.DataFrame(norm_scaler.transform(X_test), columns=X_test.columns)

    print("\n🔹 Evaluating Model with Normalized Data:")
    metrics_norm, y_true_norm, y_pred_norm = evaluate_model(X_train_norm, y_train)
    plot_confusion_matrix(y_true_norm, y_pred_norm, title="Normalized Data (Validation)")

    # Hold-out test set is reserved for final evaluation
    print("\n Final test set (20%) is reserved and untouched for future model testing.")

# ----------------- RUN -----------------

if __name__ == "__main__":
    main("../alzheimers_disease_data.csv")
