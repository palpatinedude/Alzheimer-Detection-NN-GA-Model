import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from visualization.evalutation_plots import plot_histograms

# Data loading from csv to pandas dataframe
def load_data(file_path):
    return pd.read_csv(file_path)


# Determines the type of each attribute in the dataset
def determine_attribute_type(df):
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


# Prints the range of each feature in the dataset
def print_feature_ranges(df, types):
    print("\nFeature Ranges:")
    for col, typ in types.items():
        print(f"{col} ({typ}): Min = {df[col].min()}, Max = {df[col].max()}")


# Detects outliers using z-score method where z-score > threshold is considered an outlier
def detect_outliers(df, threshold=3):
    outliers = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        z_scores = zscore(df[col].dropna())  # Calculate Z-scores
        outliers[col] = np.where(np.abs(z_scores) > threshold)[0]  # Indices of outliers

    return outliers



# Scales the data using either standardization or normalization
def scale_data(X, method='standard'):
    """
    Scales data using either standardization or normalization.
    """
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), scaler


# Main function to inspect the data
def inspect_data(file_path,):
    df = load_data(file_path)
    df.drop(columns=['PatientID', 'DoctorInCharge'], inplace=True, errors='ignore')

    types = determine_attribute_type(df)
    print_feature_ranges(df, types)

    # For every type except binary features--> plot histograms 
  #  for col, typ in types.items():
  #      plot_histograms(df[[col]], {col: typ})


    outliers = detect_outliers(df)
    print("\nOutliers detected at indices:")
    for col, indices in outliers.items():
        print(f"{col}: {indices}")

    X, y = df.drop(columns='Diagnosis'), df['Diagnosis']
   
    return X,y






'''
# ----------------- STEP 4: MODEL EVALUATION -----------------

def evaluate_model_A1(X, y):
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

'''

