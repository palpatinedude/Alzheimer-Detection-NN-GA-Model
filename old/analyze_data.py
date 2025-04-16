
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../alzheimers_disease_data.csv")  

# General Information
print("General Information:")
print(df.info())
print("\nDataset Shape:", df.shape)

# Check for Missing Values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
print("\nMissing Values:")
print(missing_data[missing_data["Missing Values"] > 0])  # Show only columns with missing values

# Summary Statistics (Numerical Data)
print("\nSummary Statistics (Numerical Features):")
print(df.describe())

# Identify Categorical & Numerical Features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
print("\nFeature Types:")
print(f"Categorical Features: {categorical_cols}")
print(f"Numerical Features: {numerical_cols}")

# Outlier Detection (Boxplots)
print("\nChecking for Outliers:")
plt.figure(figsize=(12, 6))
df[numerical_cols].boxplot(rot=90)  # Rotate labels for better readability
plt.title("Outlier Detection using Boxplots")
plt.show()

# Distribution of Numerical Features
print("\nFeature Distributions:")
df[numerical_cols].hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.show()

# Preprocessing Suggestions
preprocessing_steps = {}

for col in df.columns:
    if col in numerical_cols:
        if df[col].isnull().sum() > 0:
            preprocessing_steps[col] = "Impute missing values (Mean/Median)"
        
        # Check for skewness (suggest transformation)
        if abs(df[col].skew()) > 1:
            preprocessing_steps[col] = preprocessing_steps.get(col, "") + " + Apply log/sqrt transformation"

        # Outlier detection (using IQR)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        if outlier_condition.sum() > 0:
            preprocessing_steps[col] = preprocessing_steps.get(col, "") + " + Handle outliers (e.g., winsorization, capping)"

    elif col in categorical_cols:
        if df[col].isnull().sum() > 0:
            preprocessing_steps[col] = "Impute missing values (Most Frequent)"
        
        if df[col].nunique() < 10:  # Low cardinality
            preprocessing_steps[col] = preprocessing_steps.get(col, "") + " + One-Hot Encoding"
        else:
            preprocessing_steps[col] = preprocessing_steps.get(col, "") + " + Label Encoding"

print("\nSuggested Preprocessing Steps:")
for col, steps in preprocessing_steps.items():
    print(f"{col}: {steps}")

'''


'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("../alzheimers_disease_data.csv")

# General information about the dataset
print("General Information:")
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
print("\nMissing Values:")
print(missing_data[missing_data["Missing Values"] > 0])  # Only columns with missing values

# Summary statistics for numerical columns
print("\nSummary Statistics (Numerical Features):")
print(df.describe())

# Check for skewness in numerical features
print("\nSkewness of Numerical Features:")
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
skewness = df[numerical_cols].skew()
print(skewness)

# Check for correlations between numerical features
correlation_matrix = df[numerical_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize correlations (heatmap)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Visualize distributions of numerical features (for outliers and skewness)
print("\nFeature Distributions:")
df[numerical_cols].hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.show()

# Visualize skewed distributions (for specific analysis)
print("\nSkewed Distributions:")
for col in numerical_cols:
    if abs(df[col].skew()) > 1:  # Highly skewed
        sns.histplot(df[col], kde=True)
        plt.title(f"Skewed Distribution of {col}")
        plt.show()

# Categorical features check
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("\nCategorical Features:")
print(categorical_cols)
for col in categorical_cols:
    print(f"{col}: {df[col].value_counts()}")

# Check cardinality of categorical features
print("\nCardinality of Categorical Features:")
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")
'''
# RESULTS:
# Numerical attributes : Everyting except DoctorInCharge 
# No missing values 
# Variables like FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, and others have positive skewness, indicating that majority of patients tend to have low values.
# Features such as Smoking, HeadInjury, and Hypertension show high skewness values, suggesting that they might have a long tail in their distributions.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file path)
data = pd.read_csv('alzheimers_disease_data.csv')

# 1. Data Overview
print("Data Overview:")
print(data.info())  # Basic information about the dataset (data types, non-null count)

# Check for missing values in each column
missing_values = data.isnull().sum()
print("\nMissing Values in Each Feature:")
print(missing_values)

# Percentage of missing data per column
missing_percentage = (missing_values / len(data)) * 100
print("\nPercentage of Missing Data in Each Feature:")
print(missing_percentage)

# 2. Numerical Feature Analysis
# Identifying numerical columns
numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()

print("\nNumerical Features:", numerical_features)

# Check summary statistics for numerical features
numerical_summary = data[numerical_features].describe()
print("\nSummary Statistics for Numerical Features:")
print(numerical_summary)

# Visualize distributions of numerical features
for feature in numerical_features:
    print(f"\nVisualizing Distribution of {feature}:")
    plt.figure(figsize=(6, 4))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Calculate skewness and kurtosis for numerical features to decide on the scaling method
skewness = data[numerical_features].apply(lambda x: skew(x.dropna()))
kurtosis_values = data[numerical_features].apply(lambda x: kurtosis(x.dropna()))

print("\nSkewness of Numerical Features:")
print(skewness)

print("\nKurtosis of Numerical Features:")
print(kurtosis_values)

# 3. Categorical Feature Analysis
# Identifying categorical columns
categorical_features = data.select_dtypes(include=[object]).columns.tolist()

print("\nCategorical Features:", categorical_features)

# Check the number of unique values in each categorical feature
unique_categories = data[categorical_features].nunique()
print("\nNumber of Unique Categories in Each Categorical Feature:")
print(unique_categories)

# Visualize the distribution of categorical features
for feature in categorical_features:
    print(f"\nVisualizing Distribution of Categorical Feature {feature}:")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=data[feature])
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)
    plt.show()


ordinal_features = ['EducationLevel', 'Diagnosis']  # Example: You should modify based on your dataset
print("\nOrdinal Features:", ordinal_features)

# Check if ordinal features follow the expected order
for feature in ordinal_features:
    print(f"\nUnique values in {feature}: {data[feature].unique()}")

# 5. Correlation Analysis (For Numerical Features)
correlation_matrix = data[numerical_features].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Heatmap of correlations to visually inspect relationships
print("\nGenerating Heatmap of Correlation Matrix:")
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Check if there are any features with high correlation (> 0.8)
high_correlation = correlation_matrix[correlation_matrix > 0.8].stack().index.tolist()
print("\nHighly Correlated Features (correlation > 0.8):")
print(high_correlation)
'''


''' LINEAR REGRESSION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ====== FUNCTION DEFINITIONS ======

def explore_data(df):
    """Displays basic information about the dataset."""
    print("Dataset Info:\n", df.info())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nSample Data:\n", df.head())

def clean_data(df):
    """Handles missing values and removes unnecessary columns."""
    df = df.dropna()  # Drop rows with missing values
    df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')  # Remove irrelevant columns
    return df

def encode_data(df):
    """One-hot encodes categorical variables."""
    categorical_cols = ['Gender', 'Ethnicity', 'Smoking', 'AlcoholConsumption', 'FamilyHistoryAlzheimers',
                        'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # One-hot encoding
    return df

def apply_standardization(X):
    """Applies Standardization (zero mean, unit variance)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def apply_normalization(X):
    """Applies Min-Max Normalization (scales between 0 and 1)."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def apply_centering(X):
    """Centers data by subtracting the mean."""
    return X - np.mean(X, axis=0)

def evaluate_preprocessing(X_processed, y, method_name):
    """Performs 5-fold cross-validation using Logistic Regression."""
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifier = LogisticRegression(max_iter=4000, random_state=42)

    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'AUC': []}

    for train_idx, val_idx in kfold.split(X_processed, y):
        if isinstance(X_processed, pd.DataFrame):
            # Pandas DataFrame indexing (use iloc)
            X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
        else:
            # NumPy array indexing (integer-based indexing)
            X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        y_prob = classifier.predict_proba(X_val)[:, 1]
    
        metrics['Accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['Precision'].append(precision_score(y_val, y_pred))
        metrics['Recall'].append(recall_score(y_val, y_pred))
        metrics['F1 Score'].append(f1_score(y_val, y_pred))
        metrics['AUC'].append(roc_auc_score(y_val, y_prob))

    # Print the average results
    print(f"\nPerformance for {method_name}:")
    for metric, values in metrics.items():
        print(f"{metric}: {np.mean(values):.4f}")

def plot_linear_regression(X_processed, y, method_name):
    """Fits and plots Linear Regression for the first principal component with the regression line."""
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Print the linear regression results for debugging
    print(f"\nLinear Regression Results for {method_name}:")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"R-squared: {model.score(X_test, y_test):.4f}")

    # Scatter plot of actual vs predicted values
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
    
    # Add the regression line
    plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red', linewidth=2, label='Regression Line')

    plt.xlabel("Actual Diagnosis")
    plt.ylabel("Predicted Diagnosis")
    plt.title(f"Linear Regression - {method_name}")
    plt.legend()
    plt.show()

# ====== MAIN EXECUTION FLOW ======

# Load dataset
df = pd.read_csv("../alzheimers_disease_data.csv")

# Data Exploration
explore_data(df)

# Data Cleaning
df = clean_data(df)

# Encoding Categorical Data
df = encode_data(df)

# Separate Features & Target
X = df.drop(columns=['Diagnosis'])  # Features
y = df['Diagnosis']  # Target

# Convert X to a NumPy array (if it isn't already)
X = X.values if isinstance(X, pd.DataFrame) else X

# Apply Preprocessing Techniques
X_standardized = apply_standardization(X)
X_normalized = apply_normalization(X)
X_centered = apply_centering(X)

# Evaluate Different Preprocessing Methods
evaluate_preprocessing(X_standardized, y, "Standardization")
evaluate_preprocessing(X_normalized, y, "Normalization")
evaluate_preprocessing(X_centered, y, "Centering")

# Visualize Linear Regression Analysis
plot_linear_regression(X_standardized, y, "Standardization")
plot_linear_regression(X_normalized, y, "Normalization")
plot_linear_regression(X_centered, y, "Centering")
'''
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# ====== FUNCTION DEFINITIONS ======

def explore_data(df):
    """Displays basic information about the dataset."""
    print("Dataset Info:\n", df.info())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nSample Data:\n", df.head())

def clean_data(df):
    """Handles missing values and removes unnecessary columns."""
    df = df.dropna()  # Drop rows with missing values
    df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')  # Remove irrelevant columns
    return df

def encode_data(df):
    """One-hot encodes categorical variables."""
    categorical_cols = ['Gender', 'Ethnicity', 'Smoking', 'AlcoholConsumption', 'FamilyHistoryAlzheimers',
                        'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # One-hot encoding
    return df

def apply_standardization(X):
    """Applies Standardization (zero mean, unit variance)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def apply_normalization(X):
    """Applies Min-Max Normalization (scales between 0 and 1)."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def apply_centering(X):
    """Centers data by subtracting the mean."""
    return X - np.mean(X, axis=0)

def evaluate_preprocessing(X_processed, y, method_name):
    """Performs 5-fold cross-validation using Logistic Regression."""
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifier = LogisticRegression(max_iter=4000, random_state=42)

    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'AUC': []}

    for train_idx, val_idx in kfold.split(X_processed, y):
        if isinstance(X_processed, pd.DataFrame):
            # Pandas DataFrame indexing (use iloc)
            X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
        else:
            # NumPy array indexing (integer-based indexing)
            X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        y_prob = classifier.predict_proba(X_val)[:, 1]
    
        metrics['Accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['Precision'].append(precision_score(y_val, y_pred))
        metrics['Recall'].append(recall_score(y_val, y_pred))
        metrics['F1 Score'].append(f1_score(y_val, y_pred))
        metrics['AUC'].append(roc_auc_score(y_val, y_prob))

    # Print the average results
    print(f"\nPerformance for {method_name}:")
    for metric, values in metrics.items():
        print(f"{metric}: {np.mean(values):.4f}")

def plot_logistic_regression(X_processed, y, method_name):
    """Fits and plots Logistic Regression, displaying the ROC curve."""
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=4000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Print the logistic regression results for debugging
    print(f"\nLogistic Regression Results for {method_name}:")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # ROC Curve
    plt.figure(figsize=(8, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='blue', label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Logistic Regression ({method_name})')
    plt.legend(loc='lower right')
    plt.show()

# ====== MAIN EXECUTION FLOW ======

# Load dataset
df = pd.read_csv("../alzheimers_disease_data.csv")

# Data Exploration
explore_data(df)

# Data Cleaning
df = clean_data(df)

# Encoding Categorical Data
df = encode_data(df)

# Separate Features & Target
X = df.drop(columns=['Diagnosis'])  # Features
y = df['Diagnosis']  # Target

# Convert X to a NumPy array (if it isn't already)
X = X.values if isinstance(X, pd.DataFrame) else X

# Apply Preprocessing Techniques
X_standardized = apply_standardization(X)
X_normalized = apply_normalization(X)
X_centered = apply_centering(X)

# Evaluate Different Preprocessing Methods using Logistic Regression
evaluate_preprocessing(X_standardized, y, "Standardization")
evaluate_preprocessing(X_normalized, y, "Normalization")
evaluate_preprocessing(X_centered, y, "Centering")

# Visualize Logistic Regression Analysis (ROC Curve)
plot_logistic_regression(X_standardized, y, "Standardization")
plot_logistic_regression(X_normalized, y, "Normalization")
plot_logistic_regression(X_centered, y, "Centering")
'''


'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# ====== FUNCTION DEFINITIONS ======

def explore_data(df):
    """Explore and print dataset information."""
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSample Data:")
    print(df.head())

def clean_data(df):
    """Clean data by removing unnecessary columns and rows with missing values."""
    df = df.dropna()
    df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')  # Drop non-useful columns
    return df

def apply_preprocessing(X, method):
    """Apply different preprocessing methods."""
    if method == 'Standardization':
        scaler = StandardScaler()
    elif method == 'Normalization':
        scaler = MinMaxScaler()
    elif method == 'Centering':
        return X - np.mean(X, axis=0)  # Centering simply subtracts the mean
    
    return scaler.fit_transform(X)

def evaluate_and_plot_model(X_processed, y, method_name):
    """Evaluate the model performance with cross-validation and plot the ROC curve."""
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=4000, random_state=42)
    
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'AUC': []}
    y_true_all, y_pred_all, y_prob_all = [], [], []  # Collect predictions and true values for ROC curve
    
    for train_idx, val_idx in kfold.split(X_processed, y):
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # Collect metrics
        metrics['Accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['Precision'].append(precision_score(y_val, y_pred))
        metrics['Recall'].append(recall_score(y_val, y_pred))
        metrics['F1 Score'].append(f1_score(y_val, y_pred))
        metrics['AUC'].append(roc_auc_score(y_val, y_prob))
        
        # Collect data for the ROC curve
        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    # Print cross-validation performance
    print(f"\nPerformance for {method_name}:")
    for metric, values in metrics.items():
        print(f"{metric}: {np.mean(values):.4f}")
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    auc = roc_auc_score(y_true_all, y_prob_all)
    
    # Plot ROC curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {method_name}")
    plt.legend()
    plt.show()

# ====== MAIN EXECUTION ======

# Load dataset
df = pd.read_csv("../alzheimers_disease_data.csv")

# Explore and clean data
explore_data(df)
df = clean_data(df)

# Features and target variable
X = df.drop(columns=['Diagnosis']).values
y = df['Diagnosis']

# List of preprocessing methods
preprocessing_methods = ['Standardization', 'Normalization', 'Centering']

# Evaluate performance and plot ROC curves for each preprocessing method
for method in preprocessing_methods:
    X_processed = apply_preprocessing(X, method)
    evaluate_and_plot_model(X_processed, y, method)
'''

'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

# ====== FUNCTION DEFINITIONS ======

def explore_data(df):
    """Explore and print dataset information."""
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSample Data:")
    print(df.head())

def clean_data(df):
    """Clean data by removing unnecessary columns and rows with missing values."""
    df = df.dropna()
    df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')  # Drop non-useful columns
    return df

def apply_preprocessing(X, method):
    """Apply different preprocessing methods."""
    if method == 'Standardization':
        scaler = StandardScaler()
    elif method == 'Normalization':
        scaler = MinMaxScaler()
    elif method == 'Centering':
        return X - np.mean(X, axis=0)  # Centering simply subtracts the mean
    
    return scaler.fit_transform(X)

def evaluate_and_plot_model(X_processed, y, method_name):
    """Evaluate the model performance with cross-validation and plot the ROC curve."""
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=4000, random_state=42)
    
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'AUC': []}
    y_true_all, y_pred_all, y_prob_all = [], [], []  # Collect predictions and true values for ROC curve
    
    for train_idx, val_idx in kfold.split(X_processed, y):
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # Collect metrics
        metrics['Accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['Precision'].append(precision_score(y_val, y_pred))
        metrics['Recall'].append(recall_score(y_val, y_pred))
        metrics['F1 Score'].append(f1_score(y_val, y_pred))
        metrics['AUC'].append(roc_auc_score(y_val, y_prob))
        
        # Collect data for the ROC curve
        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    # Print cross-validation performance
    print(f"\nPerformance for {method_name}:")
    for metric, values in metrics.items():
        print(f"{metric}: {np.mean(values):.4f}")
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    auc = roc_auc_score(y_true_all, y_prob_all)
    
    # Plot ROC curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {method_name}")
    plt.legend()
    plt.show()

# ====== MAIN EXECUTION ======

# Load dataset
df = pd.read_csv("../alzheimers_disease_data.csv")

# Explore and clean data
explore_data(df)
df = clean_data(df)

# Features and target variable
X = df.drop(columns=['Diagnosis']).values
y = df['Diagnosis']

# Define the columns based on the type of attributes
binary_columns = ['Gender', 'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 
                  'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'MemoryComplaints', 
                  'BehavioralProblems', 'Confusion', 'Disorientation', 'PersonalityChanges', 
                  'DifficultyCompletingTasks', 'Forgetfulness','Diagnosis']
continuous_columns = ['BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 
                      'SleepQuality', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 
                      'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL']
discrete_columns = ['Age', 'Ethnicity', 'EducationLevel', 'SystolicBP', 'DiastolicBP']
categorical_columns = ['DoctorInCharge']

# Create a preprocessor for the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('binary', FunctionTransformer(), binary_columns),  # No transformation for binary columns
        ('continuous', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values if any
            ('scaler', StandardScaler())  # Standardize continuous features
        ]), continuous_columns),
        ('discrete', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode discrete columns
        ]), discrete_columns)
    ])

# Preprocess data
X_processed = preprocessor.fit_transform(df)

# List of preprocessing methods
preprocessing_methods = ['Standardization', 'Normalization', 'Centering']

# Evaluate performance and plot ROC curves for each preprocessing method
for method in preprocessing_methods:
    X_method = apply_preprocessing(X_processed, method)
    evaluate_and_plot_model(X_method, y, method)
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

# ====== FUNCTION DEFINITIONS ======

def explore_data(df):
    """Explore and print dataset information."""
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSample Data:")
    print(df.head())

def clean_data(df):
    """Clean data by removing unnecessary columns and rows with missing values."""
    df = df.dropna()
    df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')  # Drop non-useful columns
    return df

def apply_preprocessing(X, method):
    """Apply different preprocessing methods."""
    if method == 'Standardization':
        scaler = StandardScaler()
    elif method == 'Normalization':
        scaler = MinMaxScaler()
    elif method == 'Centering':
        return X - np.mean(X, axis=0)  # Centering simply subtracts the mean
    
    return scaler.fit_transform(X)

def evaluate_and_plot_model(X_processed, y, method_name):
    """Evaluate the model performance with cross-validation and plot the ROC curve."""
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=4000, random_state=42)
    
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'AUC': []}
    cv_results = {'Fold': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'AUC': []}
    y_true_all, y_pred_all, y_prob_all = [], [], []  # Collect predictions and true values for ROC curve
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_processed, y)):
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # Collect metrics for the current fold
        fold_metrics = {
            'Fold': fold + 1,
            'Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred),
            'Recall': recall_score(y_val, y_pred),
            'F1 Score': f1_score(y_val, y_pred),
            'AUC': roc_auc_score(y_val, y_prob)
        }
        cv_results['Fold'].append(fold + 1)
        cv_results['Accuracy'].append(fold_metrics['Accuracy'])
        cv_results['Precision'].append(fold_metrics['Precision'])
        cv_results['Recall'].append(fold_metrics['Recall'])
        cv_results['F1 Score'].append(fold_metrics['F1 Score'])
        cv_results['AUC'].append(fold_metrics['AUC'])
        
        # Collect data for the ROC curve
        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)
        
    # Print fold-wise metrics and the cross-validation performance
    print(f"\nCross-validation performance for {method_name}:")
    cv_df = pd.DataFrame(cv_results)
    print(cv_df)
    
    # Print overall performance
    print(f"\nPerformance for {method_name}:")
    for metric, values in metrics.items():
        print(f"{metric}: {np.mean(values):.4f}")
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    auc = roc_auc_score(y_true_all, y_prob_all)
    
    # Plot ROC curve
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {method_name}")
    plt.legend()
    plt.show()

def preview_transformed_data(X_processed):
    """Preview transformed data after preprocessing."""
    print("\nPreview of Transformed Data:")
    print(X_processed[:5])  # Show the first 5 rows of transformed data

# ====== MAIN EXECUTION ======

# Load dataset
df = pd.read_csv("../alzheimers_disease_data.csv")

# Explore and clean data
explore_data(df)
df = clean_data(df)

# Features and target variable
X = df.drop(columns=['Diagnosis']).values
y = df['Diagnosis']

# Define the columns based on the type of attributes
binary_columns = ['Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 
                  'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'MemoryComplaints', 
                  'BehavioralProblems', 'Confusion', 'Disorientation', 'PersonalityChanges', 
                  'DifficultyCompletingTasks', 'Forgetfulness']
continuous_columns = ['BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 
                      'SleepQuality', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 
                      'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL','SystolicBP', 'DiastolicBP','Age']
discrete_columns = ['Age', 'Ethnicity', 'EducationLevel', 'SystolicBP', 'DiastolicBP']
categorical_columns = ['DoctorInCharge']

# Create a preprocessor for the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('binary', FunctionTransformer(), binary_columns),  # No transformation for binary columns
        ('continuous', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values if any
            ('scaler', StandardScaler())  # Standardize continuous features
        ]), continuous_columns),
        ('discrete', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode Ethnicity and EducationLevel
        ]), ['Ethnicity', 'EducationLevel']) 
    ])

# Preprocess data
X_processed = preprocessor.fit_transform(df)

# Preview the transformed data
preview_transformed_data(X_processed)

# List of preprocessing methods
preprocessing_methods = ['Standardization', 'Normalization', 'Centering']

# Evaluate performance and plot ROC curves for each preprocessing method
for method in preprocessing_methods:
    print(f"\nApplying {method} to the transformed data...")
    X_method = apply_preprocessing(X_processed, method)
    
    # Print a preview of the data after applying preprocessing
    print(f"\nPreview of data after {method} transformation:")
    print(X_method[:5])  # Preview of the transformed data
    # Evaluate the model and plot ROC curve
    evaluate_and_plot_model(X_method, y, method)
