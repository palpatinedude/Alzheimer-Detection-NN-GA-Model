# this file loads and inspects dataset: identifies types, prints ranges, detects outliers, and prepares data for modeling

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from visualization.evalutation_plots import plot_histograms

# this function loads csv data into a pandas dataframe
def load_data(file_path):
    return pd.read_csv(file_path)

# this function determines the type of each attribute in the dataframe
def determine_attribute_type(df):
    types = {}
    for col in df.columns:
        if col == 'Diagnosis':
            continue  # skip target column
        if df[col].dtype == 'int64':
            types[col] = 'Binary (0/1)' if df[col].nunique() == 2 else 'Discrete Integer'
        elif df[col].dtype == 'float64':
            types[col] = 'Numeric'
        else:
            types[col] = 'Unknown'
    return types

# this function prints min and max values of each feature
def print_feature_ranges(df, types):
    print("\nFeature Ranges:")
    for col, typ in types.items():
        print(f"{col} ({typ}): Min = {df[col].min()}, Max = {df[col].max()}")

# this function detects outliers using z-score method
def detect_outliers(df, threshold=3):
    outliers = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        z_scores = zscore(df[col].dropna())  # compute z-scores
        outliers[col] = np.where(np.abs(z_scores) > threshold)[0]  # store indices of outliers

    return outliers

# this function scales the data using standardization or normalization
def scale_data(X, norm):
    scaler = MinMaxScaler() if norm else StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), scaler

# this function loads, cleans, inspects, and splits the dataset
def inspect_data(file_path):
    df = load_data(file_path)
    # remove columns not useful for modeling if they exist
    df.drop(columns=['PatientID', 'DoctorInCharge'], inplace=True, errors='ignore')

  #  types = determine_attribute_type(df)
   # print_feature_ranges(df, types)

    # plot histograms for visual inspection
  #  for col, typ in types.items():
   #      plot_histograms(df[[col]], {col: typ})

    outliers = detect_outliers(df)
    print("\nOutliers detected at indices:")
    for col, indices in outliers.items():
        print(f"{col}: {indices}")

    # separate features and target variable
    X, y = df.drop(columns='Diagnosis'), df['Diagnosis']
   
    return X, y

