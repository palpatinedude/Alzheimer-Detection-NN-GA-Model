from A1_Preprocessing import load_data, detect_outliers, determine_attribute_type, print_feature_ranges, plot_histograms,evaluate_model,plot_confusion_matrix,scale_data
from A2_NeuralNetwork import evaluate_with_different_hidden_units, create_model,k_fold_evaluation
import pandas as pd
from sklearn.model_selection import train_test_split

def main(file_path):
    print(" ############# A1: Preprocessing and Model Evaluation using Logistic Regression ###############" )
    # Load and clean data
    df = load_data(file_path)
    df.drop(columns=['PatientID', 'DoctorInCharge'], inplace=True, errors='ignore')

    # Identify feature types
    attribute_types = determine_attribute_type(df)
    print_feature_ranges(df, attribute_types)
   # plot_histograms(df, attribute_types)

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
    print("\n Final test set (20%) is reserved and untouched for future model testing.\n")

    print(" ############# A2 : Neural Network  Building   ###############" )
    evaluate_with_different_hidden_units(X_train_std, y_train, X_test_std, y_test)

# ----------------- RUN -----------------

if __name__ == "__main__":
    main("../alzheimers_disease_data.csv")    