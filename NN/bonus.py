# A5: Explore deeper neural networks (2-3 hidden layers) to test if added depth improves alzheimer's prediction.
# Try different neuron setups per layer, evaluate with CE, MSE, and Accuracy using 5 fold CV.

# ------------------------- Imports ------------------------- #
from preprocessing.preprocessing import inspect_data
import os
import shutil
from sklearn.model_selection import train_test_split
from modeling.cross_validation import logistic_scaling
from config import RESULTS_DIR_NN_BONUS
from bonus_dir.tuning_hidden import tuning_hidden_layers_neurons
from bonus_dir.model import create_model_bonus
from modeling.evaluation import evaluate_performance
from modeling.training import train_model
from helpers import print_test_metrics
from visualization.evalutation_plots import plot_confusion_matrix


# ------------------------- Main Function ------------------------- #
def main(file_path):

    # Setup results directory
    output_dir = RESULTS_DIR_NN_BONUS
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------- A1: Preprocessing and Logistic Regression Evaluation ------------------------- #
    print("##### A1: Preprocessing and Logistic Regression Evaluation #####")
    X, y = inspect_data(file_path)

    # Split data into training and test sets (keeping 20% for final evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Evaluate Logistic Regression with Standardization 
    X_train_std, X_test_std,_,_ = logistic_scaling(X_train, X_test, y_train, norm=False)

      # ------------------------- A2: Tuning Hidden Numbers of Neurons and Layers  ------------------------- #
    print("\n##### A2: Tuning Hidden Numbers of Neurons and Layers  #####")
    input_dim = X_train_std.shape[1]

    # Define optimized hyperparameters from the initial step and standardization
    learning_rate = 0.05
    momentum = 0.2
    reg = 0.01

    best_layers, best_neurons, best_score, best_result =  tuning_hidden_layers_neurons(X_train_std, y_train, input_dim, learning_rate, momentum, reg)
    print(f"Best model → Layers: {best_layers}, Neurons per Layer: {best_neurons}, Accuracy: {best_score:.4f}")

    hidden_units = [best_neurons] * best_layers

    # Now based on the best num_layer and neurons per layer create the model
    final_model = create_model_bonus(input_dim=input_dim,hidden_units=hidden_units,learning_rate=learning_rate,momentum=momentum,regularization=reg)
    train_stats = train_model(model=final_model, X_train=X_train_std, y_train=y_train, X_val=None, y_val=None, model_type='ann')
    test_metrics = evaluate_performance(model=final_model, X_val=X_test_std, y_val=y_test, model_type='ann')

    # Extract test set results and print metrics
    y_pred_test, eval_results_test, metrics_test = test_metrics
    accuracy, precision, recall, f1, roc_auc, confusion = metrics_test
    print_test_metrics(accuracy, precision, recall, f1, roc_auc)

    # Plot confusion matrix
    print("\n##### Confusion Matrix for Final Test Evaluation #####")
    plot_confusion_matrix(y_test, y_pred_test, title="Confusion Matrix - Final Test Evaluation")


    # Prepare output text
    config_summary = (
     f"Test Set Metrics:\n"
     f"Accuracy: {accuracy:.4f}\n"
     f"Precision: {precision:.4f}\n"
     f"Recall: {recall:.4f}\n"
     f"F1 Score: {f1:.4f}\n"
     f"ROC AUC: {roc_auc:.4f}\n"
   )

    # Save to txt file
    results_path = os.path.join(RESULTS_DIR_NN_BONUS, "best_deep_model_summary.txt")
    with open(results_path, "w") as f:
      f.write(config_summary)

    print(f"\nBest model summary written to {results_path}")

    
if __name__ == "__main__":
    main("alzheimers_disease_data.csv")