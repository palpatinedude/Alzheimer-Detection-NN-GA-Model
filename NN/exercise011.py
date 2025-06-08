# ============================================================================================
# Project: Alzheimer’s Disease Diagnosis using ANN
# Description: This script performs preprocessing, model training, evaluation, and 
#    hyperparameter tuning using a simple feedforward  neural network .
#  Goal: Classify Alzheimer’s diagnosis based on patient data with performance comparison 
#    between Standardization and Normalization techniques,Number of Hidden Neurons,Learning Rate,Momentum and Regularization coefficents .
#  Author: Marianthi Thodi
#  AM: 1084576
# ============================================================================================



# ------------------------- Imports ------------------------- #
from NN.config import RESULTS_DIR_NN
from .preprocessing.preprocessing import inspect_data
from .reporting.experiments import select_best_config_hidden, select_best_config_hyper,select_best_config_regularization
import os
import shutil
from sklearn.model_selection import train_test_split
from .helpers import print_kfold_results, print_test_metrics
from .modeling.cross_validation import logistic_scaling
from .modeling.architecture import create_model_wrapper
from .modeling.training import train_model
from .modeling.evaluation import evaluate_performance
from .visualization.evalutation_plots import plot_confusion_matrix
from .visualization.training_plots import plot_training_history
from .reporting.result_saving import save_best_model_results



# ------------------------- Main Function ------------------------- #
def main(file_path):

    # Setup results directory
    output_dir = RESULTS_DIR_NN
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------- A1: Preprocessing and Logistic Regression Evaluation ------------------------- #
    print("##### A1: Preprocessing and Logistic Regression Evaluation #####")
    X, y = inspect_data(file_path)

    # Split data into training and test sets (keeping 20% for final evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


    # Evaluate Logistic Regression with Standardization and Normalization
    X_train_std, X_test_std,fold_results,final_metrics = logistic_scaling(X_train, X_test, y_train,False)
    print_kfold_results('logistic', fold_results, final_metrics)

    X_train_norm, X_test_norm,fold_results,final_metrics = logistic_scaling(X_train, X_test, y_train,True)
    print_kfold_results('logistic', fold_results, final_metrics)


      # ------------------------- A2: Number of Hidden Neurons and Transformation Method Tuning ------------------------- #
    print("\n##### A2: Tuning ANN Hidden Layer and Transformation Method  Standardization#####")

    # Select best configuration for hidden layer size
    best_h1, best_result,results_all = select_best_config_hidden(X_train_std, y_train)
    print_kfold_results('ann', best_result[0], best_result[1]) # from best config



    # ------------------------- A3: Hyperparameter Tuning (Learning Rate, Momentum) ------------------------- #
    print("\n##### A3: Hyperparameter Tuning (Learning Rate, Momentum) #####")
    best_lr, best_momentum, fold_results, final_metrics = select_best_config_hyper(X_train_std, y_train, best_h1)
    print_kfold_results('ann', fold_results, final_metrics)
    

    # ------------------------- A4: Regularization Coefficient Tuning ------------------------- #
    print("\n##### A4: Regularization Coefficient Tuning #####")
    best_reg, fold_results, final_metrics = select_best_config_regularization(X_train_std, y_train, best_h1, best_lr, best_momentum)
    print_kfold_results('ann', fold_results, final_metrics)
    print(best_reg)


   
    # ------------------------- A5: Final Evaluation on Test Set ------------------------- #
    print("\n##### A5: Final Evaluation on Test Set #####")

    # Split train data into train_sub and val_sub
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
       X_train_std, y_train, test_size=0.2, stratify=y_train, random_state=42
     )


    # Create, train, and evaluate the final ANN model
    final_model = create_model_wrapper('ann', X_train_std.shape[1], best_h1, best_lr, best_momentum, best_reg,simple_metrics=None)
    train_stats = train_model(model=final_model, X_train=X_train_sub, y_train=y_train_sub, X_val=None, y_val=None, model_type='ann')

    # After training the final model and getting train_stats
    train_acc_history = train_stats['accuracy_history']
    val_acc_history = train_stats['val_accuracy_history']
    train_loss_history = train_stats['history'].history['loss'] if train_stats['history'] else None
    val_loss_history = train_stats['history'].history.get('val_loss', None) if train_stats['history'] else None
    epochs_ran = train_stats['epochs_ran']

    final_train_acc = train_acc_history[-1] if train_acc_history else None
    final_val_acc = val_acc_history[-1] if val_acc_history else None
    final_train_loss = train_loss_history[-1] if train_loss_history else None
    final_val_loss = val_loss_history[-1] if val_loss_history else None

    # Plot them
    plot_training_history(
    train_acc=train_acc_history,
    val_acc=val_acc_history,
    train_loss=train_loss_history,
    val_loss=val_loss_history,
    epochs_ran=epochs_ran
    )

    # --- Evaluate on validation set ---
    y_pred_val, eval_results_val, metrics_val = evaluate_performance(
        model=final_model, X_val=X_val_sub, y_val=y_val_sub, model_type='ann'
    )

    val_loss = eval_results_val[0] if eval_results_val else None
    val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_confusion = metrics_val

    # --- Evaluate on test set ---
    y_pred_test, eval_results_test, metrics_test = evaluate_performance(
        model=final_model, X_val=X_test_std, y_val=y_test, model_type='ann'
    )

    test_loss = eval_results_test[0] if eval_results_test else None
    test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_confusion = metrics_test

    # --- Print all metrics ---
    print("\nTraining Set Metrics:")
    print(f"Loss: {final_train_loss:.4f}")
    print(f"Accuracy: {final_train_acc:.4f}")

    print("\nValidation Set Metrics:")
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    print(f"ROC AUC: {val_roc_auc:.4f}")

    print("\nTest Set Metrics:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"ROC AUC: {test_roc_auc:.4f}")


    # --- Plot confusion matrix for final test set ---
    print("\n##### Confusion Matrix for Final Test Evaluation #####")
    plot_confusion_matrix(y_test, y_pred_test, title="Confusion Matrix - Final Test Evaluation")

    # --- Save model and summary ---
    config_summary = (
        f"Best ANN Model Configuration:\n"
        f"Hidden Units: {best_h1}\n"
        f"Learning Rate: {best_lr}\n"
        f"Momentum: {best_momentum}\n"
        f"Regularization (Lambda): {best_reg}\n\n"
        f"Training Set Metrics:\n"
        f"Loss: {final_train_loss:.4f}\n"
        f"Accuracy: {final_train_acc:.4f}\n\n"
        f"Validation Set Metrics:\n"
        f"Loss: {val_loss:.4f}\n"
        f"Accuracy: {val_accuracy:.4f}\n"
        f"Precision: {val_precision:.4f}\n"
        f"Recall: {val_recall:.4f}\n"
        f"F1 Score: {val_f1:.4f}\n"
        f"ROC AUC: {val_roc_auc:.4f}\n\n"
        f"Test Set Metrics:\n"
        f"Loss: {test_loss:.4f}\n"
        f"Accuracy: {test_accuracy:.4f}\n"
        f"Precision: {test_precision:.4f}\n"
        f"Recall: {test_recall:.4f}\n"
        f"F1 Score: {test_f1:.4f}\n"
        f"ROC AUC: {test_roc_auc:.4f}\n"
    )

    save_best_model_results(
        output_dir=RESULTS_DIR_NN,
        model=final_model,
        best_h1=best_h1,
        best_lr=best_lr,
        best_momentum=best_momentum,
        best_reg=best_reg,
        config_summary=config_summary,
        X_val=X_val_sub,
        y_val=y_val_sub,
        X_test=X_test_std,
        y_test=y_test
    )
    # save results to pass to the final united code
    best_summary = {
        'Selected Features': 32, 
        'Validation Metrics': {
            'Loss': val_loss,
            'Accuracy': val_accuracy,
            'Precision': val_precision,
            'Recall': val_recall,
            'F1 Score': val_f1,
            'ROC AUC': val_roc_auc
        },
        'Test Metrics': {
            'Loss': test_loss,
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1 Score': test_f1,
            'ROC AUC': test_roc_auc
        }
    }
    return best_summary    


'''
    test_metrics = evaluate_performance(model=final_model, X_val=X_test_std, y_val=y_test, model_type='ann')

    # Extract test set results and print metrics
    y_pred_test, eval_results_test, metrics_test = test_metrics
    accuracy, precision, recall, f1, roc_auc, confusion = metrics_test
    print_test_metrics(accuracy, precision, recall, f1, roc_auc)

    




    # Plot confusion matrix
    print("\n##### Confusion Matrix for Final Test Evaluation #####")
    plot_confusion_matrix(y_test, y_pred_test, title="Confusion Matrix - Final Test Evaluation")

       # Add training/validation final stats to config summary (separate from test metrics)
    config_summary = (
        f"Best ANN Model Configuration:\n"
        f"Hidden Units: {best_h1}\n"
        f"Learning Rate: {best_lr}\n"
        f"Momentum: {best_momentum}\n"
        f"Regularization (Lambda): {best_reg}\n\n"
        f"Final Training Accuracy: {final_train_acc:.4f}\n"
        f"Final Validation Accuracy: {final_val_acc:.4f}\n"
        f"Final Training Loss: {final_train_loss:.4f}\n"
        f"Final Validation Loss: {final_val_loss:.4f}\n"
        f"Epochs Ran: {epochs_ran}\n\n"
        f"Test Set Metrics:\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"ROC AUC: {roc_auc:.4f}\n"
    )

    save_best_model_results(
      output_dir=RESULTS_DIR_NN,
      model=final_model,
      best_h1=best_h1,
      best_lr=best_lr,
      best_momentum=best_momentum,
      best_reg=best_reg,
      config_summary=config_summary,
      X_val=X_val_sub,
      y_val=y_val_sub,
      X_test=X_test_std,
      y_test=y_test
    )
    '''
   
if __name__ == "__main__":
    main("../alzheimers_disease_data.csv")





