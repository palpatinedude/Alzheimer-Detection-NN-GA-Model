# this code contains utility functions to log and store results of a genetic algorithm
# it includes saving generation statistics, detailed trial results, and summarized parameter evaluations

import os
import numpy as np


# this function saves detailed results from multiple trials for a given parameter set
def save_trial_details(param_set_idx, param_dict, trial_results, output_dir,selected_features):
    os.makedirs(output_dir, exist_ok=True)  # create the output directory if it doesn't exist
    path = os.path.join(output_dir, f"params_set_{param_set_idx}_trials.txt")  # define the output file path

    with open(path, 'w') as f:
        # write the parameter configuration
        f.write(f"Parameter Set: Population Size={param_dict['pop_size']}, "
                f"Crossover Prob={param_dict['crossover_prob']}, "
                f"Mutation Prob={param_dict['mutation_prob']}\n\n")
        
        # write individual trial results
        for i, trial in enumerate(trial_results, 1):
            f.write(f"Trial {i}:\n")
            f.write(f"  Best Fitness: {trial['fitness']:.4f}\n")
            f.write(f"  Generations: {trial['generations']}\n")
            f.write(f"  Selected Features (mask): {trial['mask']}\n")
            

    
        # calculate and write average metrics
        avg_fitness = np.mean([t['fitness'] for t in trial_results])
        avg_generations = np.mean([t['generations'] for t in trial_results])
        f.write(f"Average Best Fitness: {avg_fitness:.4f}\n")
        f.write(f"Average Generations: {avg_generations:.2f}\n")
        f.write(f"Number of Selected Features: {selected_features:.2f}\n")


# this function saves the best parameter set configuration based on aggregated results from all trials
def save_best_set_config(best_set, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "best_set.txt")
    with open(path, 'w') as f:
        f.write(f"Best Parameter Set:\n")
        f.write(f"  Set: {best_set['Set']}\n")
        f.write(f"  Population Size: {best_set['Population Size']}\n")
        f.write(f"  Crossover Probability: {best_set['Crossover Probability']}\n")
        f.write(f"  Mutation Probability: {best_set['Mutation Probability']}\n")
        f.write(f"  Average Best Fitness: {best_set['Average Best Fitness']:.4f}\n")
        f.write(f"  Average Generations: {best_set['Average Generations']:.2f}\n")
        f.write(f"  Average Selected Features: {best_set['Average Selected Features']:.2f}\n")
        f.write(f"  Best Individual Mask: {best_set['Best Individual Mask']}\n")

def save_metrics_to_file(val_metrics, test_metrics, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, "best_ga_model_summary.txt")
    with open(file_path, "w") as f:
        f.write("Best Model Evaluation Summary\n")
        f.write("==============================\n\n")

        f.write("Validation Set Metrics:\n")
        for k, v in val_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\nTest Set Metrics:\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Saved metrics summary to {file_path}")