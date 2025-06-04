import csv
import os
import numpy as np

def save_generation_report(population_history, output_csv_path):
    """
    Save best and average fitness and feature counts to CSV.

    Args:
        population_history (list of Population): History of population over generations.
        output_csv_path (str): File path to save the report.
    """
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generation", "Best Fitness", "Average Fitness", "Best Features", "Average Features"])

        for i, pop in enumerate(population_history):
            generation = i + 1
            best = pop.get_best_individual()
            avg_fitness = sum(ind.fitness for ind in pop.individuals) / len(pop.individuals)
            avg_features = sum(ind.num_selected_features() for ind in pop.individuals) / len(pop.individuals)

            writer.writerow([
                generation,
                best.fitness,
                avg_fitness,
                best.num_selected_features(),
                avg_features
            ])


def save_trial_details(param_set_idx, param_dict, trial_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"params_set_{param_set_idx}_trials.txt")

    with open(path, 'w') as f:
        f.write(f"Parameter Set: Population Size={param_dict['pop_size']}, "
                f"Crossover Prob={param_dict['crossover_prob']}, "
                f"Mutation Prob={param_dict['mutation_prob']}\n\n")
        
        for i, trial in enumerate(trial_results, 1):
            f.write(f"Trial {i}:\n")
            f.write(f"  Best Fitness: {trial['fitness']:.4f}\n")
            f.write(f"  Generations: {trial['generations']}\n")
            f.write(f"  Selected Features (mask): {trial['mask']}\n")
    
        avg_fitness = np.mean([t['fitness'] for t in trial_results])
        avg_generations = np.mean([t['generations'] for t in trial_results])
        f.write(f"Average Best Fitness: {avg_fitness:.4f}\n")
        f.write(f"Average Generations: {avg_generations:.2f}\n")

def save_results(params, avg_best_fitness, avg_generations, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "ga_results.csv")

    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:  
            writer.writerow(["Population Size", "Crossover Probability", "Mutation Probability",
                             "Average Best Fitness", "Average Generations"])
        
        writer.writerow([
            params['pop_size'],
            params['crossover_prob'],
            params['mutation_prob'],
            avg_best_fitness,
            avg_generations
        ])