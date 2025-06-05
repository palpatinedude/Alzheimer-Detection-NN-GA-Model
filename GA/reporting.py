# this code contains utility functions to log and store results of a genetic algorithm
# it includes saving generation statistics, detailed trial results, and summarized parameter evaluations

import csv
import os
import numpy as np

# this function saves generation-wise statistics to a csv file
def save_generation_report(population_history, output_csv_path):
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # write the header row
        writer.writerow(["Generation", "Best Fitness", "Average Fitness", "Best Features", "Average Features"])

        for i, pop in enumerate(population_history):
            generation = i + 1  # generation index starts from 1
            best = pop.get_best_individual()  # get the best individual
            avg_fitness = sum(ind.fitness for ind in pop.individuals) / len(pop.individuals)  # calculate average fitness
            avg_features = sum(ind.num_selected_features() for ind in pop.individuals) / len(pop.individuals)  # average number of selected features

            # write stats of the current generation
            writer.writerow([
                generation,
                best.fitness,
                avg_fitness,
                best.num_selected_features(),
                avg_features
            ])

# this function saves detailed results from multiple trials for a given parameter set
def save_trial_details(param_set_idx, param_dict, trial_results, output_dir):
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

# this function appends summarized average results for a parameter configuration to a csv file
def save_results(params, avg_best_fitness, avg_generations, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)  # ensure output directory exists
    output_csv_path = os.path.join(output_dir, "ga_results.csv")  # define the path to the results file

    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # write the header only if the file is empty
        if csvfile.tell() == 0:
            writer.writerow(["Population Size", "Crossover Probability", "Mutation Probability",
                             "Average Best Fitness", "Average Generations"])
        
        # write the summary row
        writer.writerow([
            params['pop_size'],
            params['crossover_prob'],
            params['mutation_prob'],
            avg_best_fitness,
            avg_generations
        ])
