import csv
import os

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


def save_best_individual_report(best_individual, output_txt_path):
    """
    Save the best individual's chromosome and fitness to a text file.

    Args:
        best_individual (Individual): The best individual from the final population.
        output_txt_path (str): File path to save the text report.
    """
    with open(output_txt_path, 'w') as f:
        f.write("Best Individual Report\n")
        f.write("======================\n")
        f.write(f"Fitness: {best_individual.fitness:.4f}\n")
        f.write(f"Selected Features: {best_individual.num_selected_features()}\n")
        f.write(f"Chromosome: {best_individual.chromosome.tolist()}\n")


def save_results(params, avg_best_fitness, avg_generations, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "ga_results.csv")

    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:  # Check if file is empty
            writer.writerow(["Population Size", "Crossover Probability", "Mutation Probability",
                             "Average Best Fitness", "Average Generations"])
        
        writer.writerow([
            params['pop_size'],
            params['crossover_prob'],
            params['mutation_prob'],
            avg_best_fitness,
            avg_generations
        ])