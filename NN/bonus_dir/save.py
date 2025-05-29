# save results (table) for deep neural network
def save_results_deep_network(results, file_path):
    with open(file_path, 'w') as file:
        file.write(f"{'Hidden Neurons':<25}{'Hidden Layers':<15}{'CE Loss':<15}{'MSE':<15}{'Accuracy':<15}\n")
        file.write("-" * 85 + "\n")
        for result in results:
            file.write(f"{str(result['neurons_per_layer']):<25}{result['num_layers']:<15}"
                       f"{result['ce_loss']:<15.6f}{result['mse']:<15.6f}{result['accuracy']:<15.6f}\n")
    print(f"\nResults saved to: {file_path}")