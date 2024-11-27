import sys
import os
import torch
torch.cuda.is_available()

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

from train import Trainer


def main():
    # Check if a config file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[1]

    # Verify if the provided file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)

    print(f"Using configuration file: {config_file}")

    # Initialize the Trainer with the provided config file
    my_trainer = Trainer(config_path=config_file)

    # Load all datasets
    datasets = my_trainer.load_datasets()

    selected_models = my_trainer.model_select(datasets)  # Assuming `selected_models` is now a dict with dataset_name as keys


    # Iterate over datasets and their respective models
    for dataset_name, dataset_data in datasets.items():
        print(f"\n=== Processing dataset: {dataset_name} ===")
        # print(dataset_data)
        
        # Get the models specific to this dataset
        if dataset_name not in selected_models:
            print(f"No models selected for dataset {dataset_name}")
            continue

        models = selected_models[dataset_name]
        print(f"Loaded models for {dataset_name}: {models.keys()}")
        
        # Train each model
        for model_name, model in models.items():
            try:
                print(f"\nTraining model: {model_name} for dataset: {dataset_name}")
                model.to()
                model.train()
                model.umap()  # Perform UMAP visualization if needed
            except Exception as e:
                print(f"Error while processing model '{model_name}' for dataset '{dataset_name}': {e}")
                # Continue to the next model or dataset
                continue

if __name__ == "__main__":
    main()
