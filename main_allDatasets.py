import sys
import json
import os
import torch
torch.cuda.is_available()

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

    # Iterate over datasets
    for dataset_name, dataset_data in datasets.items():
        print(f"\n=== Processing dataset: {dataset_name} ===")
        
        # Select models for this dataset
        models = my_trainer.model_select(dataset_name, dataset_data)
        print(f"Loaded models for {dataset_name}: {models.keys()}")
        
        # Train each model
        for model_name, model in models.items():
            print(f"\nTraining model: {model_name} for dataset: {dataset_name}")
            model.train()
            model.umap()  # Perform UMAP visualization if needed

if __name__ == "__main__":
    main()
