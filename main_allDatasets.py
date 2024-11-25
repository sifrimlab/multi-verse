import json
import os
from train import Trainer
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

import torch
torch.cuda.is_available()

# Initialize the Trainer
my_trainer = Trainer(config_path="./config_alldatasets.json")

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
