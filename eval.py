import os
import json
import scanpy as sc
import scib
import anndata as ad
from model import PCA_Model, MOFA_Model, MultiVI_Model, Mowgli_Model
import numpy as np
import pandas as pd

class Evaluator:
    def __init__(self, latent_dir, output_file):
        """
        Initializes the Evaluator class for batch processing of latent space files.
        
        :param latent_dir: Directory containing latent space files (.h5ad format).
        :param output_file: Path to save the combined JSON file.
        """
        self.latent_dir = latent_dir
        self.output_file = output_file
        self.models = {
            "pca_output": PCA_Model,
            "mofa_output": MOFA_Model,
            "multivi_output": MultiVI_Model,
            "mowgli_output": Mowgli_Model,
        }

    def process_models(self):
        if not os.path.exists(self.latent_dir):
            raise FileNotFoundError(f"Directory not found: {self.latent_dir}")

        results = {}

        for model_folder, model_class in self.models.items():
            model_dir = os.path.join(self.latent_dir, model_folder)
            if not os.path.exists(model_dir):
                print(f"Skipping {model_folder}: Directory {model_dir} not found.")
                continue

            for file in os.listdir(model_dir):
                if file.endswith(".h5ad"):
                    try:
                        dataset_name = "_".join(file.split("_")[1:]).split(".")[0]
                        print(f"Processing {model_folder} for file {file}, dataset {dataset_name}")

                        config_path = "./config.json"
                        if not os.path.exists(config_path):
                            raise FileNotFoundError(f"Configuration file not found: {config_path}")

                        model_instance = model_class(dataset=None, dataset_name=dataset_name, config_path=config_path)
                        model_instance.latent_filepath = os.path.join(model_dir, file)
                        latent_data = model_instance.load_latent()
                        if latent_data is None:
                            print(f"Failed to load latent data for file {file}, dataset {dataset_name}")
                            continue
                        latent_data.var_names_make_unique()
                        if isinstance(model_instance, MOFA_Model):
                            sc.pp.neighbors(latent_data, use_rep="X_mofa")  # Update as necessary
                        elif isinstance(model_instance, Mowgli_Model):
                            sc.pp.neighbors(latent_data, use_rep="X_mowgli")
                        elif isinstance(model_instance, MultiVI_Model):
                            sc.pp.neighbors(latent_data, use_rep="X_multivi")
                        else:
                            sc.pp.neighbors(latent_data, use_rep="X_pca")

                        # Calculate SCIB metrics
                        metrics = self.calculate_metrics(latent_data)
                        print("\nMetrics for dataset '{}':\n{}".format(dataset_name, metrics))

                        # Define metric names to map to

                        # Store metrics in the results dictionary
                        if model_folder not in results:
                            results[model_folder] = {}
                        results[model_folder][dataset_name] ="{}".format(metrics)

                    except Exception as e:
                        print(f"Error processing {model_folder} for {file}: {e}")

        with open(self.output_file, "w+") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {self.output_file}")

    def calculate_metrics(self, latent_data):
        """
        Simplified wrapper for scib.metrics.metrics to automatically infer parameters.
        
        :param latent_data: AnnData object containing latent space data.
        :param batch_key: Key for batch information in .obs (optional).
        :param label_key: Key for label information in .obs (optional).
        :param embed: Key for embeddings in .obsm (optional).
        :return: Dictionary with calculated metrics.
        """
        batch_key = None
        label_key = None
        embed_key = None
        # Auto-detect batch_key if not provided
        if batch_key is None:
            batch_key = next((key for key in latent_data.obs.keys() if key.endswith("batch")), None)

        # Auto-detect label_key if not provided
        if label_key is None:
            label_key = next((key for key in latent_data.obs.keys() if key.endswith("cell_type")), None)

        # Auto-detect embedding if not provided
        if embed_key is None:
            embed_key = next((key for key in latent_data.obsm.keys() if key.startswith("X_")), None)

        if not batch_key or not label_key or not embed_key:
            raise ValueError(f"Failed to detect batch_key={batch_key}, label_key={label_key}, or embed={embed_key}")

        print(f"Using parameters: batch_key={batch_key}, label_key={label_key}, embed={embed_key}")


        # Calculate metrics
        metrics = scib.metrics.metrics(
            latent_data,
            latent_data,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed_key,
            ari_=True,
            nmi_=True,
            silhouette_=True,
            graph_conn_=True,
            isolated_labels_asw_=True,
        )
        return metrics


# Usage
if __name__ == "__main__":
    evaluator = Evaluator(latent_dir="./outputs", output_file="./outputs/results.json")
    evaluator.process_models()
