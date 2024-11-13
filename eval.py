import os
import json
import scanpy as sc
import scib
import anndata as ad


class Evaluator:
    def __init__(self, latent_dir, output_file="results.json"):
        """
        Initializes the Evaluator class for batch processing of latent space files.
        
        :param latent_dir: Directory containing latent space files (.h5ad format).
        :param output_file: Path to save the combined JSON file.
        """
        self.latent_dir = latent_dir
        self.output_file = output_file

    def process_all_latents(self):
        """
        Processes all latent space files in the specified directory, calculates metrics, 
        and writes them to a single JSON file.
        """
        if not os.path.exists(self.latent_dir):
            raise FileNotFoundError(f"Directory not found: {self.latent_dir}")

        # Find all .h5ad files in the directory
        latent_files = [f for f in os.listdir(self.latent_dir) if f.endswith('.h5ad')]
        if not latent_files:
            print(f"No latent space files found in {self.latent_dir}.")
            return

        print(f"Found {len(latent_files)} latent space files in {self.latent_dir}.")

        # Create a dictionary to store results
        results = {}

        for file in latent_files:
            try:
                result = self.process_single_latent(file)
                results[file] = result
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        # Write all results to a single JSON file
        print(f"Writing combined results to {self.output_file}")
        with open(self.output_file, "w") as json_file:
            json.dump(results, json_file, indent=4)
        print(f"All results saved successfully.")

    def process_single_latent(self, filename):
        """
        Processes a single latent space file, calculates metrics, and returns the results as a dictionary.

        :param filename: Name of the latent space file.
        :return: A dictionary containing the results for this file.
        """
        input_path = os.path.join(self.latent_dir, filename)

        # Load latent space data
        print(f"Processing file: {input_path}")
        latent_data = sc.read_h5ad(input_path)

        # Convert AnnData to dictionary
        latent_dict = {
            "obs": latent_data.obs.to_dict(),
        }

        # Calculate metrics using scib
        metrics = self.calculate_metrics(latent_data)

        # Merge metrics into the result dictionary
        latent_dict["metrics"] = metrics
        return latent_dict

    def calculate_metrics(self, latent_data, batch_key=None, label_key=None, embed=None):
        """
        Simplified wrapper for scib.metrics.metrics to automatically infer parameters.
        
        :param latent_data: AnnData object containing latent space data.
        :param batch_key: Key for batch information in .obs (optional).
        :param label_key: Key for label information in .obs (optional).
        :param embed: Key for embeddings in .obsm (optional).
        :return: Dictionary with calculated metrics.
        """
        # Auto-detect batch_key if not provided
        if batch_key is None:
            batch_key = next((key for key in latent_data.obs.keys() if key.endswith(":batch")), None)

        # Auto-detect label_key if not provided
        if label_key is None:
            label_key = next((key for key in latent_data.obs.keys() if key.endswith(":cell_type")), None)

        # Auto-detect embedding if not provided
        if embed is None:
            embed = next((key for key in latent_data.obsm.keys() if key.startswith("X_")), None)

        if not batch_key or not label_key or not embed:
            raise ValueError(f"Failed to detect batch_key={batch_key}, label_key={label_key}, or embed={embed}")

        print(f"Using parameters: batch_key={batch_key}, label_key={label_key}, embed={embed}")

        # Calculate metrics
        return scib.metrics.metrics(
            latent_data,
            latent_data,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            ari_=True,
            nmi_=True,
            silhouette_=True,
            graph_conn_=True,
            isolated_labels_asw_=True,
        )


# Usage
if __name__ == "__main__":
    evaluator = Evaluator(latent_dir="./outputs", output_file="results.json")
    evaluator.process_all_latents()
