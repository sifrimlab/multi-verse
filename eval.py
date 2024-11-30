import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scib
from train import Trainer
from model import PCA_Model, MOFA_Model, MultiVI_Model, Mowgli_Model
from config import load_config

class Evaluator:
    def __init__(self, latent_dir="./outputs", output_file="./outputs/results.json", trainer: Trainer = None):
        """
        Initializes the Evaluator class for batch processing of latent space files.
        
        :param latent_dir: Directory containing latent space files (.h5ad format).
        :param output_file: Path to save the combined JSON file.
        """
        self.trainer = trainer
        self.latent_dir = latent_dir
        self.output_file = output_file

        # Initialize metrics and result
        self.adata_unint = None
        self.model_list = {}
        self.metrics=None
        self.results={}

    def unintegrated_adata(self):
        if self.trainer.data == None:
            self.trainer.load_datasets()
        loader = self.trainer.data
    
        if self.trainer.models == None:
            self.trainer.model_select(loader) # Load model to use load_latent()
        model_trainer = self.trainer.models

        # Unintegrated anndata for scib metrics is from PCA model
        self.adata_unint={}
        for dataset_name, model_dict in model_trainer.items():
            for model_name, model_obj in model_dict.items():
                if model_name == "pca" and isinstance(model_obj, PCA_Model):
                    adata_unint =  model_obj.dataset # Original input dataset for PCA_model
                    if "batch" not in adata_unint.obs_keys():
                        adata_unint.obs["batch"] = "batch_1"
                    sc.pp.neighbors(adata_unint, use_rep='X', n_neighbors=20) 
                    sc.tl.leiden(adata_unint)
                    self.adata_unint[dataset_name] = adata_unint
        self.model_list = model_trainer
        return self.adata_unint

    def process_models(self,config_path="./config.json"):
        if not os.path.exists(self.latent_dir):
            raise FileNotFoundError(f"Directory not found: {self.latent_dir}")
        
        if self.adata_unint==None:
            adata_unint = self.unintegrated_adata()
        else:
            adata_unint = self.adata_unint

        for dataset_name, model_dict in self.model_list.items():
            self.results[dataset_name] = {}
            for model_name, model_obj in model_dict.items():
                try:
                    if isinstance(model_obj, PCA_Model):
                        embed_key = "X_pca"
                    elif isinstance(model_obj, MOFA_Model):
                        embed_key = "X_mofa"
                    elif isinstance(model_obj, Mowgli_Model):
                        embed_key = "X_mowgli"
                    elif isinstance(model_obj, MultiVI_Model):
                        embed_key = "X_multivi"

                    latent_data = model_obj.load_latent()
                    print(latent_data)
                    print(f"\n==== Start calculating metrics =========\n")

                    check_annotation = load_config(config_path=config_path)["data"][dataset_name]["rna"]["annotation"]
                    sc.pp.neighbors(latent_data, use_rep=embed_key)
                    sc.tl.leiden(latent_data)
                    if check_annotation != None:
                        metrics = self.calculate_metrics(unintegrated_data=adata_unint[dataset_name],
                                                    latent_data= latent_data,
                                                    emb_key=embed_key,
                                                    batch="batch", 
                                                    annotate="cell_type")
                    else:
                        metrics = self.calculate_metrics(unintegrated_data=adata_unint[dataset_name],
                                                    latent_data= latent_data,
                                                    emb_key=embed_key,
                                                    batch="batch", 
                                                    annotate="leiden")

                    print("\nMetrics for dataset '{}', model {}:\n".format(dataset_name, model_name))
                    
                    metrics_dict = metrics.squeeze().apply(lambda x: None if pd.isna(x) else x).to_dict()
                    # Store metrics in the results dictionary
                    filtered_metrics = {key: value for key, value in metrics_dict.items() if not np.isnan(value)}
                    self.results[dataset_name][model_name]=filtered_metrics
                except Exception as e:
                    print(f"Skipping {dataset_name} {model_name} due to an error: {e}")
                    continue  # Move on to the next iteration
            print("====================================================\n")
            print(self.results)

        with open(self.output_file, "w+") as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {self.output_file}")

    def calculate_metrics(self, unintegrated_data, latent_data, emb_key:str = None, batch: str = None, annotate: str = None):
        """
        Simplified wrapper for scib.metrics.metrics to automatically infer parameters.
        
        :param latent_data: AnnData object containing latent space data.
        :param batch_key: Key for batch information in .obs (optional).
        :param label_key: Key for label information in .obs (optional).
        :param embed: Key for embeddings in .obsm (optional).
        :return: Dictionary with calculated metrics.
        """
        batch_key = batch
        label_key = annotate
        embed_key = emb_key
        # Auto-detect batch_key if not provided
        if batch_key is None:
            batch_key = next((key for key in latent_data.obs.keys() if key.endswith("batch")), None)

        # Auto-detect label_key if not provided
        if label_key is None:
            label_key = next((key for key in latent_data.obs.keys() if key.endswith("cell_type")), None)

        # Auto-detect embedding if not provided
        if embed_key is None:
            embed_key = next((key for key in latent_data.obsm.keys() if key.startswith("X_")), None)

        if not batch_key or not embed_key:
            raise ValueError(f"Failed to detect batch_key={batch_key}, or embed={embed_key}")

        print(f"Using parameters: batch_key={batch_key}, label_key={label_key}, embed={embed_key}")

        metrics = scib.metrics.metrics(
                unintegrated_data,
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
        self.metrics = metrics
        return self.metrics

