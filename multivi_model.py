import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import scvi
import muon as mu
import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import scanpy as sc
import seaborn as sns
import torch
import pandas as pd

class MultiVI_Model:
    """MultiVI Model implementation."""
    
    def __init__(self, data_dir, dataset, adata_rna, adata_atac, latent_key, join='outer', axis=1, label='modality', name="datasetName"):
        print("Initializing MultiVI Model")
        self.data_dir = data_dir
        self.dataset = dataset
        self.adata_rna = adata_rna 
        self.adata_atac = adata_atac
        self.latent_key = latent_key
        self.name = name

        self.adata_mvi = ad.concat([self.adata_rna, self.adata_atac], join=join, axis=axis, label=label)

        scvi.model.MULTIVI.setup_anndata(self.adata_mvi)
        self.model = scvi.model.MULTIVI(
            self.adata_mvi,
            n_genes=(self.adata_mvi.var["modality"] == "0").sum(),
            n_regions=(self.adata_mvi.var["modality"] == "1").sum(),
            )
            
    def to(self, device='cpu'):
        """
        Method to set GPU or CPU mode for MOFA+.
        """
        # Convert 'gpu' to 'cuda:0'
        if device == 'gpu':
            device = 'cuda:0'

        try:
            print(f"Moving MultiVI model to {device}")
            self.model.to_device(device)
            print(f"Model successfully moved to {device}")
        except Exception as e:
            print(f"Invalid device '{device}' specified. Use 'cpu' or 'gpu'.")
        
    def train(self):
        print("Training MultiVI Model")
        try:
            self.model.train()
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        """Save latent data generated with the MultiVI model."""
        print("Saving latent data")
        try:
            self.adata_mvi.obsm[self.latent_key] = self.model.get_latent_representation()
            self.adata_mvi.write(self.data_dir+f"multivi_{self.name}.txt")

            print(f"MultiVI model for dataset {self.name} was saved as multivi_{self.dataset}.txt")
        except Exception as e:
            print(f"Error saving latent data: {e}")


    def load_latent(self):
        """Load latent data from saved files."""

    def umap(self, filename, min_dist=0.2, color_type="cell_type"):
        """Generate UMAP visualization."""
        print("Generating UMAP plot")
        try:
            sc.pp.neighbors(self.adata_mvi, use_rep=self.latent_key)
            sc.tl.umap(self.adata_mvi, min_dist=min_dist)
            sc.pl.umap(self.adata_mvi, color=color_type, save=f"multivi_{filename}_umap_plot.png")
            print(f"A UMAP plot for MultiVI model with dataset {self.name} was succesfully generated and saved as {filename}")

        except Exception as e:
            print(f"Error generating UMAP: {e}")
