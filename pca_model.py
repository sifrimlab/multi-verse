import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import muon as mu
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

class PCA_Model:
    """PCA implementation"""
    
    def __init__(self, data_dir, dataset, n_components=20):
        """Initialize the PCA model with the specified dataset."""

        print("Initializing PCA Model")
        self.data_dir = data_dir
        self.dataset = dataset
        self.n_components = n_components

        # Ensure observations intersect between modalities
        self._prepare_dataset()
        self.pca = PCA(n_components=self.n_components)
    
    def train(self):
        """Perform PCA on the RNA modality data."""
        print("Training PCA Model")
        
        # Check if RNA data is available
        if 'rna' not in self.dataset.mod:
            raise ValueError("RNA modality not found in dataset")
        
        # Perform PCA on the RNA expression matrix
        adata_rna = self.dataset['rna']
        rna_data = adata_rna.X.toarray() if hasattr(adata_rna.X, 'toarray') else adata_rna.X

        # Fit PCA and transform the data
        latent_representation = self.pca.fit_transform(rna_data)
        adata_rna.obsm['X_pca'] = latent_representation
        print(f"PCA completed with {self.n_components} components")

    def save_latent(self):
        """Save the PCA latent representations."""
        print("Saving PCA latent embeddings")
        output_path = os.path.join(self.data_dir, f"pca_{self.dataset}.h5ad")
        self.dataset.write(output_path)
        print(f"Latent data saved to {output_path}")

    def load_latent(self, filepath):
        """Load latent data from a saved file."""
        print(f"Loading latent data from {filepath}")
        if os.path.exists(filepath):
            self.dataset = mu.read(filepath)
            print("Latent data loaded successfully.")
        else:
            print(f"File not found: {filepath}")

    def umap(self, random_state=1, filename=None):
        """Generate UMAP visualization using PCA embeddings."""
        print("Generating UMAP with PCA embeddings")
        
        # Use the PCA latent representation for UMAP
        sc.pp.neighbors(self.dataset['rna'], use_rep="X_pca")
        sc.tl.umap(self.dataset['rna'], random_state=random_state)
        self.dataset['rna'].obsm["X_pca_umap"] = self.dataset['rna'].obsm["X_umap"]

        # Plotting UMAP and saving the figure
        if not filename:
            filename = os.path.join(self.data_dir, f"pca_{self.dataset}_umap_plot.png")
        sc.pl.umap(self.dataset['rna'], color=["celltype"], save=filename)
        print(f"UMAP plot saved as {filename}")