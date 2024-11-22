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
    
    def __init__(self, data_dir, dataset, n_components=20, name="datasetName"):
        """Initialize the PCA model with the specified dataset."""

        print("Initializing PCA Model")
        self.data_dir = data_dir
        self.dataset = dataset
        self.n_components = n_components
        self.name = name
        self.gpu_mode = False  # Default to CPU mode


        # Ensure observations intersect between modalities
        # self._prepare_dataset()
        self.pca = PCA(n_components=self.n_components)
    
    # def _prepare_dataset(self):
    #     """
    #     Ensure modalities share the same observations.
    #     This step guarantees that all modalities have the same set of observations.
    #     """
    #     print("Ensuring observations are consistent across modalities")
    #     common_obs = None
    #     for mod in self.dataset.mod.keys():
    #         if common_obs is None:
    #             common_obs = set(self.dataset[mod].obs_names)
    #         else:
    #             common_obs &= set(self.dataset[mod].obs_names)
        
    #     # Filter each modality for the common observations
    #     for mod in self.dataset.mod.keys():
    #         self.dataset[mod] = self.dataset[mod][list(common_obs)]
    #     print(f"Number of shared observations: {len(common_obs)}")

    def to(self, device='cpu'):
        """
        Method to set GPU or CPU mode for MOFA+.
        """
        if device != 'cpu':
            print("PCA does not support GPU. Using CPU instead.")
        else:
            print("Using CPU mode for PCA.")
        self.gpu_mode = False
    
    def train(self):
        """Perform PCA on all modalities concatenated."""
        print("Training PCA Model")
        # Concatenate modalities for PCA
        modality_data = []
        for modality in self.dataset.mod.keys():
            data = self.dataset[modality].X.toarray() if hasattr(self.dataset[modality].X, 'toarray') else self.dataset[modality].X
            modality_data.append(data)

        # Concatenate all modalities along the feature axis
        combined_data = np.concatenate(modality_data, axis=1)

        # Fit PCA and transform the data
        latent_representation = self.pca.fit_transform(combined_data)

        # Assign the latent representation to each modality
        for modality in self.dataset.mod.keys():
            self.dataset[modality].obsm['X_pca'] = latent_representation
        print(f"PCA completed with {self.n_components} components")

    def save_latent(self):
        """Save the PCA latent representations."""
        print("Saving PCA latent embeddings")
        output_path = os.path.join(self.data_dir, f"pca_{self.name}.h5ad")
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
        """Generate UMAP visualization using combined PCA embeddings."""
        print("Generating UMAP with combined PCA embeddings")
    
        # Ensure PCA has been performed
        combined_pca_key = "X_pca"
        if not all(combined_pca_key in self.dataset[modality].obsm for modality in self.dataset.mod.keys()):
            raise RuntimeError("PCA results not found. Run `train` first.")

        # Use the latent PCA embeddings (assumes all modalities share the same embeddings)
        latent_representation = next(iter(self.dataset.mod.values())).obsm[combined_pca_key]

        # Create a synthetic AnnData object for UMAP
        umap_data = ad.AnnData(X=latent_representation)
        umap_data.obs = self.dataset.obs.copy()  # Copy metadata for UMAP visualization

        # Compute neighbors and UMAP
        sc.pp.neighbors(umap_data)
        sc.tl.umap(umap_data, random_state=random_state)

        # Save UMAP coordinates back to all modalities
        for modality in self.dataset.mod.keys():
            self.dataset[modality].obsm["X_pca_umap"] = umap_data.obsm["X_umap"]

        # Plot UMAP and save the figure
        if not filename:
            filename = os.path.join(self.data_dir, f"pca_{self.name}_umap_plot.png")
        sc.pl.umap(umap_data, color=["celltype"], save=filename)
        print(f"UMAP plot saved as {filename}")
