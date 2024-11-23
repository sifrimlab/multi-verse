import numpy as np
import scanpy as sc
import anndata as ad
import muon as mu
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

        self.pca = PCA(n_components=self.n_components)

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

    def umap(self, random_state=1, filename=None, color_type='celltype'):
        """Generate UMAP visualization using PCA embeddings for all modalities."""
        print("Generating UMAP with PCA embeddings for all modalities")
        
        for modality in self.dataset.mod.keys():
            print(f"Processing modality: {modality}")
            
            # Use the PCA latent representation for UMAP
            sc.pp.neighbors(self.dataset[modality], use_rep="X_pca")
            sc.tl.umap(self.dataset[modality], random_state=random_state)
            
            # Save UMAP representation in `obsm`
            self.dataset[modality].obsm["X_pca_umap"] = self.dataset[modality].obsm["X_umap"]

            # Plotting UMAP and saving the figure
            if not filename:
                filename = os.path.join(self.data_dir, f"pca_{modality}_umap_plot.png")
            sc.pl.umap(self.dataset[modality], color=color_type, save=filename)
            print(f"UMAP plot for {modality} saved as {filename}")

