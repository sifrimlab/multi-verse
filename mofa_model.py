import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import muon as mu

mu.set_options(display_style = "html", display_html_expand = 0b000)

class MOFA_Model:
    """MOFA+ Model implementation"""
    
    def __init__(self, data_dir, dataset, name="datasetName"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.name = name
        
        print("Initializing MOFA+ Model")

        
    def train(self, n_factors=20, outfile=None, gpu_mode=True):
        print("Training MOFA+ Model")
        print("Training MOFA+ Model")
        if not outfile:
            outfile = os.path.join(self.data_dir, f"mofa_{self.name}.hdf5")
        try:
            mu.tl.mofa(self.dataset, n_factors=n_factors, outfile=outfile, gpu_mode=gpu_mode)
            print(f"Model saved at: {outfile}")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        """
        Save the latent space embeddings from the trained MOFA model.
        """
        print("Saving latent embeddings")
        self.dataset.obsm[f'X_mofa_{self.name}'] = self.dataset.obsm.get('X_mofa')
        self.dataset.varm[f'LFs_mofa_{self.name}'] = self.dataset.varm.get('LFs')

        # Save the updated dataset to a file
        output_path = os.path.join(self.data_dir, f"mofa_{self.dataset}.h5ad")
        self.dataset.write(output_path)

        print(f"Latent data saved to {output_path}")

    def load_latent(self, filepath):
        """Load latent data from saved files."""
        print(f"Loading latent data from {filepath}")
        if os.path.exists(filepath):
            self.dataset = mu.read(filepath)
            print("Latent data loaded successfully.")
        else:
            print(f"File not found: {filepath}")

    def umap(self, random_state=1, filename=None):
        """Generate UMAP visualization."""
        print("Generating UMAP with MOFA embeddings")
        sc.pp.neighbors(self.dataset, use_rep="X_mofa")
        sc.tl.umap(self.dataset, random_state=random_state)
        self.dataset.obsm["X_mofa_umap"] = self.dataset.obsm["X_umap"]

        # Plotting UMAP and saving the figure
        if not filename:
            filename = os.path.join(self.data_dir, f"mofa_{self.name}_umap_plot.png")
        mu.pl.embedding(self.dataset, basis="X_mofa_umap", color=["rna:celltype", "atac:celltype"], save=filename)
        print(f"UMAP plot saved as {filename}")