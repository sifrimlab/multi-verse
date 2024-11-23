import numpy as np
import scanpy as sc
import anndata as ad
import muon as mu
from sklearn.decomposition import PCA
import os
import mowgli
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import scanpy as sc
import muon as mu
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import scvi

from config import load_config 
   

class PCA_Model:
    """PCA implementation"""

    def __init__(self, dataset):
        """Initialize the PCA model with the specified parameters."""

        pca_params= load_config()["models"]["pca"].get("model_params")

        print("Initializing PCA Model")
        self.data_dir = pca_params.get("data_dir")
        self.dataset = dataset
        self.n_components = pca_params.get("n_components")
        self.name = pca_params.get("name")
        self.gpu_mode = False  # Default to CPU mode
        self.device = pca_params.get("device")
        self.latent_filepath = None
        self.umap_random_state = pca_params.get("umap_random_state")
        self.umap_filename = pca_params.get("umap_filename")
        self.umap_color_type = pca_params.get("umap_color_type")
        self.output_dir = os.path.join(self.data_dir, "pca_output")

        # For demonstration, we'll assume PCA is just a placeholder here
        # self.pca = PCA(n_components=self.n_components)
        print(f"PCA initialized with {self.dataset}, {self.n_components} components.")

    def to(self):
        """
        Method to set GPU or CPU mode.
        """
        if  self.device != 'cpu':
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
        output_path = os.path.join(self.output_dir, f"pca_{self.name}.h5ad")
        self.dataset.write(output_path)
        print(f"Latent data saved to {output_path}")
        # Store the path for loading later (e.g., assign it to self.latent_filepath)
        self.latent_filepath = output_path

    def load_latent(self):
        """Load latent data from a saved file."""
        print(f"Loading latent data from {self.latent_filepath}")
        if os.path.exists(self.latent_filepath):
            self.dataset = mu.read(self.latent_filepath)
            print("Latent data loaded successfully.")
        else:
            print(f"File not found: {self.latent_filepath}")

    def umap(self):
        """Generate UMAP visualization using PCA embeddings for all modalities."""
        print("Generating UMAP with PCA embeddings for all modalities")
        
        for modality in self.dataset.mod.keys():
            print(f"Processing modality: {modality}")
            
            # Use the PCA latent representation for UMAP
            sc.pp.neighbors(self.dataset[modality], use_rep="X_pca")
            sc.tl.umap(self.dataset[modality], random_state=self.umap_random_state)
            
            # Save UMAP representation in `obsm`
            self.dataset[modality].obsm["X_pca_umap"] = self.dataset[modality].obsm["X_umap"]

            # Plotting UMAP and saving the figure
            if not self.umap_filename:
                self.umap_filename = os.path.join(self.output_dir, f"pca_{modality}_umap_plot.png")
            sc.pl.umap(self.dataset[modality], color=self.umap_color_type, save=self.umap_filename)
            print(f"UMAP plot for {modality} saved as {self.umap_filename}")


class MOFA_Model:
    """MOFA+ Model implementation"""
    mu.set_options(display_style = "html", display_html_expand = 0b000)

    def __init__(self, mofa_params):
        """Initialize the MOFA model with the specified parameters."""
        mofa_params= load_config()["models"]["mofa"].get("model_params")

        self.data_dir = mofa_params.get("data_dir")
        self.dataset = mofa_params.get("dataset")
        self.name = mofa_params.get("datasetName")
        self.gpu_mode = mofa_params.get("gpu_mode")
        self.device = mofa_params.get("device")
        self.n_factors = mofa_params.get("n_factors")
        self.outfilepath = mofa_params.get("outfilepath")
        self.latent_filepath = mofa_params.get("latent_filepath")
        self.umap_random_state=mofa_params.get("umap_random_state")
        self.umap_use_representation=mofa_params.get("umap_use_representation")
        self.umap_color_type=mofa_params.get("umap_color_type")
        self.output_dir = os.path.join(self.data_dir, "mofa_output")

        print("Initializing MOFA+ Model")

    def to(self):
        """
        Method to set GPU or CPU mode for MOFA+.
        """
        if self.device=='cpu':
            self.gpu_mode = False
        if self.device=='gpu':
            self.gpu_mode = True
        else:
            print(f"Invalid device '{self.device}' specified. Use 'cpu' or 'gpu'.")
            return  # Exit early if the device is invalid
        
        print(f"Switching to {self.device} mode")

    def train(self):
        """
        Train the MOFA model.
        """
        print("Training MOFA+ Model")
        if not self.outfilepath:
            self.outfilepath = os.path.join(self.output_dir, f"mofa_{self.name}.hdf5")
        try:
            mu.tl.mofa(self.dataset, n_factors=self.n_factors, 
                       outfile=self.outfilepath, gpu_mode=self.gpu_mode)
            print(f"Model saved at: {self.outfilepath}")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        """
        Save the latent space embeddings from the trained MOFA model.
        """
        try:
            # Save the latent embeddings to the dataset object
            print("Saving latent embeddings to dataset object...")
            
            # Ensure the necessary embeddings and factors are in the dataset
            X_mofa = self.dataset.obsm.get('X_mofa', [])
            LFs_mofa = self.dataset.varm.get('LFs', [])
            
            # Create a dictionary to store the latent embeddings and factors
            latent_data = {
                'X_mofa': X_mofa,
                'LFs_mofa': LFs_mofa
            }
            
            # Define the output path for the saved .npy file
            output_path = os.path.join(self.output_dir, f"mofa_{self.name}.npy")
            
            # Save the latent data to a .npy file (using np.save)
            np.save(output_path, latent_data)
            
            print(f"Latent data saved to {output_path}")

        except Exception as e:
            print(f"Error saving latent embeddings to dataset: {e}")

    def load_latent(self):
        """Load latent data from saved .npy files."""
        print(f"Loading latent data from {self.latent_filepath}")
        
        if os.path.exists(self.latent_filepath):
            try:
                # Load the latent data from the .npy file (using np.load)
                latent_data = np.load(self.latent_filepath, allow_pickle=True).item()  # `.item()` to get the dictionary
                
                # Restore the saved latent embeddings and factors into the dataset object
                if 'X_mofa' in latent_data:
                    self.dataset.obsm['X_mofa'] = latent_data['X_mofa']
                if 'LFs_mofa' in latent_data:
                    self.dataset.varm['LFs'] = latent_data['LFs_mofa']
                    
                print("Latent data loaded successfully.")
            
            except Exception as e:
                print(f"Error loading latent data: {e}")
        else:
            print(f"File not found: {self.latent_filepath}")

    def umap(self):
        """Generate UMAP visualization."""
        print("Generating UMAP with MOFA embeddings")
        sc.pp.neighbors(self.dataset, use_rep=self.umap_use_representation)
        sc.tl.umap(self.dataset, random_state=self.umap_random_state)

        # Plotting UMAP and saving the figure
        filename = os.path.join(self.output_dir, f"mofa_{self.name}_umap_plot.png") # something off with the filename - gives errors
        sc.pl.umap(self.dataset, color=self.umap_color_type, save=filename)
        print(f"UMAP plot saved as {filename}")


class Mowgli_Model:
    """Mowgli model implementation."""
    
    def __init__(self, mowgli_params):
        """Initialize the Mowgli model with the specified parameters."""

        print("Initializing Mowgli Model")

        mowgli_params= load_config()["models"]["mowgli"].get("model_params")

        self.data_dir = mowgli_params.get("data_dir")
        self.dataset = mowgli_params.get("dataset")
        self.name = mowgli_params.get("datasetName")
        self.device = mowgli_params.get("device")
        self.latent_dimensions = mowgli_params.get("latent_dimensions")
        self.umap_num_neighbors = mowgli_params.get("umap_num_neighbors")
        self.umap_size = mowgli_params.get("umap_size")
        self.umap_alpha = mowgli_params.get("umap_alpha")
        self.umap_filename=mowgli_params.get("umap_filename") 

        # Create the model instance during initialization
        self.model = mowgli.models.MowgliModel(latent_dim=self.latent_dimensions)
        
        # Ensure output directory exists
        self.output_dir = os.path.join(self.data_dir, "mowgli_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def to(self):
        """
        Method to set GPU or CPU mode for MOFA+.
        """
        try:
            print(f"Moving Mowgli model to {self.device}")
            torch_device = torch.device(self.device)
            self.model.to_device(self.device)
            print(f"Mowgli model successfully moved to {self.device}")
        except Exception as e:
            print(f"Invalid device '{self.device}' specified. Use 'cpu' or 'gpu'.")
            raise

    def train(self):
        """Train the Mowgli model."""
        print("Training Mowgli Model")
        try:
            self.model.train(
                self.dataset,
                device=self.device,
                optim_name='sgd',
                lr=self.learning_rate,
                tol_inner=1e-5
            )
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def evaluate(self):
        """Evaluate the Mowgli model."""
        print("Evaluating Mowgli Model")
        # Add evaluation logic here (e.g., metrics calculation)
        raise NotImplementedError("Evaluation method not implemented yet.")


    def save_latent(self):
        """Save latent data generated with the Mowgli model."""
        print("Saving latent data")
        try:
            np.save(
                os.path.join(self.output_dir, f"mowgli_{self.name}.npy"),
                {
                    "W": self.dataset.obsm["W_OT"],
                    **{"H_" + mod: self.dataset[mod].uns["H_OT"] for mod in self.dataset.mod},
                    "loss_overall": self.model.losses,
                    "loss_w": self.model.losses_w,
                    "loss_h": self.model.losses_h
                }
            )

        except Exception as e:
            print(f"Error saving latent data: {e}")

    def load_latent(self):
        """Load latent data from saved files."""
        print("Loading latent data")
        try:
            file_path = os.path.join(self.output_dir, f"mowgli_{self.name}.npy")
            mowgli_data = np.load(file_path, allow_pickle=True).item()
            self.dataset.obsm["W_mowgli"] = mowgli_data["W"]
            self.dataset.uns = {}
            return mowgli_data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading latent data: {e}")

    def umap(self):
        """Generate UMAP visualization."""
        print("Generating UMAP plot")
        try:
            sc.pp.neighbors(self.dataset, use_rep="X_mowgli", n_neighbors=self.umap_num_neighbors)
            sc.tl.umap(self.dataset)
            sc.pl.umap(self.dataset, size=self.umap_size, alpha=self.umap_alpha)
        
            # If filename is not provided, use default that includes self.name
            if self.umap_filename is None:
                self.umap_filename = f"mowgli_{self.name}_umap_plot.png"
        
            # Save the plot
            plt.savefig(os.path.join(self.output_dir, self.umap_filename))
            plt.close()
        
            print(f"A UMAP plot for Mowgli model with dataset {self.name} was successfully 
                  generated and saved as {self.umap_filename}")

        except Exception as e:
            print(f"Error generating UMAP: {e}")


class MultiVI_Model:
    """MultiVI Model implementation."""
    
    def __init__(self, multivi_params):
        """Initialize the MultiVi model with the specified parameters."""

        print("Initializing MultiVI Model")

        multivi_params= load_config()["models"]["multivi"].get("model_params")

        self.data_dir = multivi_params.get("data_dir")
        self.dataset = multivi_params.get("dataset")
        self.adata_rna = multivi_params.get("adata_rna") 
        self.adata_atac = multivi_params.get("adata_atac")
        self.latent_key = multivi_params.get("latent_key")
        self.name = multivi_params.get("datasetName")
        self.join = multivi_params.get("join")
        self.axis = multivi_params.get("axis")
        self.label = multivi_params.get("label")
        self.join = multivi_params.get("join")
        self.umap_filename = multivi_params.get("umap_filename")
        self.umap_min_dist = multivi_params.get("umap_min_dist")
        self.umap_color_type = multivi_params.get("umap_color_type")
        self.output_dir = os.path.join(self.data_dir, "multivi_output")


        self.adata_mvi = ad.concat([self.adata_rna, self.adata_atac], 
                                   join=self.join, axis=self.axis, label=self.label)

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
            self.adata_mvi.write(self.output_dir+f"multivi_{self.name}.txt")

            print(f"MultiVI model for dataset {self.name} was saved as multivi_{self.dataset}.txt")
        except Exception as e:
            print(f"Error saving latent data: {e}")

    def load_latent(self):
        """Load latent data from saved files."""
        print(f"Loading latent data from {self.output_dir}multivi_{self.name}.txt")
        
        # Check if the file exists
        if os.path.exists(self.output_dir + f"multivi_{self.name}.txt"):
            try:
                # Assuming the file is a tab-delimited text file, you can use pandas or numpy to load it.
                # We use pandas here because it's commonly used for reading tabular data into AnnData objects
                import pandas as pd
                
                # Load the latent data into a DataFrame first
                latent_data = pd.read_csv(self.output_dir + f"multivi_{self.name}.txt", sep="\t", index_col=0)
                
                # Store the latent representation in the adata_mvi object's `obsm`
                self.adata_mvi.obsm[self.latent_key] = latent_data.values
                
                print("Latent data loaded successfully.")
            except Exception as e:
                print(f"Error loading latent data: {e}")
        else:
            print(f"File not found: {self.output_dir}multivi_{self.name}.txt")

    def umap(self):
        """Generate UMAP visualization."""
        print("Generating UMAP plot")
        try:
            sc.pp.neighbors(self.adata_mvi, use_rep=self.latent_key)
            sc.tl.umap(self.adata_mvi, min_dist=self.umap_min_dist)
            sc.pl.umap(self.adata_mvi, color=self.umap_color_type, 
                       save=(self.output_dir+f"multivi_{self.name}_umap_plot.png"))
            print(f"A UMAP plot for MultiVI model with dataset {self.name} 
                  was succesfully generated and saved as multivi_{self.name}_umap_plot.png")

        except Exception as e:
            print(f"Error generating UMAP: {e}")
