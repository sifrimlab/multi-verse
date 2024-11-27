import numpy as np
import scanpy as sc
import anndata as ad
import mudata as md
import muon as mu
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
import re

from config import load_config 

class ModelFactory:
    """ 
    Other classes will inherit initial attributes of this class (config_file, dataset, dataset_name, ...)
    List of functions in each model is same as ModelFactory but how it works is different for each model
    """
    def __init__(self, dataset, dataset_name: str, model_name:str = "", outdir="./outputs", config_path: str="./config.json"):
        self.model_params = load_config(config_path=config_path).get("model")
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.outdir = outdir
        self.model_name = model_name
        # Embeddings of the latent space
        self.latent = np.zeros((2,2))
        self.latent_filepath = None
    
    def to(self):
        print("Setting device for model CPU or GPU.")

    def train(self):
        print("Training the model.")
    
    def save_latent(self):
        print("Saving latent representation of the model.")
    
    def load_latent(self):
        print("Loading the available latent representation.")
    
    def umap(self):
        print("Create umap for presentation.")
    

class PCA_Model(ModelFactory):
    """PCA implementation"""

    def __init__(self, dataset: ad.AnnData, dataset_name, config_path: str="./config.json"):
        """
        Initialize the PCA model with the specified parameters.
        Input data is AnnData object that was concatenated of multiple modality
        """
        print("Initializing PCA Model")

        super().__init__(dataset, dataset_name, config_path=config_path,model_name="pca")
        pca_params= self.model_params.get(self.model_name)

        # PCA parameters from config file
        self.n_components = pca_params.get("n_components")
        self.device = pca_params.get("device")
        self.gpu_mode = False # Cpu default mode
        self.umap_random_state = pca_params.get("umap_random_state")
        self.umap_color_type = pca_params.get("umap_color_type")  # Default ass 'cell_type'

        # Output for PCA model is in ./outputs/pca_output
        self.output_dir = os.path.join(self.outdir, "pca_output")
        self.latent_filepath = os.path.join(self.output_dir, f"pca_{self.dataset_name}.h5ad")  # Create h5ad filename for saving latent space later
        os.makedirs(self.output_dir, exist_ok=True)
        
        # For demonstration, we'll assume PCA is just a placeholder here
        # self.pca = PCA(n_components=self.n_components)
        print(f"PCA initialized with {self.dataset_name}, {self.n_components} components.")

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

        # Fit PCA and transform the data with scanpy
        sc.pp.pca(data=self.dataset, n_comps=self.n_components, use_highly_variable=True)
        
        self.latent = self.dataset.obsm["X_pca"]

        print(f"Training PCA completed with {self.n_components} components")

    def save_latent(self):
        """Save the PCA latent representations."""
        print("Saving PCA latent embeddings")

        # Modify AnnData object for scib evaluation
        self.dataset.obs["batch"] = "batch_1"
        self.dataset.write(self.latent_filepath)
        print(f"Latent data saved to {self.latent_filepath}")

    def load_latent(self) -> ad.AnnData:
        """Load latent data from a saved file."""
        print(f"Loading latent data from {self.latent_filepath}")

        if os.path.exists(self.latent_filepath):
            self.dataset = sc.read_h5ad(self.latent_filepath)
            print("Latent data loaded successfully.")
        else:
            print(f"File not found: {self.latent_filepath}")
        return self.dataset

    def umap(self):
        """Generate UMAP visualization using PCA embeddings for all modalities."""
        print("Generating UMAP with PCA embeddings for all modalities")
        
        # Use the PCA latent representation for UMAP
        sc.pp.neighbors(self.dataset, use_rep="X_pca", random_state=self.umap_random_state)
        sc.tl.umap(self.dataset, random_state=self.umap_random_state)
        # Save UMAP representation in `obsm`
        self.dataset.obsm["X_pca_umap"] = self.dataset.obsm["X_umap"].copy()

        # Plotting UMAP and saving the figure
        sc.settings.figdir = self.output_dir
        umap_filename = f"_pca_{self.dataset_name}_plot.png"
        sc.pl.umap(self.dataset, color=self.umap_color_type, save=umap_filename)

        print(f"UMAP plot for {self.model_name} {self.dataset_name} saved as {umap_filename}")


class MOFA_Model(ModelFactory):
    """MOFA+ Model implementation"""
    mu.set_options(display_style = "html", display_html_expand = 0b000)

    def __init__(self, dataset: md.MuData, dataset_name, config_path: str="./config.json"):
        """
        Initialize the MOFA model with the specified parameters.
        Input data is MuData object that contains multiple modality
        """
        print("Initializing MOFA+ Model")
        
        super().__init__(dataset, dataset_name, config_path=config_path, model_name="mofa+")
        mofa_params= self.model_params.get(self.model_name)

        # MOFA+ parameters from config file
        self.device = mofa_params.get("device")
        self.device = False
        self.n_factors = mofa_params.get("n_factors")
        self.n_iteration = mofa_params.get("n_iteration")
        self.umap_random_state=mofa_params.get("umap_random_state")
        self.umap_color_type=mofa_params.get("umap_color_type")

        if self.umap_color_type not in self.dataset.obs:    
            print(f"Warning: '{self.umap_color_type}' not found in dataset. Defaulting to None for coloring.")
            self.umap_color_type = None  # Fallback to None if not found


        # Output for MOFA+ model is in ./outputs/mofa_output
        self.output_dir = os.path.join(self.outdir, "mofa_output")
        self.latent_filepath = os.path.join(self.output_dir, f"mofa_{self.dataset_name}.h5ad")  # Create h5ad filename for saving latent space later
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"MOFA+ initialized with {self.dataset_name}, {self.n_factors} factors to be trained with.")

    def to(self):
        """
        Method to set GPU or CPU mode for MOFA+.
        """
        if self.device !='cpu':
            self.gpu_mode = True
        else:
            self.gpu_mode = False
        print(f"Switching to {self.gpu_mode} mode")

    def train(self):
        """
        Train the MOFA model.
        """
        print("Training MOFA+ Model")
        try:
            mu.tl.mofa(data=self.dataset, n_factors=self.n_factors, gpu_mode=self.gpu_mode)
            print(f"MOFA+ training completed with {self.n_factors} factors")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        """
        Save the latent space embeddings from the trained MOFA model.
        """
        print("Model saved at: {self.outfilepath} during training")

    def load_latent(self):
        """
        Load latent data from the saved MOFA+ HDF5 file.
        """
        try:
            # Save the latent embeddings to the dataset object
            print("Saving MOFA+ latent embeddings as .h5ad file to dataset...")
            # Modify AnnData object for scib evaluation
            metadata = {
                'obs': self.dataset.obs.copy(),  # Observation metadata
                'var': self.dataset.var.copy(),  # Variable metadata
                'uns': self.dataset.uns.copy(),  # Unstructured metadata
                'obsm': self.dataset.obsm.copy(), # Observation matrices
                'varm': self.dataset.varm.copy()  # Variable matrices
            }
            adata = ad.AnnData(None, **metadata) 
            pattern = r".*:cell_type"  # Only keep "cell_type" in .obs and not "rna:cell_type" or "atac:cell_type"
            columns_to_drop = [col for col in adata.obs.columns if re.match(pattern, col)]
            adata.obs.drop(columns=columns_to_drop, inplace=True)

            adata.obs["batch"] = "batch_1"
            adata.write(self.latent_filepath)
            print(f"Latent data saved to {self.latent_filepath}")
        except Exception as e:
            print(f"Error loading latent data: {e}")
            raise

    def load_latent(self):
        """Load latent data from saved .h5ad files."""
        print(f"Loading latent data from {self.latent_filepath}")
        
        if os.path.exists(self.latent_filepath):
            self.dataset = sc.read_h5ad(self.latent_filepath)
            print("Latent data loaded successfully.")
        else:
            print(f"File not found: {self.latent_filepath}")
        return self.dataset


    def umap(self):
        """Generate UMAP visualization."""
        print("Generating UMAP with MOFA embeddings")
        sc.pp.neighbors(self.dataset, use_rep="X_mofa", random_state=self.umap_random_state)
        sc.tl.umap(self.dataset, random_state=self.umap_random_state)

        # Plotting UMAP and saving the figure
        sc.settings.figdir = self.output_dir
        filename = f"_mofa_{self.dataset_name}_plot.png"
        sc.pl.umap(self.dataset, color=self.umap_color_type, save=filename)
        print(f"UMAP plot saved as {filename}")


class MultiVI_Model(ModelFactory):
    """MultiVI Model implementation."""
    
    def __init__(self, dataset: ad.AnnData, dataset_name, config_path: str="./config.json"):
        """
        Initialize the MultiVi model with the specified parameters.
        Input data is AnnData object that was concatenated of multiple modality
        """
        print("Initializing MultiVI Model")

        super().__init__(dataset, dataset_name, config_path=config_path, model_name="multivi")
        multivi_params= self.model_params.get(self.model_name)

        # Multivi parameters from config file
        self.device = multivi_params.get("device")
        self.max_epochs = multivi_params.get("max_epochs")
        self.learning_rate = multivi_params.get("learning_rate")
        self.latent_key = "X_multivi"
        self.umap_color_type = multivi_params.get("umap_color_type")

        if self.umap_color_type not in self.dataset.obs:    
            print(f"Warning: '{self.umap_color_type}' not found in dataset. Defaulting to None for coloring.")
            self.umap_color_type = None  # Fallback to None if not found

        # Output for Multivi model is in ./outputs/multivi_output
        self.output_dir = os.path.join(self.outdir, "multivi_output")
        self.latent_filepath = os.path.join(self.output_dir, f"multivis_{self.dataset_name}.h5ad")  # Create h5ad filename for saving latent space later
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up data for MultiVI model
        self.dataset = self.dataset[:, self.dataset.var["feature_types"].argsort()].copy()
        scvi.model.MULTIVI.setup_anndata(self.dataset, protein_expression_obsm_key=None)
        self.model = scvi.model.MULTIVI(
            self.dataset,
            n_genes=(self.dataset.var["feature_types"] == "Gene Expression").sum(),
            n_regions=(self.dataset.var["feature_types"] == "Peaks").sum(),
            )
            
    def to(self):
        """
        Method to set GPU or CPU mode for MOFA+.
        """
        try:
            if self.device !='cpu':
                print(f"Moving MultiVI model to {self.device}")
                self.model.to_device(self.device)
                print(f"Model successfully moved to {self.device}")
            else:
                self.model.to_device(self.device)
                print(f"Recommend to use GPU instead of {self.device}")
        except Exception as e:
            print(f"Invalid device '{self.device}' specified. Use 'cpu' or 'gpu'.")
        
    def train(self):
        print("Training MultiVI Model")
        try:
            self.to()
            self.model.train()
            self.dataset.obsm[self.latent_key] = self.model.get_latent_representation()
            print(f"Multivi training completed.")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        """Save latent data generated with the MultiVI model."""
        print("Saving latent data")
        try:
            self.dataset.obsm[self.latent_key] = self.model.get_latent_representation()
            # Modify AnnData object for scib evaluation
            self.dataset.obs["batch"] = "batch_1"
            self.dataset.write(self.latent_filepath)
            #self.model.save(self.output_dir)

            print(f"MultiVI model for dataset {self.dataset_name} was saved as {self.latent_filepath}")
        except Exception as e:
            print(f"Error saving latent data: {e}")

    def load_latent(self) -> ad.AnnData:
        """Load latent data from saved files."""
        print(f"Loading latent data from {self.latent_filepath}")
        
        # Check if the file exists
        if os.path.exists(self.latent_filepath):
            try:
                self.dataset = sc.read_h5ad(self.latent_filepath)
                print("Latent data loaded successfully.")
            except Exception as e:
                print(f"Error loading latent data: {e}")
        else:
            print(f"File not found: {self.output_dir}, multivi_{self.dataset_name}.h5ad")
        return self.dataset

    def umap(self):
        """Generate UMAP visualization."""
        print("Generating UMAP plot")
        try:
            sc.settings.figdir = self.output_dir
            umap_filename = f"_multivi_{self.dataset_name}_plot.png"
            sc.pp.neighbors(self.dataset, use_rep=self.latent_key)
            sc.tl.umap(self.dataset)
            sc.pl.umap(self.dataset, color=self.umap_color_type, 
                       save=umap_filename)
            print(f"A UMAP plot for MultiVI model with dataset {self.dataset_name} "\
                  f"was succesfully generated and saved as multivi_{self.dataset_name}_umap_plot.png")

        except Exception as e:
            print(f"Error generating UMAP: {e}")



class Mowgli_Model(ModelFactory):
    """Mowgli model implementation."""
    
    def __init__(self, dataset, dataset_name, config_path: str="./config.json"):
        """Initialize the Mowgli model with the specified parameters."""

        print("Initializing Mowgli Model")
        
        super().__init__(dataset, dataset_name, config_path=config_path, model_name="mowgli")
        mowgli_params = self.model_params.get(self.model_name)

        self.dataset = dataset
        self.dataset_name = dataset_name
        self.device = mowgli_params.get("device")
        self.latent_dimensions = mowgli_params.get("latent_dimensions")
        self.learning_rate = mowgli_params.get("learning_rate")
        self.umap_num_neighbors = mowgli_params.get("umap_num_neighbors")
        self.umap_size = mowgli_params.get("umap_size")
        self.umap_alpha = mowgli_params.get("umap_alpha")

        # Create the model instance during initialization
        self.model = mowgli.models.MowgliModel(latent_dim=self.latent_dimensions)
        
        # Ensure output directory exists
        self.latent_filepath = None
        self.output_dir = os.path.join("outputs", "mowgli_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def to(self):
        """
        Method to set GPU or CPU mode for MOFA+.
        """
        try:
            print(f"Moving Mowgli model to {self.device}")
            self.device = torch.device(self.device)
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
            print("Training completed.")

            # Check if W is stored in the dataset after training
            # print("Dataset obsm keys after training:", self.dataset.obsm.keys())
            # print("Shape of dataset embeddings (if found):", self.dataset.obsm.get("W_OT", "Not Found").shape)
   
            # Debug: Check the shape of W
            # if hasattr(self.model, "W"):
            #     print("Shape of self.model.W:", self.model.W.shape)
            # else:
            #     raise ValueError("self.model.W is not defined after training.")

            # Transpose the embeddings if needed
            if self.model.W.shape[0] == self.latent_dimensions and self.model.W.shape[1] == self.dataset.n_obs:
                W_corrected = self.model.W.T  # Transpose to match (n_obs, latent_dimensions)
                # print("Transposed embeddings shape:", W_corrected.shape)
            else:
                W_corrected = self.model.W  # Use directly if the shape is already correct

            # Convert Torch Tensor to NumPy Array
            if isinstance(W_corrected, torch.Tensor):
                W_corrected = W_corrected.cpu().numpy()  # Convert to NumPy array (move to CPU if needed)

            # Assign embeddings to obsm
            if W_corrected.shape[0] == self.dataset.n_obs and W_corrected.shape[1] == self.latent_dimensions:
                self.dataset.obsm["W_OT"] = W_corrected
                # print("Assigned embeddings to W_OT:", self.dataset.obsm["W_OT"].shape)
            else:
                raise ValueError(
                    f"Embeddings shape mismatch after transpose: Expected ({self.dataset.n_obs}, {self.latent_dimensions}), "
                    f"but got {W_corrected.shape}."
                )

            # print("Overall Loss:", self.model.losses)

        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        """Save latent data generated with the Mowgli model in .h5ad format."""
        print("Saving latent data")

        # Create AnnData object to store embeddings and losses
        adata = sc.AnnData(X=self.dataset.obsm["W_OT"])  # W_OT contains embeddings
        adata.uns["loss_overall"] = self.model.losses  # Save overall loss
        adata.uns["loss_w"] = self.model.losses_w  # Save W-specific loss
        adata.uns["loss_h"] = self.model.losses_h  # Save H-specific loss
        
        # Add modality-specific H_ matrices to the AnnData object
        for mod in self.dataset.mod:
            if f"H_OT" in self.dataset[mod].uns:
                adata.uns[f"H_{mod}"] = self.dataset[mod].uns["H_OT"]
            
        # Define the output path for the saved .h5ad file
        output_path = os.path.join(self.output_dir, f"mowgli_latent_{self.dataset_name}.h5ad")
        self.latent_filepath = output_path
        
        # Save the AnnData object to .h5ad format
        try:
            adata.write(self.latent_filepath)
            print(f"Latent data saved to {self.latent_filepath}")
        except Exception as e:
            print(f"Error saving latent data: {e}")

    def load_latent(self):
        """Load latent data from saved .h5ad file."""
        print("Loading latent data")
        
        try:
            # Load the latent data from the .h5ad file
            adata = sc.read(self.latent_filepath)
            
            # Restore the latent embeddings
            self.dataset.obsm["W_mowgli"] = adata.X
            
            # Restore the modality-specific H_ matrices (for each modality)
            for mod in self.dataset.mod:
                if f"H_{mod}" in adata.uns:
                    self.dataset[mod].uns["H_OT"] = adata.uns[f"H_{mod}"]
            
            # Restore losses (if necessary)
            if "loss_overall" in adata.uns:
                self.model.losses = adata.uns["loss_overall"]
            if "loss_w" in adata.uns:
                self.model.losses_w = adata.uns["loss_w"]
            if "loss_h" in adata.uns:
                self.model.losses_h = adata.uns["loss_h"]
            
            print("Latent data loaded successfully.")
            return adata
            
        except FileNotFoundError:
            print(f"File not found: {self.latent_filepath}")
        except Exception as e:
            print(f"Error loading latent data: {e}")

    def umap(self):
        """Generate UMAP visualization."""
        print("Generating UMAP plot")
        try:
            self.dataset.uns = {}
            # Load and assign embeddings for visualization
            # Ensure embeddings exist
            if "W_OT" not in self.dataset.obsm:
                raise ValueError("Embeddings not found in obsm['W_OT']. Run train() first.")

            # Assign embeddings for visualization
            self.dataset.obsm["X_mowgli"] = self.dataset.obsm["W_OT"]

            # Set Scanpy figure directory
            sc.settings.figdir = self.output_dir

            # UMAP
            sc.pp.neighbors(self.dataset, use_rep="X_mowgli", n_neighbors=self.umap_num_neighbors)
            sc.tl.umap(self.dataset)
            umap_plot_path = f"mowgli_{self.dataset_name}_umap_plot.png"
            sc.pl.umap(self.dataset, size=self.umap_size, alpha=self.umap_alpha, save=umap_plot_path)
            print(f"UMAP plot saved to {os.path.join(self.output_dir, umap_plot_path)}")


            # Leiden clustering
            sc.tl.leiden(self.dataset)
            leiden_plot_path = f"mowgli_{self.dataset_name}_leiden_plot.png"
            sc.pl.embedding(self.dataset, "X_umap", color=['leiden'], save=leiden_plot_path)
            print(f"Leiden plot saved to {os.path.join(self.output_dir, leiden_plot_path)}")


        except Exception as e:
            print(f"Error generating UMAP: {e}")
            raise
      