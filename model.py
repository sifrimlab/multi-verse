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
import pandas as pd
from sklearn.metrics import silhouette_score

from config import load_config 

class ModelFactory:
    """ 
    Other classes will inherit initial attributes of this class (config_file, dataset, dataset_name, ...)
    List of functions in each model is same as ModelFactory but how it works is different for each model
    """
    def __init__(self, dataset, dataset_name: str, 
                 model_name:str = "", outdir="./outputs", 
                 config_path: str="./config.json", is_gridsearch=False):
        self.model_params = load_config(config_path=config_path).get("model")
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.outdir = outdir
        self.model_name = model_name
        # Embeddings of the latent space
        self.latent = np.zeros((2,2))
        self.latent_filepath = None

        self.is_grid_search = is_gridsearch  # Flag for grid search runs
        self.base_filename = f"{self.model_name}_{self.dataset_name}"
        self.output_dir = os.path.join(self.outdir, f"{self.model_name}_output") # Default
        os.makedirs(self.output_dir, exist_ok=True)

    def update_output_dir(self):
        if self.is_grid_search:
            self.output_dir = os.path.join(self.outdir, "gridsearch_output")
            self.base_filename += "_gridsearch"
            self.latent_filepath = os.path.join(self.output_dir, f"{self.base_filename}.h5ad")
            self.umap_filename = os.path.join(self.output_dir, f"_{self.base_filename}.png")
        else:
            self.output_dir = os.path.join(self.outdir, f"{self.model_name}_output")

        os.makedirs(self.output_dir, exist_ok=True)

    def update_parameters(self, **kwargs):
        """
        Updates the model parameters.
        Args:
            **kwargs: Keyword arguments with parameter names and their new values.
                     Example: update_parameters(n_factors=10, n_iteration=500)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Handle invalid parameter names if necessary
                print(f"Warning: Invalid parameter name '{key}'")

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
    
    def evaluate_model(self):
        print("Write evaluation metrics for thi specific model for gridsearch")

class PCA_Model(ModelFactory):
    """PCA implementation"""

    def __init__(self, dataset: ad.AnnData, dataset_name, config_path: str="./config.json", is_gridsearch=False):
        """
        Initialize the PCA model with the specified parameters.
        Input data is AnnData object that was concatenated of multiple modality
        """
        print("Initializing PCA Model")

        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="pca", is_gridsearch=is_gridsearch)
        pca_params= self.model_params.get(self.model_name)

        # PCA parameters from config file
        self.n_components = pca_params.get("n_components")
        self.device = pca_params.get("device")
        self.gpu_mode = False # Cpu default mode
        self.umap_random_state = pca_params.get("umap_random_state")
        self.umap_color_type = pca_params.get("umap_color_type")  # Default ass 'cell_type'

        # Output for PCA model is in ./outputs
        self.latent_filepath = os.path.join(self.output_dir, f"{self.base_filename}.h5ad")
        self.umap_filename = os.path.join(self.output_dir, f"_{self.base_filename}.png")

        print(f"PCA initialized with {self.dataset_name}, {self.n_components} components.")

    def update_output_dir(self):
        super().update_output_dir()

    def update_parameters(self, **kwargs):
        super().update_parameters(**kwargs)

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
        if 'highly_variable' in self.dataset.var.keys():
            sc.pp.pca(data=self.dataset, n_comps=self.n_components, use_highly_variable=True)
        else:
            sc.pp.pca(data=self.dataset, n_comps=self.n_components, use_highly_variable=False)
        # Save latent representation
        self.latent = self.dataset.obsm["X_pca"]

        # Capture variance explained ratio
        self.variance_ratio = self.dataset.uns["pca"]["variance_ratio"]

        print(f"Training PCA completed with {self.n_components} components")
        # print(f"Variance explained by each component: {self.variance_ratio}")
        print(f"Total variance explained: {sum(self.variance_ratio)}")

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
        filename = os.path.basename(self.umap_filename)

        sc.pl.umap(self.dataset, color=self.umap_color_type, save=filename)

        print(f"UMAP plot for {self.model_name} {self.dataset_name} saved as umap{filename}")

    def evaluate_model(self):
        """
        Evaluate the trained PCA model based on variance explained.
        """
        print("Evaluating PCA model...")

        if hasattr(self, "variance_ratio"):
            total_variance = sum(self.variance_ratio)
            print(f"Total Variance Explained: {total_variance}")
            return total_variance
        else:
            raise ValueError("PCA variance ratio not available in the model.")


class MOFA_Model(ModelFactory):
    """MOFA+ Model implementation"""
    mu.set_options(display_style = "html", display_html_expand = 0b000)

    def __init__(self, dataset: md.MuData, dataset_name, config_path: str="./config.json", is_gridsearch=False):
        """
        Initialize the MOFA model with the specified parameters.
        Input data is MuData object that contains multiple modality
        """
        print("Initializing MOFA+ Model")
        
        super().__init__(dataset, dataset_name, config_path=config_path, model_name="mofa", is_gridsearch=is_gridsearch)
    
        mofa_params= self.model_params.get(self.model_name)

        # MOFA+ parameters from config file
        self.device = mofa_params.get("device")
        self.device = False
        self.n_factors = mofa_params.get("n_factors")
        self.n_iteration = mofa_params.get("n_iteration")
        self.umap_random_state=mofa_params.get("umap_random_state")
        self.umap_color_type=mofa_params.get("umap_color_type")

        # Output for MOFA+ model is in ./outputs
        self.latent_filepath = os.path.join(self.output_dir, f"{self.base_filename}.h5ad")
        self.umap_filename = os.path.join(self.output_dir, f"_{self.base_filename}.png")

        print(f"MOFA+ initialized with {self.dataset_name}, {self.n_factors} factors to be trained with.")

    def update_output_dir(self):
        super().update_output_dir()
    
    def update_parameters(self, **kwargs):
        super().update_parameters(**kwargs)

    def _compute_explained_variance(self):
        """
        Compute explained variance for MOFA factors.
        """
        try:
            factors = self.dataset.obsm["X_mofa"]  # Extract latent factors
            total_variance = np.var(self.dataset.X, axis=0).sum()  # Total variance in the dataset
            factor_variances = np.var(factors, axis=0)  # Variance explained by each factor

            explained_variance_ratio = factor_variances / total_variance
            return explained_variance_ratio
        except Exception as e:
            print(f"Error computing explained variance: {e}")
            return []
    
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
            print("MOFA training completed.")
        
            # Debugging output
            # print(f"Keys in dataset.uns['mofa']: {self.dataset.uns.get('mofa', {}).keys()}")
            
            # Compute explained variance if not available
            if "explained_variance" in self.dataset.uns.get("mofa", {}):
                self.explained_variance = self.dataset.uns["mofa"]["explained_variance"]
                print(f"Explained variance per factor: {self.explained_variance}")
            else:
                # Manually calculate explained variance
                self.explained_variance = self._compute_explained_variance()
                print(f"Computed explained variance per factor: {self.explained_variance}")
            
            print(f"Total explained variance: {sum(self.explained_variance)}")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        """
        Save the latent space embeddings from the trained MOFA model.
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
            print(f"Error saving latent embeddings to .h5ad: {e}")
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
        filename = os.path.basename(self.umap_filename) 
        sc.pl.umap(self.dataset, color=self.umap_color_type, save=filename)
        print(f"UMAP plot saved as umap{filename}")

    def evaluate_model(self):
        """
        Evaluate the trained MOFA+ model based on explained variance.
        """
        if hasattr(self, "explained_variance"):
            total_variance = sum(self.explained_variance)
            print(f"Total Explained Variance (MOFA+): {total_variance}")
            return total_variance
        else:
            raise ValueError("Explained variance not available for MOFA+.")


class MultiVI_Model(ModelFactory):
    """MultiVI Model implementation."""
    
    def __init__(self, dataset: ad.AnnData, dataset_name, config_path: str="./config.json", is_gridsearch=False):
        """
        Initialize the MultiVi model with the specified parameters.
        Input data is AnnData object that was concatenated of multiple modality
        """
        print("Initializing MultiVI Model")

        super().__init__(dataset, dataset_name, config_path=config_path, 
                         model_name="multivi", is_gridsearch=is_gridsearch)
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

        # Output for Multivi model is in ./outputs
        self.latent_filepath = os.path.join(self.output_dir, f"{self.base_filename}.h5ad")
        self.umap_filename = os.path.join(self.output_dir, f"_{self.base_filename}.png")
        
        # Set up data for MultiVI model
        if "feature_types" in self.dataset.var.keys():
            try: 
                self.dataset = self.dataset[:, self.dataset.var["feature_types"].argsort()].copy()

                # Extract protein information if any
                if "Protein Expression" in self.dataset.var["feature_types"].unique():
                    protein_indices = self.dataset.var["feature_types"] == "protein expression"
                    protein_expression = self.dataset.X[:, protein_indices]
                    protein_names = self.dataset.var_names[protein_indices]
                    protein_expression_df = pd.DataFrame(protein_expression, 
                                     index=self.dataset.obs_names, 
                                     columns=protein_names)
                    self.dataset.obsm["protein_expression"] = protein_expression_df
                    scvi.model.MULTIVI.setup_anndata(self.dataset, protein_expression_obsm_key="protein_expression")
                else:
                    scvi.model.MULTIVI.setup_anndata(self.dataset, protein_expression_obsm_key=None)
                
                self.model = scvi.model.MULTIVI(self.dataset,
                                                n_genes=(self.dataset.var["feature_types"] == "Gene Expression").sum(),
                                                n_regions=(self.dataset.var["feature_types"] == "Peaks").sum(),
                                                )
            except Exception as e:
                print(f"Something is wrong in MultiVI initialization: {e}")
                raise
        else:
            raise ValueError("MultiVI initialization needs 'feature_types' in variable keys to setup genes (RNA-seq) and genomic regions (ATAC-seq)!")
        
    def update_output_dir(self):
        super().update_output_dir()
    
    def update_parameters(self, **kwargs):
        super().update_parameters(**kwargs)

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
            filename = os.path.basename(self.umap_filename) 
            sc.pp.neighbors(self.dataset, use_rep=self.latent_key, random_state=1)
            sc.tl.umap(self.dataset, random_state=1)
            sc.pl.umap(self.dataset, color=self.umap_color_type, 
                       save=filename)
            print(f"A UMAP plot for MultiVI model with dataset {self.dataset_name} "\
                  f"was succesfully generated and saved as umap{filename}")

        except Exception as e:
            print(f"Error generating UMAP: {e}")

    def evaluate_model(self):
        """
        Evaluate the trained MultiVI model using clustering metrics.
        """
        if "X_multivi" in self.dataset.obsm:
            latent = self.dataset.obsm["X_multivi"]
            labels = self.dataset.obs.get(self.umap_color_type, None)

            if labels is not None:
                silhouette = silhouette_score(latent, labels)
                print(f"Silhouette Score (MultiVI): {silhouette}")
                return silhouette
            else:
                raise ValueError("Labels not found for clustering evaluation.")
        else:
            raise ValueError("Latent representation (X_multivi) not found.")


class Mowgli_Model(ModelFactory):
    """Mowgli model implementation."""
    
    def __init__(self, dataset, dataset_name, config_path: str="./config.json", is_gridsearch=False):
        """Initialize the Mowgli model with the specified parameters."""

        print("Initializing Mowgli Model")
        
        super().__init__(dataset, dataset_name, config_path=config_path, 
                         model_name="mowgli", is_gridsearch=is_gridsearch)
        mowgli_params = self.model_params.get(self.model_name)

        # Parameters for model settings and training
        self.device = mowgli_params.get("device")
        self.torch_device = 'cpu' # Default for Mowgli
        self.latent_dimensions = mowgli_params.get("latent_dimensions")
        self.optimizer = mowgli_params.get("optimizer")
        self.learning_rate = mowgli_params.get("learning_rate")
        self.inner_tolerance = mowgli_params.get("tol_inner")
        self.max_inner_iteration = mowgli_params.get("max_iter_inner")
        self.umap_color_type = mowgli_params.get("umap_color_type")

        # Create the model instance during initialization
        self.model = mowgli.models.MowgliModel(latent_dim=self.latent_dimensions)
        print(f"Mowgli model initiated with {self.latent_dimensions} dimension.")
        
        # Output for Mowgli model is in ./outputs
        self.latent_filepath = os.path.join(self.output_dir, f"{self.base_filename}.h5ad")
        self.umap_filename = os.path.join(self.output_dir, f"_{self.base_filename}.png")
    
    def update_output_dir(self):
        super().update_output_dir()
    
    def update_parameters(self, **kwargs):
        super().update_parameters(**kwargs)
        
    def to(self):
        """
        Method to set GPU or CPU mode for MOFA+.
        """
        try:
            if self.device !='cpu':
                if torch.cuda.is_available():
                    print("GPU available")
                    print(f"Moving Mowgli model to {self.device}")
                    self.torch_device = torch.device(self.device)
                    print(f"Mowgli model successfully moved to {self.device}")
                else:
                    print("GPU cuda not available. Mowgli model will run with cpu")
            else:
                print("Mowgli model will run with cpu. Recommend to use GPU for computational efficiency.")
        except Exception as e:
            print(f"Invalid device '{self.device}' specified. Use 'cpu' or 'gpu'.")
            raise

    def train(self):
        """Train the Mowgli model."""
        print("Training Mowgli Model")
        try:
            self.model.train(
                self.dataset,
                device=self.torch_device,
                optim_name=self.optimizer,
                lr=self.learning_rate,
                tol_inner=self.inner_tolerance,
                max_iter_inner=self.max_inner_iteration
            )
            self.dataset.obsm["X_mowgli"] = self.dataset.obsm["W_OT"]
            self.loss = self.model.losses[-1]  # Capture the final loss
            print(f"Final training loss: {self.loss}")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        """Save latent data generated with the Mowgli model in .h5ad format."""
        try:
            print("Saving latent data")

            # LOSS of from the model (I don't know if we need it, but just leave it for now)
            H = {f"H_{mod}": self.dataset[mod].uns["H_OT"] for mod in self.dataset.mod}
            losses = {
                "loss_overall": self.model.losses,
                "loss_w": self.model.losses_w,
                "loss_h": self.model.losses_h
            }

            # Prepare the latent anndata for saving
            metadata = {
                    'obs': self.dataset.obs.copy(),  # Observation metadata
                    'var': self.dataset.var.copy(),  # Variable metadata
                    'uns': self.dataset.uns.copy(),  # Unstructured metadata
                    'obsm': self.dataset.obsm.copy(), # Observation matrices
                    'varm': self.dataset.varm.copy()  # Variable matrices
                }
            adata = ad.AnnData(None, **metadata)
            if "X_mowgli" not in adata.obsm:
                adata.obsm["X_mowgli"] = adata.obsm["W_OT"]
            pattern = r".*:cell_type"  # Only keep "cell_type" in .obs and not "rna:cell_type" or "atac:cell_type"
            columns_to_drop = [col for col in adata.obs.columns if re.match(pattern, col)]
            adata.obs.drop(columns=columns_to_drop, inplace=True)
            adata.obs["batch"] = "batch_1"
            adata.write(self.latent_filepath)
            print(f"Latent data saved to {self.latent_filepath}")
        except Exception as e:
            print(f"Error saving latent data: {e}")

    def load_latent(self):
        """Load latent data from saved .h5ad file."""
        print(f"Loading latent data from {self.latent_filepath}")
        
        if os.path.exists(self.latent_filepath):
            self.dataset = sc.read_h5ad(self.latent_filepath)
            print("Latent data loaded successfully.")
        else:
            print(f"File not found: {self.latent_filepath}")
        return self.dataset

    def umap(self):
        """Generate UMAP visualization."""
        print("Generating UMAP plot")
        try:
            sc.settings.figdir = self.output_dir
            filename = os.path.basename(self.umap_filename) 
            sc.pp.neighbors(self.dataset, use_rep="X_mowgli", random_state=1)
            sc.tl.umap(self.dataset, random_state=1)
        
            # Plotting UMAP and saving the figure
            sc.pl.umap(self.dataset, color=self.umap_color_type, save=filename)
        
            print(f"A UMAP plot for Mowgli model with dataset {self.dataset_name} was successfully" \
                  f"generated and saved as umap{filename}")

        except Exception as e:
            print(f"Error generating UMAP: {e}")

    def evaluate_model(self):
        """
        Evaluate the trained Mowgli model using the final training loss.
        """
        if hasattr(self, "loss"):
            print(f"Optimal Transport Loss (Mowgli): {self.loss}")
            return -self.loss  # Lower loss is better, so return negative for maximization
        else:
            raise ValueError("Loss not available for Mowgli.")


