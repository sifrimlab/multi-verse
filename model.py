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
        if  self.device != 'cpu':
            self.gpu_mode = True
        else:
            self.gpu_mode = False
        self.umap_random_state = pca_params.get("umap_random_state")
        self.umap_color_type = pca_params.get("umap_color_type")

        # Output for PCA model is in ./outputs/pca_output
        self.output_dir = os.path.join(self.outdir, "pca_output")
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

        print(f"PCA completed with {self.n_components} components")

    def save_latent(self):
        """Save the PCA latent representations."""
        print("Saving PCA latent embeddings")
        output_path = os.path.join(self.output_dir, f"pca_{self.dataset_name}.h5ad")
        self.dataset.write(output_path)
        print(f"Latent data saved to {output_path}")
        # Store the path for loading later (e.g., assign it to self.latent_filepath)
        self.latent_filepath = output_path

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
        if self.device !='cpu':
            self.gpu_mode = False
        else:
            self.gpu_mode = True
        self.n_factors = mofa_params.get("n_factors")
        self.n_iteration = mofa_params.get("n_iteration")
        self.umap_random_state=mofa_params.get("umap_random_state")
        self.umap_use_representation=mofa_params.get("umap_use_representation")
        self.umap_color_type=mofa_params.get("umap_color_type")

        # Output for MOFA+ model is in ./outputs/mofa_output
        self.output_dir = os.path.join(self.outdir, "mofa_output")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"MOFA+ initialized with {self.dataset_name}, {self.n_factors} factors to be trained with.")

    def to(self):
        """
        Method to set GPU or CPU mode for MOFA+.
        """
        self.gpu_mode = True
        print(f"Switching to {self.device} mode")

    def train(self):
        """
        Train the MOFA model.
        """
        print("Training MOFA+ Model")
        outfilepath = os.path.join(self.output_dir, f"mofa_{self.dataset_name}.hdf5")
        try:
            mu.tl.mofa(data=self.dataset, n_factors=self.n_factors, 
                       outfile=outfilepath, gpu_mode=self.gpu_mode)
            print(f"Model saved at: {outfilepath}")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        """
        Save the latent space embeddings from the trained MOFA model.
        """
        try:
            # Save the latent embeddings to the dataset object
            print("Saving latent embeddings as .npy file to dataset object...")
            
            # Ensure the necessary embeddings and factors are in the dataset
            X_mofa = self.dataset.obsm["X_mofa"]
            LFs_mofa = self.dataset.varm["LFs_mofa"]
            
            # Create a dictionary to store the latent embeddings and factors
            latent_data = {
                'X_mofa': X_mofa,
                'LFs_mofa': LFs_mofa
            }
            
            # Define the output path for the saved .npy file
            output_path = os.path.join(self.output_dir, f"mofa_latent_{self.dataset_name}.npy")
            
            self.latent_filepath = output_path
            
            # Save the latent data to a .npy file (using np.save)
            np.save(output_path, latent_data)
            
            print(f"Latent data saved to {output_path}")

        except Exception as e:
            print(f"Error saving latent embeddings to dataset: {e}")

    def load_latent(self) :
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
        sc.pp.neighbors(self.dataset, use_rep=self.umap_use_representation, random_state=self.umap_random_state)
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
        self.protein_expression = multivi_params.get("protein_expression_obsm_key")
        self.max_epochs = multivi_params.get("max_epochs")
        self.learning_rate = multivi_params.get("learning_rate")
        self.latent_key = "X_multivi"
        self.umap_color_type = multivi_params.get("umap_color_type")

        # Output for MOFA+ model is in ./outputs/multivi_output
        self.output_dir = os.path.join(self.outdir, "multivi_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up data for MultiVI model
        self.dataset = self.dataset[:, self.dataset.var["feature_types"].argsort()].copy()
        scvi.model.MULTIVI.setup_anndata(self.dataset, protein_expression_obsm_key=self.protein_expression)
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
            print(f"Moving MultiVI model to {self.device}")
            self.model.to_device(self.device)
            print(f"Model successfully moved to {self.device}")
        except Exception as e:
            print(f"Invalid device '{self.device}' specified. Use 'cpu' or 'gpu'.")
        
    def train(self):
        print("Training MultiVI Model")
        try:
            self.to()
            self.model.train()
            self.dataset.obsm[self.latent_key] = self.model.get_latent_representation()
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        """Save latent data generated with the MultiVI model."""
        print("Saving latent data")
        output = os.path.join(self.output_dir, f"multivi_latent_{self.dataset_name}.h5ad"),
        self.latent_filepath = output
        try:
            self.model.save(self.output_dir)
            self.dataset.obsm[self.latent_key] = self.model.get_latent_representation()
            self.dataset.write(self.latent_filepath)

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
            print(f"File not found: {self.output_dir}, multivi_{self.dataset_name}.txt")

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

        mowgli_params= load_config()["models"]["mowgli"].get("model_params")

        self.dataset = dataset
        self.dataset_name = dataset_name
        self.device = mowgli_params.get("device")
        self.latent_dimensions = mowgli_params.get("latent_dimensions")
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
        output_path = os.path.join(self.output_dir, f"mowgli_latent_{self.dataset_name}.npy"),
        self.latent_filepath = output_path
        try:
            np.save(
                self.latent_filepath,
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
            mowgli_data = np.load(self.latent_filepath, allow_pickle=True).item()
            self.dataset.obsm["W_mowgli"] = mowgli_data["W"]
            self.dataset.uns = {}
            return mowgli_data
        except FileNotFoundError:
            print(f"File not found: {self.latent_filepath}")
        except Exception as e:
            print(f"Error loading latent data: {e}")

    def umap(self):
        """Generate UMAP visualization."""
        print("Generating UMAP plot")
        try:
            sc.pp.neighbors(self.dataset, use_rep="X_mowgli", n_neighbors=self.umap_num_neighbors)
            sc.tl.umap(self.dataset)
            sc.pl.umap(self.dataset, size=self.umap_size, alpha=self.umap_alpha)
        
            # If filename is not provided, use default that includes self.dataset_name
            umap_filename = os.path.join(self.output_dir, f"mowgli_{self.dataset_name}_umap_plot.png")
        
            # Save the plot
            plt.savefig(os.path.join(self.output_dir, umap_filename))
            plt.close()
        
            print(f"A UMAP plot for Mowgli model with dataset {self.dataset_name} was successfully" \
                  f"generated and saved as {umap_filename}")

        except Exception as e:
            print(f"Error generating UMAP: {e}")


