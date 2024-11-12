import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import mowgli
import mudata as md
import scanpy as sc
import muon as mu
from muon import MuData
import numpy as np
import matplotlib.pyplot as plt
import os
from mowgli import models
import torch
import leidenalg

class Mowgli_Model:
    """Mowgli model implementation."""
    
    def __init__(self, data_dir, dataset, latent_dimensions, device, learning_rate):
        print("Initializing Mowgli Model")
        self.data_dir = data_dir
        self.dataset = dataset
        self.latent_dimensions = latent_dimensions
        self.device = device
        self.learning_rate = learning_rate
        
        # Create the model instance during initialization
        self.model = mowgli.models.MowgliModel(latent_dim=self.latent_dimensions)
        
        # Ensure output directory exists
        self.output_dir = os.path.join(self.data_dir, "mowgli_output")
        os.makedirs(self.output_dir, exist_ok=True)


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
                os.path.join(self.output_dir, f"mowgli_{self.dataset}.npy"),
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
            file_path = os.path.join(self.output_dir, f"mowgli_{self.dataset}.npy")
            mowgli_data = np.load(file_path, allow_pickle=True).item()
            self.dataset.obsm["W_mowgli"] = mowgli_data["W"]
            self.dataset.uns = {}
            return mowgli_data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading latent data: {e}")

    def umap(self, num_neighbors=15, umap_size=20, umap_alpha=0.8, filename='umap_plot.png'):
        """Generate UMAP visualization."""
        print("Generating UMAP plot")
        try:
            sc.pp.neighbors(self.dataset, use_rep="X_mowgli", n_neighbors=num_neighbors)
            sc.tl.umap(self.dataset)
            sc.pl.umap(self.dataset, size=umap_size, alpha=umap_alpha)
            
            # Save the plot
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
        except Exception as e:
            print(f"Error generating UMAP: {e}")