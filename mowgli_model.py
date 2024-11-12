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
    """Mowgli model implementation with train, evaluate, and predict methods."""
    
    def __init__(self, data_dir, dataset, latent_dimensions, device, learning_rate):
        print("Initializing Mowgli Model")
        self.data_dir = data_dir
        self.dataset = dataset
        self.latent_dimensions = latent_dimensions
        self.device = device
        self.learning_rate = learning_rate
        # Add any initialization logic (e.g., loading data, model setup, etc.)

    def model(self)
        print("Defining the model")
        model = mowgli.models.MowgliModel(latent_dim=self.latent_dimensions)
        return model

    def train(self, model):
        print("Training Mowgli Model")
        model.train(self.dataset, device=self.device, optim_name='sgd', lr=self.learning_rate, tol_inner=1e-5)

    def evaluate(self):
        print("Evaluating Mowgli Model")
        # Add evaluation logic for Mowgli here
        raise NotImplementedError

    def save_latent(self):
        print("Saving data generated with Mowgli Model")
        np.save(self.data_dir + f("mowgli_output/mowgli_",self.dataset,".npy"),
            {
                "W": self.dataset.obsm["W_OT"],
                **{"H_" + mod: self.dataset[mod].uns["H_OT"] for mod in self.dataset.mod},
                "loss_overall": model.losses,
                "loss_w": model.losses_w,
                "loss_h": model.losses_h
            },
            )

    def load_latent(self):
        print("Loading data")
        mowgli_data = np.load(data_dir+f("mowgli_output/mowgli_",self.dataset,".npy", allow_pickle=True).item()["W"]
        self.dataset.obsm["mowgli_data"] = mowgli_data
        self.dataset.uns = {}
        return mowgli_data

    def umap(self, num_neighbors, umap_size, umap_alpha, filename):
        print("Mapping the Mowgli umap")
        sc.pp.neighbors(self.dataset, use_rep="X_mowgli", n_neighbors=num_neighbors)
        sc.tl.umap(self.dataset)
        sc.pl.umap(self.dataset, size=umap_size, alpha=umap_alpha, save=data_dir+filename)