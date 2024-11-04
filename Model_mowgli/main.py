import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import mowgli
import mudata as md
import scanpy as sc

# Load data into a Muon object.
import muon as mu
from muon import MuData
import numpy as np
import matplotlib.pyplot as plt
import os
from mowgli import models
import torch
import leidenalg


def main():
    data_dir = "/mowgli/data_TEA/"  # Path to the mounted data directory
    # Load the data.
    mdata_TEA = mu.read_h5mu(data_dir + "GSM4949911_tea_preprocessed.h5mu.gz")

    # Initialize and train the model.
    # Define the model.
    model = mowgli.models.MowgliModel(latent_dim=15)

    # Train the model
    boo = torch.cuda.is_available()
    if boo:
        print("GPU available")
        device=torch.device("cuda:0")
        model.train(mdata_TEA, device=device, optim_name='sgd', lr=0.01, tol_inner=1e-7)
        #Save model and losses
        np.save(data_dir +"embeddings/mowgli_GSM4949911.npy",
            {
                "W": mdata_TEA.obsm["W_OT"],
                **{"H_" + mod: mdata_TEA[mod].uns["H_OT"] for mod in mdata_TEA.mod},
                "loss_overall": model.losses,
                "loss_w": model.losses_w,
                "loss_h": model.losses_h
            },
            )

        # Plotting embeddings
        #Load embeddings
        X_mowgli = np.load(data_dir+"embeddings/mowgli_GSM4949911.npy", allow_pickle=True).item()["W"]
        mdata_TEA.obsm["X_mowgli"] = X_mowgli
        mdata_TEA.uns = {}

        #Save umap
        sc.pp.neighbors(mdata_TEA, use_rep="X_mowgli", n_neighbors=20)
        sc.tl.umap(mdata_TEA)
        sc.pl.umap(mdata_TEA, size=50, alpha=0.7, save=data_dir+"_TEA.png")

        #leiden
        sc.tl.leiden(mdata_TEA)
        sc.pl.embedding(mdata_TEA, "X_umap", color=['leiden'], save=data_dir+"_leiden.png")

    else:
        print("cuda not available? Quitting python.")
        quit()

if __name__ == "__main__":
    main()