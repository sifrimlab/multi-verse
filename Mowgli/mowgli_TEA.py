import sys
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

print(sys.executable)
print(os.environ['USER'])

# Load the data.
mdata_TEA = mu.read_h5mu("/scratch/leuven/351/vsc35109/Mowgli_data/PBMC/100_tea_preprocessed.h5mu.gz")

# Initialize and train the model.
# Define the model.
model = mowgli.models.MowgliModel(latent_dim=15)

#Train the model
model.train(mdata_TEA, max_iter_inner=500, max_iter= 50) # I reduce inner iteration from default 1000 to 500, outer iteration from 100 to 50

#Save model.

np.save("mowgli_TEA_100.npy",
        {
            "W": mdata_TEA.obsm["W_OT"],
            **{"H_" + mod: mdata_TEA[mod].uns["H_OT"] for mod in mdata_TEA.mod},
        },
        )

#Save umap
sc.pp.neighbors(mdata_TEA, use_rep="W_OT", key_added="mowgli")
sc.tl.umap(mdata_TEA, neighbors_key="mowgli")
sc.pl.umap(mdata_TEA, "mowgli", size=50, alpha=0.7, save="15.png")

