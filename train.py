from muon import atac as ac
import muon as mu
import numpy as np
import scanpy as sc
import dataloader
import pandas as pd
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt

"""
Multi-omics integration
"""


def train(mdata):
    mdata.update()
    mu.pp.intersect_obs(mdata)
    mu.tl.mofa(mdata, n_factors=20, outfile="brain3k_mofa_model.hdf5", gpu_mode=True)
    sc.pp.neighbors(mdata, use_rep="X_mofa")
    sc.tl.umap(mdata, random_state=1)
    mdata.obsm["X_mofa_umap"] = mdata.obsm["X_umap"]
    mu.pl.embedding(mdata, basis="X_mofa_umap", color=["rna:celltype", "atac:celltype"])

    # WNN
    # Since subsetting was performed after calculating nearest neighbours,
    # we have to calculate them again for each modality.
    sc.pp.neighbors(mdata['rna'])
    sc.pp.neighbors(mdata['atac'])

    # Calculate weighted nearest neighbors
    mu.pp.neighbors(mdata, key_added='wnn')
    mu.tl.umap(mdata, neighbors_key='wnn', random_state=10)
    mdata.obsm["X_wnn_umap"] = mdata.obsm["X_umap"]
    mu.pl.embedding(mdata, basis="X_wnn_umap", color=["rna:celltype", "atac:celltype"])

    return mdata