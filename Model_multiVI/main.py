import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import scvi
import muon as mu
import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import scanpy as sc
import seaborn as sns
import torch
import pandas as pd

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

def main():
    data_dir = "/multivi/data_TEA/"
    adata_rna = ad.read_h5ad(data_dir+"Chen-2019-RNA.h5ad")
    adata_atac = ad.read_h5ad(data_dir+"Chen-2019-ATAC.h5ad")

    adata_rna.obs.index = adata_rna.obs.index.str.replace('_RNA', '', regex=False)
    adata_atac.obs.index = adata_atac.obs.index.str.replace('_ATAC', '', regex=False)
    print(adata_atac.obs)

    adata_mvi = ad.concat([adata_rna, adata_atac], join="outer", axis=1, label="modality")
    adata_mvi.obs['cell_type'] = adata_rna.obs['cell_type'] #Cell types based on rna file
    print(adata_mvi)

    scvi.model.MULTIVI.setup_anndata(adata_mvi)
    model = scvi.model.MULTIVI(
            adata_mvi,
            n_genes=(adata_mvi.var["modality"] == "0").sum(),
            n_regions=(adata_mvi.var["modality"] == "1").sum(),
            )
    model.view_anndata_setup()
    
    model.train()
    model.save("trained_chen")

    MULTIVI_LATENT_KEY = "X_multivi"
    adata_mvi.obsm[MULTIVI_LATENT_KEY] = model.get_latent_representation()
    adata_mvi.write(data_dir+"Chen_multiVI.h5ad")

    sc.pp.neighbors(adata_mvi, use_rep=MULTIVI_LATENT_KEY)
    sc.tl.umap(adata_mvi, min_dist=0.2)
    sc.pl.umap(adata_mvi, color="cell_type", save="_CHEN.png")


if __name__ == "__main__":
    main()