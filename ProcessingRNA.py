from muon import atac as ac
import muon as mu
import numpy as np
import scanpy as sc
import dataloader
import pandas as pd
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt


def Processing_RNA(mdata):
    # `rna` will point to `mdata['rna']`
    # unless we copy it
    rna = mdata['rna']

    # QC
    rna.var['mt'] = rna.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)

    print(f"Before: {rna.n_obs} cells")
    mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= 200) & (x < 8000))
    print(f"(After n_genes: {rna.n_obs} cells)")
    mu.pp.filter_obs(rna, 'total_counts', lambda x: x < 40000)
    print(f"(After total_counts: {rna.n_obs} cells)")
    mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x < 2)
    print(f"After: {rna.n_obs} cells")

    sc.pl.violin(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)

    # Normalisation
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    # rna.raw = rna
    rna.layers["lognorm"] = rna.X.copy()

    # Define informative features
    sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
    sc.pl.highly_variable_genes(rna)
    np.sum(rna.var.highly_variable)

    # Scaling and PCA
    sc.pp.scale(rna, max_value=10)
    sc.tl.pca(rna, svd_solver='arpack')
    sc.pl.pca(rna, color=['NRCAM', 'SLC1A2', 'SRGN', 'VCAN'])
    sc.pl.pca_variance_ratio(rna, log=True)

    # Finding cell neighbours and clustering cells
    sc.pp.neighbors(rna, n_neighbors=10, n_pcs=20)
    sc.tl.leiden(rna, resolution=.5)

    # Non-linear dimensionality reduction
    sc.tl.umap(rna, spread=1., min_dist=.5, random_state=11)
    sc.pl.umap(rna, color="leiden", legend_loc="on data")

    # Marker genes and celltypes
    sc.tl.rank_genes_groups(rna, 'leiden', method='t-test')
    result = rna.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    pd.set_option('display.max_columns', 50)
    pd.DataFrame({group + '_' + key[:1]: result[key][group]
                  for group in groups for key in ['names', 'pvals']}).head(10)
    sc.pl.rank_genes_groups(rna, n_genes=20, sharey=False)

    # Cell type annotation
    sc.pl.umap(rna, color=["PLP1", "CNP", "CTNNA3"])
    sc.pl.umap(rna, color=["SLC1A2", "SRGN", "VCAN"], title=["SLC1A2 (astrocytes)", "SRGN (microglia)", "VCAN (OPCs)"])
    sc.pl.umap(rna, color=["GAD1", "GAD2", "SLC17A7"],
               title=["GAD1 (inhibitory)", "GAD2 (inhibitory)", "SLC17A7 (excitatory)"])
    sc.pl.umap(rna, color=["LHX6", "ADARB2"])
    sc.pl.umap(rna, color=["RORB", "FOXP2", "LAMP5", "CBLN2"])
    new_cluster_names = {
        "0": "oligodendrocyte",
        "1": "oligodendrocyte",
        "3": "oligodendrocyte",
        "5": "oligodendrocyte",
        "14": "oligodendrocyte",
        "4": "OPC",
        "8": "microglia",
        "2": "astrocyte",
        "10": "astrocyte",
        "11": "astrocyte",
        "12": "astrocyte",
        "6": "excitatory_LAMP5",
        "13": "excitatory_RORB",
        "7": "inhibitory_LHX6",
        "9": "inhibitory_ADARB2",
        "15": "inhibitory_ADARB2",
    }
    rna.obs['celltype'] = [new_cluster_names[cl] for cl in rna.obs.leiden.astype("str").values]
    rna.obs.celltype = rna.obs.celltype.astype("category")
    rna.obs.celltype = rna.obs.celltype.cat.set_categories([
        'oligodendrocyte', 'OPC', 'microglia', 'astrocyte',
        'excitatory_LAMP5', 'excitatory_RORB',
        'inhibitory_LHX6', 'inhibitory_ADARB2'
    ])
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(rna.obs.celltype.cat.categories)))

    rna.uns["celltype_colors"] = list(map(to_hex, colors))
    sc.pl.umap(rna, color="celltype")
    marker_genes = ["PLP1", "CNP", "CTNNA3",
                    "VCAN", "SRGN", "SLC1A2",
                    "SLC17A7", "LAMP5", "CBLN2", "RORB", "FOXP2",
                    "GAD1", "GAD2", "LHX6", "ADARB2", ]
    sc.pl.dotplot(rna, marker_genes, groupby='celltype')
