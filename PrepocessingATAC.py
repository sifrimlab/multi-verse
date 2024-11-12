from muon import atac as ac
import muon as mu
import numpy as np
import scanpy as sc
import dataloader
import pandas as pd
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt


def Prepocessing_ATAC(mdata):
    atac = mdata.mod['atac']

    # QC
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    mu.pl.histogram(atac, ['n_genes_by_counts', 'total_counts'], linewidth=0)
    mu.pp.filter_var(atac, 'n_cells_by_counts', lambda x: x >= 10)
    print(f"Before: {atac.n_obs} cells")
    mu.pp.filter_obs(atac, 'total_counts', lambda x: (x >= 1000) & (x <= 80000))
    print(f"(After total_counts: {atac.n_obs} cells)")
    mu.pp.filter_obs(atac, 'n_genes_by_counts', lambda x: (x >= 100) & (x <= 30000))
    print(f"After: {atac.n_obs} cells")
    mu.pl.histogram(atac, ['n_genes_by_counts', 'total_counts'], linewidth=0)

    # Nucleosome signal
    ac.pl.fragment_histogram(atac, region='chr1:1-2000000')
    ac.tl.nucleosome_signal(atac, n=1e6)
    mu.pl.histogram(atac, "nucleosome_signal", linewidth=0)

    # TSS enrichment
    ac.tl.get_gene_annotation_from_rna(mdata['rna']).head(
        3)  # accepts MuData with 'rna' modality or mdata['rna'] AnnData directly
    tss = ac.tl.tss_enrichment(mdata, n_tss=1000)  # by default, features=ac.tl.get_gene_annotation_from_rna(mdata)
    ac.pl.tss_enrichment(tss)

    # Normalisation
    atac.layers["counts"] = atac.X.copy()
    sc.pp.normalize_total(atac, target_sum=1e4)
    sc.pp.log1p(atac)
    atac.layers["lognorm"] = atac.X.copy()

    # Define informative features
    sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
    sc.pl.highly_variable_genes(atac)
    np.sum(atac.var.highly_variable)

    # Scaling and PCA
    sc.pp.scale(atac, max_value=10)
    sc.tl.pca(atac, svd_solver='arpack')
    ac.pl.pca(atac, color=['NRCAM', 'SLC1A2', 'SRGN', 'VCAN'], layer='lognorm', func='mean')
    sc.pl.pca_variance_ratio(atac, log=True)

    # Finding cell neighbours and clustering cells
    sc.pp.neighbors(atac, n_neighbors=10, n_pcs=20)
    sc.tl.leiden(atac, resolution=.5)

    # Non-linear dimensionality reduction
    sc.tl.umap(atac, spread=1., min_dist=.5, random_state=11)
    sc.pl.umap(atac, color="leiden", legend_loc="on data")

    # Marker genes and celltypes
    ac.tl.rank_peaks_groups(atac, 'leiden', method='t-test')
    result = atac.uns['rank_genes_groups']
    groups = result['names'].dtype.names

    try:
        pd.set_option("max_columns", 50)
    except:
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html
        pd.set_option("display.max_columns", 50)

    pd.DataFrame(
        {group + '_' + key[:1]: result[key][group]
         for group in groups for key in ['names', 'genes', 'pvals']}).head(10)

    mu.pp.filter_obs(atac, "leiden", lambda x: ~x.isin(["9"]))
    new_cluster_names = {
        "0": "oligodendrocyte",
        "1": "oligodendrocyte",
        "3": "OPC",
        "7": "microglia",
        "2": "astrocyte",
        "8": "astrocyte",
        "4": "excitatory",
        "5": "inhibitory1",
        "6": "inhibitory2",
    }
    atac.obs['celltype'] = [new_cluster_names[cl] for cl in atac.obs.leiden.astype("str").values]
    atac.obs.celltype = atac.obs.celltype.astype("category")

    atac.obs.celltype = atac.obs.celltype.cat.set_categories([
        'oligodendrocyte', 'OPC', 'microglia', 'astrocyte',
        'excitatory', 'inhibitory1', 'inhibitory2'
    ])
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(atac.obs.celltype.cat.categories)))

    atac.uns["celltype_colors"] = list(map(to_hex, colors))
    sc.pl.umap(atac, color="celltype")
    marker_genes = ["PLP1", "CNP", "CTNNA3",
                    "VCAN", "SRGN", "SLC1A2",
                    "SLC17A7", "LAMP5", "CBLN2", "RORB", "FOXP2",
                    "GAD1", "GAD2", "LHX6", "ADARB2", ]
    ac.pl.dotplot(atac, marker_genes, groupby='celltype')
