from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt
import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy import io
import matplotlib.pyplot as plt
import pandas as pd
import anndata
print('Packages loaded successfully')

# Define the root directory for your data
data_dir = "/CoboltModel/data"
print("Contents of /data:", os.listdir(data_dir))

# Load RNA .h5ad file
rna_file = os.path.join(data_dir, "Chen-2019-RNA.h5ad")  # Replace with your RNA file name
rna_adata = anndata.read_h5ad(rna_file)
print('RNA file path:', rna_file)

# Extract count matrix (X), features, and barcodes for RNA
rna_count = rna_adata.X
rna_features = rna_adata.var_names.to_numpy()
rna_barcodes = rna_adata.obs_names.to_numpy()

# Remove names that rpevent from barcode recognition
rna_barcodes = [barcode.replace("_RNA", "") for barcode in rna_barcodes]

# Create SingleData object for RNA
rna_single_data = SingleData("RNA", "DATASET", rna_features, rna_count, rna_barcodes)
print('RNA single data:', rna_single_data.__dict__)
print('Type of data', type(rna_single_data))

# Extract annotations
cell_type_annotation = rna_adata.obs["cell_type"].astype(str).values
cell_type_annotation

# Load ATAC .h5ad file
atac_file = os.path.join(data_dir, "Chen-2019-ATAC.h5ad")  # Replace with your ATAC file name
atac_adata = anndata.read_h5ad(atac_file)
print('Path to ATAC file', atac_file)

# Extract count matrix (X), features, and barcodes for ATAC
atac_count = atac_adata.X
atac_features = atac_adata.var_names.to_numpy()
atac_barcodes = atac_adata.obs_names.to_numpy()

# Remove names that rpevent from barcode recognition
atac_barcodes = [barcode.replace("_ATAC", "") for barcode in atac_barcodes]

# Create SingleData object for ATAC
atac_single_data = SingleData("ATAC", "DATASET", atac_features, atac_count, atac_barcodes)
print('ATAC single data:', atac_single_data.__dict__)
print('Type of data', type(atac_single_data))

# # Perform quality filtering on features (optional)
# rna_single_data.filter_features(upper_quantile=0.99, lower_quantile=0.7)
# atac_single_data.filter_features(upper_quantile=0.99, lower_quantile=0.7)

# Combine RNA and ATAC datasets into a MultiomicDataset
multi_dt = MultiomicDataset.from_singledata(rna_single_data, atac_single_data)
print('Data were merged successfully - MultiomicDataset', multi_dt)

# Train the Cobolt Model
cobolt_model = Cobolt(dataset=multi_dt, lr=0.00005, n_latent=5) # adjust values accordingly
# note: learning rate higher than 0.001 gives an error

print('Training the model')

cobolt_model.train(num_epochs=20) # normally use 100
print('Model was trained successfully')

print('Latent representations:', cobolt_model.get_all_latent)

# Calculate Latent Variables and Perform Clustering
cobolt_model.calc_all_latent()
latent = cobolt_model.get_all_latent()
print('Clustering performed successfully')

latent_raw = cobolt_model.get_all_latent(correction=False)

# Cluster and visualise:
cobolt_model.clustering(algo="leiden", resolution=0.5)
clusters = cobolt_model.get_clusters(algo="leiden", resolution=0.5)
print('Preparing to visualise the clusters')

# Visualize clustering with UMAP
cobolt_model.scatter_plot(
    reduc="UMAP", 
    algo="leiden", 
    resolution=0.5, 
    s=0.2)

# Save the plot as a PNG or PDF
plt.savefig("/CoboltModel/data/umap_plot_h5ad_chen.png", dpi=300)  # You can also use "umap_plot.pdf"
plt.close()  # Close the plot to free memory
print('Umap saved successfully')

umap_reduc = cobolt_model.reduction["UMAP2"]["embedding"]
np.savetxt("/CoboltModel/data/umap_embedding_h5ad_chen.csv", umap_reduc, delimiter=",")

save_path_plot = "/CoboltModel/data/umap_plot_h5ad_chen.png"
save_path_csv = "/CoboltModel/data/umap_embedding_h5ad_chen.csv"
print(f"Saving UMAP plot to: {save_path_plot}")
print(f"Saving embeddings to: {save_path_csv}")

print('Code run successfully!')