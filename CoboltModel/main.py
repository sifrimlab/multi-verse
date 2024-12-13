import os
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import pandas as pd
import anndata
from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt

def load_anndata(file_path):
    """
    Load an .h5ad file using anndata and return the AnnData object.
    """
    print(f"Loading file: {file_path}")
    return anndata.read_h5ad(file_path)

def extract_data(adata, data_type):
    """
    Extract count matrix, features, and barcodes from AnnData.
    Optionally clean the barcodes by removing the data type suffix.
    """
    count_matrix = adata.X
    features = adata.var_names.to_numpy()
    barcodes = adata.obs_names.to_numpy()
    # Remove suffix from barcodes for compatibility
    barcodes = [barcode.replace(f"_{data_type}", "") for barcode in barcodes]
    return count_matrix, features, barcodes

def create_single_data(data_type, dataset_name, features, counts, barcodes):
    """
    Create a SingleData object for RNA or ATAC data.
    """
    single_data = SingleData(data_type, dataset_name, features, counts, barcodes)
    print(f"{data_type} SingleData created: {single_data.__dict__}")
    return single_data

def train_cobolt_model(multi_dataset, lr=0.00005, n_latent=5, num_epochs=10):
    """
    Initialize and train the Cobolt model on the combined dataset.
    """
    model = Cobolt(dataset=multi_dataset, lr=lr, n_latent=n_latent)
    print("Training the Cobolt model...")
    model.train(num_epochs=num_epochs)
    print("Model training completed.")
    return model

def save_umap_plot_and_embeddings(model, plot_path, csv_path):
    """
    Generate and save UMAP plot and embeddings.
    """
    print("Generating UMAP plot...")
    model.scatter_plot(reduc="UMAP", algo="leiden", resolution=0.5, s=0.2)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"UMAP plot saved at: {plot_path}")

    umap_embedding = model.reduction["UMAP2"]["embedding"]
    np.savetxt(csv_path, umap_embedding, delimiter=",")
    print(f"UMAP embeddings saved at: {csv_path}")

def main():
    """
    Main function to load data, preprocess, train model, and visualize results.
    """
    # Define the root directory for your data
    data_dir = "/CoboltModel/data"
    print("Checking contents of the data directory...")
    print("Contents of /data:", os.listdir(data_dir))

    # Load RNA data
    rna_file = os.path.join(data_dir, "Chen-2019-RNA.h5ad")
    rna_adata = load_anndata(rna_file)
    rna_count, rna_features, rna_barcodes = extract_data(rna_adata, "RNA")
    rna_single_data = create_single_data("RNA", "DATASET", rna_features, rna_count, rna_barcodes)

    # Load ATAC data
    atac_file = os.path.join(data_dir, "Chen-2019-ATAC.h5ad")
    atac_adata = load_anndata(atac_file)
    atac_count, atac_features, atac_barcodes = extract_data(atac_adata, "ATAC")
    atac_single_data = create_single_data("ATAC", "DATASET", atac_features, atac_count, atac_barcodes)

    # Combine RNA and ATAC datasets into a MultiomicDataset
    multi_dt = MultiomicDataset.from_singledata(rna_single_data, atac_single_data)
    print("Data merged successfully into MultiomicDataset.")

    # Train the Cobolt model
    cobolt_model = train_cobolt_model(multi_dt)

    # Calculate Latent Variables and Perform Clustering
    cobolt_model.calc_all_latent()
    latent = cobolt_model.get_all_latent()
    print("Latent representations calculated successfully.")
    
    # Perform clustering
    cobolt_model.clustering(algo="leiden", resolution=0.5)
    clusters = cobolt_model.get_clusters(algo="leiden", resolution=0.5)
    print(f"Clustering completed. Total clusters found: {len(set(clusters))}")

    # Visualize and save UMAP plot and embeddings
    save_path_plot = "/CoboltModel/data/umap_plot_h5ad_chen.png"
    save_path_csv = "/CoboltModel/data/umap_embedding_h5ad_chen.csv"
    save_umap_plot_and_embeddings(cobolt_model, save_path_plot, save_path_csv)

    print("Processing and visualization completed successfully.")

if __name__ == "__main__":
    main()