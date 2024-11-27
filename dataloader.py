import scanpy as sc
import anndata as ad
import mudata as md
import muon as mu
import os
import numpy as np
from config import load_config
from typing import List
import json

#Output type = anndata, mudata

class DataLoader:
    def __init__(self, file_path: str = "", modality: str = "", isProcessed=True, annotation: str=None, config_path: str="./config.json"):
        # These attributes should be loaded from the config file
        self.config_path = config_path

        self.file_path = file_path
        self.modality = modality
        self.is_preprocessed = isProcessed
        self.annotation = annotation
        self.data = None

    def read_anndata(self) -> ad.AnnData:
        """
        Read files as anndata object
        Note that if mu.read_10x_mtx is used, make sure there are barcodes.tsv.gz and features.tsv.gz are available in the same folder
        Args:
            Specific modality and file_path must be provided when defining DataLoader object
            support file format: [".csv", ".tsv", ".h5ad", ".txt", ".mtx", ".h5mu", ".h5"]
        Returns:
            A anndata.AnnData object (unprocessed)
        """
        adata = None
        # Modality and file_path should be provided to load anndata object
        if self.modality != "" and self.file_path != "": 
            if ".csv" in self.file_path:
                adata = sc.read_csv(self.file_path)
            elif ".tsv" in self.file_path:
                adata = sc.read(self.file_path, delimiter='\t').T
            elif ".h5ad" in self.file_path:
                adata = sc.read_h5ad(self.file_path)  
            elif ".txt" in self.file_path:
                adata = sc.read_text(self.file_path)
            elif ".mtx" in self.file_path:
                if self.modality in ["rna", "atac"]:
                    path = os.path.dirname(self.file_path)
                    mudata = mu.read_10x_mtx(path, extended=True) 
                    adata = mudata[self.modality]
                else:
                    adata = sc.read_mtx(self.file_path)
            elif ".h5mu" in self.file_path:
                mudata = mu.read_h5mu(self.file_path)
                adata = mudata[self.modality]
            elif ".h5" in self.file_path:
                mudata =mu.read_10x_h5(self.file_path)
                adata = mudata[self.modality]

            if adata:  # Check if adata is not None
                adata.var_names_make_unique()

                # Annotation processing
                if self.annotation == None:
                    self.annotation = "cell_type"
                    num_obs = adata.n_obs
                    adata.obs[self.annotation] = np.zeros(num_obs, dtype=int)

                self.data = adata
                return self.data
            else:
                raise ValueError("Could not read the file. Please check the file path and format.")
        else:
            raise ValueError("Modality and file_path must be provided for anndata loading.")

    def read_mudata(self) -> md.MuData:
        """
        Read h5mu file as MuData object.
        Args:
            file_path must be provided when defining DataLoader object
            Only support file format ".h5mu", ".h5", ".mtx"
        Returns:
            A mudata.MuData object
        """
        if self.file_path != "":
            if ".h5mu" in self.file_path:
                mudata = mu.read_h5mu(self.file_path)
            elif ".h5" in self.file_path:
                mudata =mu.read_10x_h5(self.file_path)
            elif ".mtx" in self.file_path:
                path = os.path.dirname(self.file_path)
                mudata = mu.read_10x_mtx(path, extended=True) 
            else:
                raise ValueError("Could not read the file. Only support file format .h5mu, .h5, .mtx.")
        else:
            raise ValueError("file_path must be provided to read mudata files")
        
        self.data = mudata
        return self.data

    def fuse_mudata(self, list_anndata: List[ad.AnnData] = None, list_modality: List[str] = None, annotate: str ="cell_type") -> md.MuData:
        """
        Fusing paired anndata as MuData
        intersect_obs will be used if number of obs not equivalent
        Args:
            list_modality: A list of strings representing the modalities (e.g., ["rna", "atac", "adt"]).
            list_anndata: A list of AnnData objects corresponding to the modalities (e.g., [adata_rna, adata_atac, adata_adt]).
        Returns:s
            A mudata.MuData object    
        """
        if len(list_modality) != len(list_anndata):
            raise ValueError("Length of list_modality and list_anndata must be equal!")
        else:
            data_dict = {}
            for i, mod in enumerate(list_modality):
                data_dict[mod] = list_anndata[i]
                try:
                    list_anndata[i].X = np.array(list_anndata[i].X.todense())
                except:
                    pass
                    
        self.data = mu.MuData(data_dict)
        mu.pp.intersect_obs(self.data)   # Make sure number of cells are the same for all modalities

        # setting annotation for mudata
        if annotate in self.data["rna"].obs.columns:
            self.data.obs[annotate] = self.data["rna"].obs[annotate]
        else:
            # If there is no 'cell_type' annotation in rna modality
            print("No annotation -> setting annotation as 'cell_type' = 0")
            num_obs = self.data.n_obs
            self.data.obs[annotate] = np.zeros(num_obs, dtype=int)
        
        return self.data

    def anndata_concatenate(self, list_anndata: List[ad.AnnData] = None, list_modality: List[str] = None, annotate: str ="cell_type") -> ad.AnnData:
        """
        Args:
            list_modality: A list of strings representing the modalities (e.g., ["rna", "atac", "adt"]).
            list_anndata: A list of AnnData objects corresponding to the modalities (e.g., [adata_rna, adata_atac, adata_adt]).
        Returns:s
            A AnnData object
        """
        mudata = self.fuse_mudata(list_anndata=list_anndata, list_modality=list_modality)
        list_ann = []
        for mod in list_modality:
            list_ann.append(mudata[mod])

        anndata = ad.concat(list_ann, axis="var") # concatenate based on "var" axis

        # setting annotation for mudata
        num_obs = anndata.n_obs
        if annotate in mudata.obs.columns:
            anndata.obs[annotate] = mudata.obs[annotate]
        else:
            anndata.obs[annotate] = np.zeros(num_obs, dtype=int)
        
        anndata.obs["modality"] = np.zeros(num_obs, dtype=int) # Adding this to prevent error in multiVI model
        self.data = anndata
        return self.data

    def preprocessing(self) -> ad.AnnData:
        """
        Preprocessing each anndata object
        Only support rna, atac, adt modality currently
        Args:
            Modality and file_path must be provided when defining DataLoader object
        Returns:
            A muon.MuData object
        """
         # Modality and file_path should be provided for the read_anndata() function to work
        if self.file_path != "":
            if self.modality != "" :
                self.read_anndata()
                if not self.is_preprocessed:
                    self.data.var_names_make_unique()
                    self.data.layers["counts"] = self.data.X.copy()
                    pre = Preprocessing(anndata=self.data, config_path=self.config_path)
                    # RNA preprocessing
                    if self.modality=="rna":
                        self.data = pre.rna_preprocessing()
                    # ATAC preprocessing
                    elif self.modality=="atac":
                        self.data = pre.atac_preprocessing()
                    # ADT preprocessing
                    elif self.modality=="adt":
                        self.data = pre.adt_preprocessing()
                    # Not applicable
                    else:
                        raise ValueError("Preprocessing for this modality is not applicable!")
            else:
                # If there is no modality (assume to load entire mudata object)
                self.read_mudata()
        else:
            raise ValueError("File_path must be provided to read anndata")
        return self.data


class Preprocessing:
    def __init__(self, anndata: ad.AnnData, config_path: str="./config.json"):
        self.data = anndata
        self.config = load_config(config_path=config_path).get("preprocess_params")
    
    def rna_preprocessing(self) -> ad.AnnData:
        """
        QC metrics for filtering obs depends on specific dataset and experimental condition.
        Top highly variable gene = 2000
        Returns:
            An anndata.AnnData object (proccessed)
        """

        rna_dict = self.config.get("rna_filtering")

        # Quality control - based on scanpy calculateQCmetrics - McCarthy et al., 2017
        self.data.var["mt"] = self.data.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(self.data, qc_vars=["mt"], inplace=rna_dict.get("qc_metric_inplace"), log1p=rna_dict.get("qc_metric_log1p"))
        
        # Filtering -> threshold metrics depend on the specific dataset and experimental conditions.
        mu.pp.filter_obs(self.data, 'n_genes_by_counts', lambda x: (x >= rna_dict.get("min_genes_by_counts")) & (x < rna_dict.get("max_genes_by_counts")))
        mu.pp.filter_obs(self.data, "total_counts", lambda x: x < rna_dict.get("max_total_counts_per_cell"))
        mu.pp.filter_obs(self.data, "pct_counts_mt", lambda x: x < rna_dict.get("max_pct_counts_mt"))

        # Filter genes by keeping only those that are expressed in at least 10 cells.
        mu.pp.filter_var(self.data, "n_cells_by_counts", lambda x: x >= rna_dict.get("min_cells_by_counts"))        
        
        # Normalisation
        sc.pp.normalize_total(self.data, target_sum=rna_dict.get("normalization_target_sum"))
        sc.pp.log1p(self.data)

        # Feature selection
        sc.pp.highly_variable_genes(self.data, n_top_genes=rna_dict.get("n_top_genes"), subset=True,flavor="seurat")

        return self.data

    def atac_preprocessing(self) -> ad.AnnData:
        """
        QC metrics for filtering obs depends on specific dataset and experimental condition.
        Top highly variable peaks = 15000
        Returns:
            An anndata.AnnData object (proccessed)
        """

        atac_dict = self.config.get("atac_filtering")

        # Quality control
        sc.pp.calculate_qc_metrics(self.data, percent_top=None, inplace=atac_dict.get("qc_metric_inplace"), log1p=atac_dict.get("qc_metric_log1p"))

        # Filter cells based on QC metrics.
        mu.pp.filter_obs(self.data, "n_genes_by_counts", lambda x: (x >= atac_dict.get("min_peaks_by_counts")) & (x <= atac_dict.get("max_peaks_by_counts")))
        mu.pp.filter_obs(self.data, "total_counts", lambda x: (x >= atac_dict.get("min_total_counts_per_cell")) & (x <= atac_dict.get("max_total_counts_per_cell")))
        
        # Filter peaks based on number of cells where they are present.
        mu.pp.filter_var(self.data, "n_cells_by_counts", lambda x: x < atac_dict.get("max_cells_by_counts"))
        mu.pp.filter_var(self.data, "total_counts", lambda x: x < atac_dict.get("max_total_counts_by_gene"))

        # Perform per-cell normalization.
        sc.pp.normalize_total(self.data, target_sum=atac_dict.get("normalization_target_sum"))
        sc.pp.log1p(self.data)

        # Feature selection
        sc.pp.highly_variable_genes(self.data, n_top_genes=atac_dict.get("n_top_peaks"), subset=True,flavor="seurat")

        return self.data

    def adt_preprocessing(self) -> ad.AnnData:
        """
        Returns:
            An anndata.AnnData object (proccessed)
        """

        adt_dict = self.config.get("adt_filtering")

        # Remove the "total" feature.
        self.data = self.data[:, 1:]
        # Make index of proteins compatible with 10X multiome.
        self.data.obs.index += "-1"

        # Perform per-cell normalization.
        if adt_dict.get("per_cell_normalization"):
            mu.prot.pp.clr(self.data)
        self.data.var["highly_variable"] = True
        self.data.var["feature_types"] = "Protein Expression"
        self.data.var["genome"] = "GRCh38"

        return self.data