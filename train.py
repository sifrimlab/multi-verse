import json
import os
from model import PCA_Model, MOFA_Model, MultiVI_Model, Mowgli_Model
from config import load_config
from dataloader import DataLoader

class Trainer: 
    def __init__(self, config_path: str="./config.json"):
        """
        Initializes the Trainer class.
        model is an object from one model class in model.py
        """
        self.config_path = config_path
        self.config = load_config(config_path=config_path)

        # Model information from config file
        self.model_info = self.config.get("model")
        excluded_keys = ["is_pca", "is_mofa+", "is_multivi", "is_mowgli"]
        self.model_available = [key for key in self.model_info.keys() if key not in excluded_keys]
        self.models=None

        # Data information from config file
        self.data_config = self.config.get("data")
        self.dataset_names = [
            key for key, value in self.data_config.items()
            if isinstance(value, dict) and "data_path" in value
        ]
        print("Datasets Detected:", self.dataset_names)


    def load_datasets(self):
        """
        Load all datasets specified in the configuration.
        Returns a dictionary where keys are dataset names, and values are the data objects.
        """
        datasets = {}
        for dataset_name in self.dataset_names:
            dataset_info = self.data_config[dataset_name]
            modality_list = [
                key for key in dataset_info.keys()
                if key not in ["data_path", "dataset_name"]
            ]
            dataset_path = dataset_info["data_path"]
            list_anndata = []
            for modality in modality_list:
                modality_info = dataset_info[modality]
                file_path = os.path.join(dataset_path, modality_info["file_name"])
                is_preprocessed = modality_info["is_preprocessed"]
                ann = DataLoader(
                    file_path=file_path,
                    modality=modality,
                    isProcessed=is_preprocessed,
                    config_path=self.config_path,
                ).preprocessing()
                list_anndata.append(ann)
            datasets[dataset_name] = {
                "modalities": modality_list,
                "data": list_anndata
            }
        return datasets
    
    def dataset_select(self, data_type: str = ""):
        """
        Concatenate list of AnnDatas or Fuse list of AnnDatas into one MuData or Load one MuData
        self.modalities=['rna', 'atac']
        """
        dataloader = DataLoader(config_path=self.config_path)
        if data_type=="concatenate":
            list_modality, list_anndata = self.load_dataset()
            data_concat = dataloader.anndata_concatenate(list_anndata=list_anndata, list_modality=list_modality)
            self.data = data_concat
        elif data_type=="mudata":
            if self.modality_list != None:
                list_modality, list_anndata = self.load_dataset()
                data_fuse = dataloader.fuse_mudata(list_anndata=list_anndata, list_modality=list_modality)
                self.data = data_fuse
            else:
                self.load_dataset()
        else:
            raise ValueError("Only accept datatype of concatenate or mudata.")
        return self.data
    
    def model_select(self, dataset_name, dataset_data):
        """
        Initialize models for a specific dataset.
        """
        models = {}
        modalities = dataset_data["modalities"]
        list_anndata = dataset_data["data"]
        dataloader = DataLoader(config_path=self.config_path)
        data_concat = dataloader.anndata_concatenate(list_anndata=list_anndata, list_modality=modalities)
        data_mudata = dataloader.fuse_mudata(list_anndata=list_anndata, list_modality=modalities)

        for model_name in self.model_available:
            if model_name == "pca" and self.model_info["is_pca"]:
                # PCA use concatenated AnnData object
                models[model_name]=PCA_Model(dataset=data_concat, dataset_name=dataset_name, config_path=self.config_path)
            if model_name == "mofa+" and self.model_info["is_mofa+"]:
                # MOFA+ use MuData object
                models[model_name]=MOFA_Model(dataset=data_mudata, dataset_name=dataset_name, config_path=self.config_path)
            if model_name == "multivi" and self.model_info["is_multivi"]:
                # Multivi use concatenated AnnData object
                models[model_name]=MultiVI_Model(dataset=data_concat, dataset_name=dataset_name, config_path=self.config_path)
            if model_name == "mowgli" and self.model_info["is_mowgli"]:
                # Mowgli use MuData object
                models[model_name]=Mowgli_Model(dataset=data_mudata, dataset_name=dataset_name, config_path=self.config_path)

        self.models = models
        return self.models

    def train(self):
        try:
            if self.models ==None:
                self.model_select()

            for model_name, model in self.models.items():
                model.train()
        except ValueError as e:
            print(f"Something is wrong in train() function: {e}")



    