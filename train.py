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
        # Data information from config file
        self.config = load_config(config_path=config_path)
        self.data_info = self.config.get("data")
        self.dataset_path = self.data_info.get("data_path")
        self.dataset_name = self.data_info.get("dataset_name")
        self.modality_list = [key for key in self.data_info.keys() if key != "data_path" and key != "dataset_name"]
        self.data=None

        # Model information from config file
        self.model_info = self.config.get("model")
        excluded_keys = ["is_pca", "is_mofa+", "is_multivi", "is_mowgli"]
        self.model_available = [key for key in self.model_info.keys() if key not in excluded_keys]
        self.models=None

    def load_dataset(self):
        """
        Load dataset to model as list of AnnDatas or one MuData
        self.modalities=['rna', 'atac']
        """
        list_modality=self.modality_list
        list_anndata=[]
        if list_modality != None:
            for i, modality in enumerate(list_modality):
                # Load modality info from config
                mod_info = self.data_info.get(modality)
                filename=mod_info.get("file_name")
                filepath = os.path.join(self.dataset_path, filename)
                is_preprocess=mod_info.get("is_preprocessed")

                # Load anndatas
                ann = DataLoader(file_path=filepath, modality=modality, isProcessed=is_preprocess).preprocessing()
                list_anndata.append(ann)
            return list_modality, list_anndata
        else:
            # If dataset is only one MuData
            filename = self.data_info.get("file_name")
            filepath = os.path.join(self.dataset_path, filename)
            is_preprocess = self.data_info.get("is_preprocessed")
            mu = DataLoader(file_path=filepath, isProcessed=is_preprocess).read_mudata()
            self.data = mu
            return self.data
    
    def dataset_select(self, data_type: str = ""):
        """
        Concatenate list of AnnDatas or Fuse list of AnnDatas into one MuData or Load one MuData
        self.modalities=['rna', 'atac']
        """
        dataloader = DataLoader()
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
    
    def model_select(self):
        models = {} # Use a dictionary to store models
        for model_name in self.model_available:
            if model_name == "pca" and self.model_info["is_pca"]:
                # PCA use concatenated AnnData object
                data_pca = self.dataset_select(data_type="concatenate")
                models[model_name]=PCA_Model(dataset=data_pca, dataset_name=self.dataset_name)
            if model_name == "mofa+" and self.model_info["is_mofa+"]:
                # MOFA+ use MuData object
                data_mofa = self.dataset_select(data_type="mudata")
                models[model_name]=MOFA_Model(dataset=data_mofa, dataset_name=self.dataset_name)
            if model_name == "multivi" and self.model_info["is_multivi"]:
                # Multivi use concatenated AnnData object
                data_multivi = self.dataset_select(data_type="concatenate")
                models[model_name]=MultiVI_Model(dataset=data_multivi, dataset_name=self.dataset_name)
            if model_name == "mowgli" and self.model_info["is_mowgli"]:
                # Mowgli use MuData object
                data_mowgli = self.dataset_select(data_type="mudata")
                models[model_name]=Mowgli_Model(dataset=data_mowgli, dataset_name=self.dataset_name)

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



    