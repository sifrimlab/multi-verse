import json
import os
import anndata as ad
from model import PCA_Model, MOFA_Model, MultiVI_Model, Mowgli_Model
from config import load_config
from dataloader import DataLoader

class Trainer: 
    def __init__(self, config_path: str="./config.json", hyperparams=None):
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
        self.data = None
        print("Datasets Detected:", self.dataset_names)

         # Update the configuration with hyperparameters, if provided
        if hyperparams:
            for model_name, params in hyperparams.items():
                if model_name in self.config["model"]:
                    valid_keys = self.config["model"][model_name].get("grid_search_params", {}).keys()
                    filtered_params = {k: v for k, v in params.items() if k in valid_keys}
                    self.config["model"][model_name].update(filtered_params)
                    print(f"Updated parameters for {model_name}: {filtered_params}")

    def load_datasets(self):
        """
        Load all datasets specified in the configuration.
        Returns a dictionary where keys are dataset names, and values are the data objects.
        """
        datasets = {}
        for dataset_name in self.dataset_names:
            dataset_info = self.data_config[dataset_name]
            # Hard-code prostate data
            if dataset_name != "dataset_prostate":
                modality_list = [
                    key for key, value in dataset_info.items()
                    if isinstance(value, dict) and "file_name" in value # modality (i.e.'rna') must be a dictionary
                ]
            else:
                modality_list = None
            dataset_path = dataset_info["data_path"]
            list_anndata = []
            # Assume there will be no modality when loading Prostate data, since loading separated modality will lose annotation metadata of this dataset
            if modality_list != None:
                for modality in modality_list:
                    modality_info = dataset_info[modality]
                    file_path = os.path.join(dataset_path, modality_info["file_name"])
                    is_preprocessed = modality_info["is_preprocessed"]
                    annotation = modality_info["annotation"]
                    ann_loader = DataLoader(
                        file_path=file_path,
                        modality=modality,
                        isProcessed=is_preprocessed,
                        annotation= annotation,
                        config_path=self.config_path,
                    )
                    ann =ann_loader.preprocessing()
                    list_anndata.append(ann)
                datasets[dataset_name] = {
                    "modalities": modality_list,
                    "data": list_anndata
                }
            else: 
                # This for loading Prostate data as whole MuData object, this looks cumbersome I know I'm just stupid
                file_path = os.path.join(dataset_path, dataset_info["file_name"])
                is_preprocessed = dataset_info["is_preprocessed"]
                annotation = dataset_info["annotation"]
                prostate = DataLoader(file_path=file_path,
                                      isProcessed=is_preprocessed,
                                      annotation= annotation,
                                      config_path=self.config_path,
                                      ).read_mudata()
                datasets[dataset_name] = {
                    "data": prostate
                }
        self.data = datasets
        return self.data
    
    def dataset_select(self, datasets_dict, data_type: str = ""):
        """
        Concatenate list of AnnDatas or Fuse list of AnnDatas into one MuData
        """
        dataloader = DataLoader(config_path=self.config_path)
        datasets = datasets_dict

        if data_type=="concatenate": # Process input object for PCA and MultiVI
            concatenate = {}
            for dataset_name, dataset_data in datasets.items():
                print(f"\n=== Concatenating dataset: {dataset_name} ===")
                if "modalities" in dataset_data: # Either pbmck_10k or TEA dataset, prostate data is assumed to be loaded as whole (no modalities)
                    modalities = dataset_data["modalities"]
                    list_anndata = dataset_data["data"]
                    data_concat = dataloader.anndata_concatenate(list_anndata=list_anndata, list_modality=modalities)
                    concatenate[dataset_name] = data_concat
                else:
                    # If loaded data is Prostate MuData (no modality)
                    prostate = dataset_data["data"]
                    list_anndata = list(prostate.mod.values())
                    modalities = list(prostate.mod.keys())
                    data_concat = dataloader.anndata_concatenate(list_anndata=list_anndata, list_modality=modalities)
                    # Hard-code annotation of prostate
                    data_concat.obs["new_ann"] = prostate.obs["new_ann"] 
                    concatenate[dataset_name] = data_concat
            self.data = concatenate
        elif data_type =="mudata":   # Process input object for MOFA+ and Mowgli
            mudata_input = {}
            for dataset_name, dataset_data in datasets.items():
                print(f"\n=== Fusing dataset as MuData object: {dataset_name} ===")
                if "modalities" in dataset_data:
                    modalities = dataset_data["modalities"]
                    list_anndata = dataset_data["data"]
                    data_fuse = dataloader.fuse_mudata(list_anndata=list_anndata, list_modality=modalities)
                    mudata_input[dataset_name] = data_fuse
                else:
                    # If loaded data is Prostate MuData (no modality)
                    prostate = dataset_data["data"]
                    mudata_input[dataset_name] = prostate
            self.data = mudata_input
        else:
            raise ValueError("Only accept datatype of concatenate or mudata.")
        return self.data
    
    def model_select(self, dataset_dict, **kwargs):
        """
        Initialize models for a specific dataset.
        """
        datasets = dataset_dict          # dataset dictionary after load_datasets()
        data_concat = self.dataset_select(datasets_dict=datasets, data_type="concatenate")
        data_mudata = self.dataset_select(datasets_dict=datasets, data_type="mudata")
        models_for_data = {}
        is_gridsearch = kwargs.get('is_gridsearch', False)  # Default to False if not provided


        for dataset_name in self.dataset_names:
            models = {}
            if self.model_info["is_pca"]:
                pca = PCA_Model(dataset=data_concat[dataset_name], dataset_name=dataset_name, 
                                config_path=self.config_path, is_gridsearch=is_gridsearch)
                models["pca"] = pca

            if self.model_info["is_multivi"]:
                multivi = MultiVI_Model(dataset=data_concat[dataset_name], dataset_name=dataset_name, 
                                        config_path=self.config_path, is_gridsearch=is_gridsearch)
                models["multivi"] = multivi

            if self.model_info["is_mofa+"]:
                mofa = MOFA_Model(dataset=data_mudata[dataset_name], dataset_name=dataset_name, 
                                  config_path=self.config_path, is_gridsearch=is_gridsearch)
                models["mofa"] = mofa

            if self.model_info["is_mowgli"]:
                mowgli = Mowgli_Model(dataset=data_mudata[dataset_name], dataset_name=dataset_name, 
                                      config_path=self.config_path, is_gridsearch=is_gridsearch)
                models["mowgli"] = mowgli

            models_for_data[dataset_name] = models 
        self.models = models_for_data
        return self.models

    def train(self):
        try:
            if self.models==None:
                if self.data==None:
                    self.load_datasets()
                self.model_select(self.data)

            for dataset_name, model_dict in self.models.items():
                print(f"\n=== Training for {dataset_name} ===")
                for model_name, model in model_dict.items():
                    print(f"\n=== {model_name} training ===")
                    model.update_output_dir() # This will create {model_name}_output folder
                    model.to()
                    model.train()
                    model.save_latent()
                    model.umap()
        except ValueError as e:
            print(f"Something is wrong in train() function: {e}")
