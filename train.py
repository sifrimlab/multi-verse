import json
from model import PCA_Model, MOFA_Model, Mowgli_Model, MultiVI_Model
from config import load_config
from dataloader import DataLoader
class Trainer(): #model training
    def __init__(self,model):
        """
        Initializes the Trainer class.
        model is an object from one model class in model.py
        """
        self.dataset_path = load_config()["basic"]["data_path"]
        self.dataset_name = load_config()["basic"]["file.name"]
        self.device = load_config()["training"]["device"]
        self.model=model
    def train(self,):
        if isinstance(model,PCA_Model):
            PCA_Model.train()
        elif isinstance(model,MOFA_Model):
            MOFA_Model.train()
        elif isinstance(model,Mowgli_Model):
            Mowgli_Model.train()
        elif isinstance(model,MultiVI_Model):
            MultiVI_Model.train()
        else:
            print("input should be a object of model")