import json
from model import PCA_Model, MOFA_Model, Mowgli_Model, MultiVI_Model
from config import load_config
from dataloader import DataLoader
class Trainer(): #model training
    def __init__(self):
        """
        Initializes the Trainer class.
        model is an object from one model class in model.py
        """
        self.dataset_path = load_config()["basic"]["data_path"]
        self.dataset_name = load_config()["basic"]["file.name"]
        self.device = load_config()["training"]["device"]
        self.model=model
    def train(self,):
        self.model.train()