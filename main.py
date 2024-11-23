from config import load_config
from dataloader import DataLoader
from model import PCA_Model, MOFA_Model, Mowgli_Model, MultiVI_Model
from train import Trainer


class Main:
    """
    please input directory and filename for your dataset
    please choose model by setting "is_model"=true or false
    """
    def __init__(self) -> None:
        pass

    def run():
        if load_config()["training"]["pca"]["is_pca"]==True:
            pca=PCA_Model()
            trainer=Trainer(pca)
            trainer.train()
        if load_config["training"]["mofa+"]["is_mofa+"]==True:
            mofa=MOFA_Model()
            train=Trainer(mofa)
            trainer.train()
        if load_config["training"]["mowgli"]["is_mowgli"] ==True:
            mowgli=Mowgli_Model()
            trainer=Trainer(mowgli)
            trainer.train()
        if load_config["training"]["multivi"]["is_multivi"]==True:
            multivi=MultiVI_Model()
            trainer=Trainer(multivi)
            trainer.train()

    

if __name__ == "__main__":
    main = Main()
    main.run()
