from config import load_config
from dataloader import DataLoaderFactory
from model import ModelFactory
from train import Trainer
import torch


class Main:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelFactory(self.config).get_model().to(self.device)
        self.dataloader_factory = DataLoaderFactory(self.config)

        # Prepare train and validation dataloaders
        self.train_dataloader = self.dataloader_factory.get_dataloader('train')
        self.val_dataloader = self.dataloader_factory.get_dataloader('val')

        # Initialize Trainer with both train and validation dataloaders
        self.trainer = Trainer(self.model, self.train_dataloader,
                               self.val_dataloader, self.config, self.device)

    def run(self):
        # Move training process to Trainer
        self.trainer.train()


if __name__ == "__main__":
    main = Main(config_path="config.json")
    main.run()
