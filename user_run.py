from train import Trainer
from config import load_config


class UserRun:
    def __init__(self, config_path="./config.json"):
        """
        Initialize the UserRun class and load configuration.
        """
        self.config = load_config(config_path)

    def run(self):
        """
        Run training and evaluation with user-specified parameters.
        """
        print("\n=== Running User-Specified Parameters ===")
        trainer = Trainer(self.config)
        datasets = trainer.load_datasets()

        models_by_dataset = trainer.model_select(datasets)

        for dataset_name, models in models_by_dataset.items():
            for model_name, model in models.items():
                print(f"\nTraining {model_name} on {dataset_name} with user-specified parameters...")
                model.to()
                model.train()
                model.umap()
                model.save_latent()
