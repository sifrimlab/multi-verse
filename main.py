import sys
from train import Trainer
from config import load_config
import torch
torch.cuda.is_available()

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

from utils import GridSearchRun

class Main:
    def __init__(self, config_path="./config.json"):
        """
        Initialize the UserRun class and load configuration.
        """
        self.config = load_config(config_path)
        self.run_user_params = self.config.get("_run_user_params", True)

    def run_default(self):
        """
        Run training and evaluation with user-specified parameters.
        """
        if not self.run_user_params:
            print("User specific parameter run is disabled in the configuration.")
            return
        
        print("\n=== Running User-Specified Parameters ===")
        trainer = Trainer(self.config)
        datasets = trainer.load_datasets()

        print("\n====== Start training ======\n")
        trainer.train()

    def main(self):
        # Check if a config file is provided as a command-line argument
        if len(sys.argv) != 2:
            print("Usage: python run.py <config_file.json>")
            sys.exit(1)

        # Pass the configuration path to the classes
        config_path = sys.argv[1]

        # Run user-specified parameters
        self.run_default()                          # 30 mins for 4 models on 1 dataset

        # Run grid search
        grid_search_run = GridSearchRun(config_path) # 1 hour for 4 models on 1 dataset
        grid_search_run.run()

        print("\n=== Code Run Succesfully ===")

if __name__ == "__main__":
    main = Main(config_path="./config.json")
    main.main()
