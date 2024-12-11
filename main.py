<<<<<<< HEAD
import dataloader
import eval
import model
import train
import utils
import ProcessingRNA
import PrepocessingATAC


filename = "brain3k_multiome"

mdata = dataloader.func1(filename)

ProcessingRNA.Processing_RNA(mdata)
PrepocessingATAC.Prepocessing_ATAC(mdata)

mdata_new = train.train(mdata)
mdata_new.write("data/brain3k_processed.h5mu")
=======
import sys
from train import Trainer
from eval import Evaluator
from config import load_config
import torch
torch.cuda.is_available()

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")
from utils import GridSearchRun


def main():
    # Check if a config file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python run.py <config_file.json>")
        sys.exit(1)

    # Pass the configuration path to the classes
    config_path = sys.argv[1]
    config = load_config(config_path)
    run_user_params = config.get("_run_user_params", True)
    trainer = Trainer(config)

    # Run user-specified parameters
    # 30 mins for 4 models on 1 dataset, default run
    if not run_user_params:
        print("User specific parameter run is disabled in the configuration.")
    else:
        print("\n=== Running User-Specified Parameters ===")
        datasets = trainer.load_datasets()

        print("\n====== Start training ======\n")
        trainer.train()                    

        evaluator = Evaluator(latent_dir="./outputs", output_file="./outputs/results.json", trainer=trainer)
        evaluator.process_models(config_path=config_path) # Config file here to check for annotation

    # Run grid search, if "_run_gridsearch" = true
    # 1 hour for 4 models on 1 dataset
    grid_search_run = GridSearchRun(config_path) 
    grid_search_run.run()

    print("\n=== Code Run Succesfully ===")

if __name__ == "__main__":
    main()
>>>>>>> ffffc021418819b07185abcbb534838868f214db
