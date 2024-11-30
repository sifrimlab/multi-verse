import sys
import torch
torch.cuda.is_available()

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

from user_run import UserRun
from gridsearch_run import GridSearchRun

def main():
    # Check if a config file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python run.py <config_file.json>")
        sys.exit(1)

    # Pass the configuration path to the classes
    config_path = sys.argv[1]

    # Run user-specified parameters
    user_run = UserRun(config_path)
    user_run.run()

    # Run grid search
    grid_search_run = GridSearchRun(config_path)
    grid_search_run.run()

    print("\n=== Code Run Succesfully ===")

if __name__ == "__main__":
    main()
