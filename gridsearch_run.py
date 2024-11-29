from itertools import product
from train import Trainer
from config import load_config

class GridSearchRun:
    def __init__(self, config_path="./config.json"):
        """
        Initialize the GridSearchRun class and load configuration.
        """
        self.config = load_config(config_path)
        self.run_gridsearch = self.config.get("_run_gridsearch", True)

        # Preload datasets once
        print("\n=== Loading and Preprocessing Datasets ===")
        self.trainer = Trainer(self.config)
        self.datasets = self.trainer.load_datasets()  # Load datasets once

    @staticmethod
    def generate_param_combinations(params):
        """
        Generate all combinations of parameters for grid search.
        """
        keys, values = zip(*params.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        return combinations

    def run(self):
        """
        Run training and evaluation using grid search for each model and dataset.
        """
        if not self.run_gridsearch:
            print("GridSearch is disabled in the configuration.")
            return

        print("\n=== Running Grid Search Hyperparameter Optimization ===")

        # Store the best results for summary
        best_results = {}

        for model_name, model_config in self.config["model"].items():
            # Skip boolean flags or other non-dict entries
            if not isinstance(model_config, dict):
                continue

            # Extract grid search parameters
            grid_params = model_config.get("grid_search_params", {})
            if not grid_params:
                print(f"No grid search parameters found for {model_name}. Skipping.")
                continue

            param_combinations = self.generate_param_combinations(grid_params)
            print(f"\n=== Running grid search for {model_name} with {len(param_combinations)} combinations...")

            # Initialize results storage for this model
            best_results[model_name] = {}

            for dataset_name in self.datasets.keys():
                print(f"\n=== Running {model_name} on {dataset_name} ===")

                best_score = float("-inf")
                best_params = None
                best_model = None

                # Loop through parameter combinations for this dataset
                for params in param_combinations:
                    # Update config with parameters for the current model
                    temp_config = self.config.copy()
                    temp_config["model"][model_name].update(params)

                    # Initialize Trainer with updated parameters
                    trainer = Trainer(temp_config, {model_name: {**model_config, **params}})
                    models_by_dataset = trainer.model_select(self.datasets, is_gridsearch=True)

                    # Train and evaluate only the current model for the current dataset
                    model = models_by_dataset[dataset_name].get(model_name)
                    if not model:
                        print(f"No model found for {model_name} on {dataset_name}. Skipping.")
                        continue

                    print(f"Training {model_name} on {dataset_name} with parameters: {params}")
                    model.to()
                    model.train()

                    # Evaluate the model
                    try:
                        score = model.evaluate_model()
                    except ValueError as e:
                        print(f"Error evaluating {model_name} on {dataset_name}: {e}")
                        continue

                    # Update the best model for this dataset
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_model = model

                # Save the best model for this dataset
                if best_model:
                    print(
                        f"\n=== Best result for {model_name} on {dataset_name}: "
                        f"Score = {best_score}, Params = {best_params}"
                    )
                    best_model.umap()
                    best_model.save_latent()

                    # Store the best result in the summary
                    best_results[model_name][dataset_name] = {
                        "score": best_score,
                        "params": best_params,
                    }

        # Print the summary
        print("\n=== Grid Search Summary ===")
        for model_name, dataset_results in best_results.items():
            for dataset_name, result in dataset_results.items():
                print(
                    f"Best result for {model_name} on {dataset_name}: "
                    f"Score = {result['score']}, Params = {result['params']}"
                )
