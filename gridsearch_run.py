from itertools import product
from train import Trainer
from config import load_config


class GridSearchRun:
    def __init__(self, config_path="./config.json"):
        """
        Initialize the GridSearchRun class and load configuration.
        """
        self.config = load_config(config_path)

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
        Run training and evaluation using grid search.
        """
        print("\n=== Running Grid Search... ===")

        results_summary = {}  # Store all results
        for model_name, model_config in self.config["model"].items():
            # Skip boolean flags like "is_pca", "is_mofa+", etc.
            if not isinstance(model_config, dict):
                continue

            # Extract grid search parameters
            grid_params = model_config.get("grid_search_params", {})
            if not grid_params:
                print(f"No grid search parameters found for {model_name}. Skipping.")
                continue

            param_combinations = self.generate_param_combinations(grid_params)
            print(f"\nRunning grid search for {model_name} with {len(param_combinations)} combinations...")

            best_score = float("-inf")
            best_params = None
            best_model = None
            best_dataset_name = None

            # Loop through datasets and test all parameter combinations
            for params in param_combinations:
                combined_params = {**model_config, **params}  # Merge default and grid params
                trainer = Trainer(self.config, {model_name: combined_params})
                datasets = trainer.load_datasets()

                models_by_dataset = trainer.model_select(datasets, is_gridsearch=True)
                for dataset_name, models in models_by_dataset.items():
                    for model_key, model in models.items():
                        print(f"Training {model_key} on {dataset_name} with parameters: {params}")
                        model.to()
                        model.train()

                        # Evaluate the model
                        try:
                            score = model.evaluate_model()
                        except ValueError as e:
                            print(f"Error evaluating {model_key} on {dataset_name}: {e}")
                            continue

                        # Save results and update the best model
                        results_summary[(dataset_name, model_key, tuple(params.items()))] = score
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_model = model
                            best_dataset_name = dataset_name

            # Save UMAP and latent for the best model
            if best_model:
                print(f"\n=== Best model for {model_name}:")
                print(f"Dataset: {best_dataset_name}, Parameters: {best_params}, Score: {best_score}")
                best_model.umap()
                best_model.save_latent()

        # Summarize all results
        print("\n=== Grid Search Summary ===")
        for (dataset_name, model_key, params), score in results_summary.items():
            print(f"Dataset: {dataset_name}, Model: {model_key}, Parameters: {dict(params)}, Score: {score}")

        return results_summary
