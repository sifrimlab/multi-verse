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
        trainer = Trainer(self.config)
        # Store the best results for summary
        best_results = {}

        # Trainer has been initialized ONCE as self.trainer
        # Load selected models for selected datasets (i.e. 2 datasets will have 2 PCAs, 2 mofa+,...)
        models_on_dataset_dict = self.trainer.model_select(self.datasets)
        print(models_on_dataset_dict)
        # models_on_dataset_dict = {'dataset_Pbmc10k': {'pca': <model.PCA_Model>,'multivi': <model.MultiVI_Model>},
                                # 'dataset_TEA': {'pca': <model.PCA_Model>,'multivi': <model.MultiVI_Model>}}
        # If you looping ALL models from config then looping dataset, then there will be error if NOT ALL models are selected for datasets
        # (i.e what if I only want PCA and Mowgli)

        for dataset_name, models_dict in models_on_dataset_dict.items():

            best_results[dataset_name] = {}

            for model_name, model_obj in models_dict.items():               # model_obj is specific model loaded for specific dataset
                model_config=self.config["model"][model_name]               # Grid_search parameter based on model_name in the selected ones
                # Skip boolean flags or other non-dict entries             
                if not isinstance(model_config, dict):
                    continue
                
                # Extract grid search parameters
                grid_params = model_config.get("grid_search_params", {})
                if not grid_params:
                    print(f"No grid search parameters found for {model_name}. Skipping.")
                    continue
                else:
                    # Could give error ig grid_params is None, so I put this in else-statement
                    param_combinations = self.generate_param_combinations(grid_params) 
                    print(f"\n=== Running grid search for {model_name} on {dataset_name} with {len(param_combinations)} hyperparameter settings")

                    # Initialize results storage for this model on this dataset
                    best_score = float("-inf")
                    best_params = None
                    best_model = None

                    # Loop through parameter combinations for this dataset
                    for params in param_combinations:
                        # Update config with parameters for the CURRENT model (which is model_obj)
                        model_obj.update_parameters(**params)      # This will update new parameters instead of re-initialize it

                        # Train and evaluate only the current model for the current dataset
                        print(f"Training {model_name} on {dataset_name} with parameters: {params}")
                        model_obj.to()
                        model_obj.train()
                        
                        # Evaluate the model
                        try:
                            score = model_obj.evaluate_model()
                        except ValueError as e:
                            print(f"Error evaluating {model_name} on {dataset_name}: {e}")
                            continue

                        # Update the best model for this dataset
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_model = model_obj

                    # Save the best (PCA/Mofa/Multivi) model for this dataset
                    if best_model:
                        print(
                            f"\n=== Best result for {model_name} on {dataset_name}: "
                            f"Score = {best_score}, Params = {best_params}"
                        )
                        best_model.is_grid_search=True             # Re-setting for current model
                        best_model.update_output_dir()            # This will store result in gridsearch_output folder
                        best_model.umap()
                        best_model.save_latent()

                        # Store the best result in the summary
                        best_results[dataset_name][model_name] = {
                            "score": best_score,
                            "params": best_params,
                        }
        # Print the summary
        print("\n=== Grid Search Summary ===")
        if best_results != None:
            for model_name, dataset_results in best_results.items():
             for dataset_name, result in dataset_results.items():
                    print(
                        f"Best result for {model_name} on {dataset_name}: "
                        f"Score = {result['score']}, Params = {result['params']}"
                    )
        else:
            print("\n=== No result for GridSearch ===")