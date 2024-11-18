import json

def load_config(config_path="./config.json"):
    """
    Load the configuration from a JSON file.

    Parameters:
    - config_path (str): Path to the JSON configuration file.

    Returns:
    - dict: Dictionary of hyperparameters and settings.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        print("Loaded configuration successfully.")
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration file: {e}")
