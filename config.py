import json

# Module-level cache for the configuration
_config_cache = None

def load_config(config_path="./config.json"):
    """
    Load the configuration from a JSON file, ensuring it's loaded only once.
    Subsequent calls return the cached configuration.

    Parameters:
    - config_path (str): Path to the JSON configuration file.

    Returns:
    - dict: Dictionary of hyperparameters and settings.
    """
    global _config_cache
    if _config_cache is None:
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                _config_cache = json.load(file)
            print("Information form json file loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration file: {e}")
    return _config_cache