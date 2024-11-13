import json
import torch


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
