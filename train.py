import torch


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, config, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epochs = config['epochs']

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            print(
                f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
