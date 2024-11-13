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

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                loss = self._evaluate_batch(batch)
                total_loss += loss.item()
        return total_loss / len(self.val_dataloader)
