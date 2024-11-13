import torch


class Evaluator:
    def __init__(self, model, dataloader, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.dataloader:
                loss = self._evaluate_batch(batch)
                total_loss += loss.item()
        return total_loss / len(self.dataloader)

    def _evaluate_batch(self, batch):
        # Compute and return batch loss
        return 0
