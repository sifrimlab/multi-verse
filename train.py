import torch
import json
from torch import optim

from config import load_config
from dataloader import DataLoader

class TrainerSL: #supervised-learning model training
    def __init__(self, model, train_loader, val_loader=None, config_path=None, device=None):
        """
        Initializes the Trainer class.
        Parameters:
        - model_name (str): The name of the model to be used.
        - config_path (str): Path to the JSON configuration file.
        - device (torch.device, optional): Device to run the training on. Defaults to GPU if available.
        """
        self.config = load_config(config_path)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = self.config["training"]['epochs']
        self.learning_rate = self.config["training"]["learning_rate"]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # adaptive momentum estimation
    def train(self):
        """
        Train the model based on hyperparameters from the configuration file.
        """
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for batch in self.train_loader:
                inputs, labels = batch['features'].to(self.device), batch['labels'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / len(self.train_loader)}]")
            if self.val_loader:
                self.validate()

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, labels = batch['features'].to(self.device), batch['labels'].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Validation accuracy: {accuracy:.2f}%")

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")


class TrainerUL: #unsupervised-learning model training
    def __init__(self, model, train_loader, config_path=None, device=None):
        """
        Initializes the Trainer class.

        Parameters:
        - model: The model to be used (e.g., MOFA_Model).
        - train_loader (DataLoader): DataLoader for training data.
        - config_path (str): Path to the JSON configuration file.
        - device (torch.device, optional): Device to run the training on. Defaults to GPU if available.
        """
        self.config = load_config(config_path)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = self.config["training"]["epochs"]
        self.learning_rate = self.config["training"]["learning_rate"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adaptive momentum estimation

    def train(self):
        """
        Train the model based on hyperparameters from the configuration file.
        """
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            # Iterate over training data to perform optimization
            for batch in self.train_loader:
                inputs = batch['features'].to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Assuming MOFA has a specific loss related to latent representation
                loss = self.model.compute_loss(outputs)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Print epoch loss for monitoring
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / len(self.train_loader)}")

    def save_model(self, filepath):
        """
        Save the trained model parameters to the specified filepath.
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")