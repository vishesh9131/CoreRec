# cross_domain implementation
import torch
import torch.nn as nn

class CROSS_DOMAIN:
    def __init__(self, source_model, target_model):
        """
        Initialize cross-domain learning.
        
        Args:
            source_model: Model trained on the source domain
            target_model: Model to be trained on the target domain
        """
        self.source_model = source_model
        self.target_model = target_model

    def transfer_knowledge(self, data_loader, criterion, optimizer, num_epochs):
        """
        Transfer knowledge from source to target domain.
        
        Args:
            data_loader: DataLoader for target domain data
            criterion: Loss function
            optimizer: Optimizer
            num_epochs: Number of training epochs
        """
        self.source_model.eval()
        self.target_model.train()

        for epoch in range(num_epochs):
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                with torch.no_grad():
                    source_features = self.source_model(inputs)
                outputs = self.target_model(source_features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    def evaluate(self, data_loader, criterion):
        """
        Evaluate the target model on the target domain.
        
        Args:
            data_loader: DataLoader for evaluation data
            criterion: Loss function
        """
        self.target_model.eval()

        total_loss = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self.target_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        
        print(f"Evaluation Loss: {total_loss / len(data_loader):.4f}")
