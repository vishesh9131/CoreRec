"""
train.py

This module provides a function to train models with options for custom loss functions and training procedures.

Functions:
    train_model(model, data_loader, criterion, optimizer, num_epochs): Trains the model using the provided data loader, loss function, and optimizer.

Usage:
    from engine.train import train_model

    # Example usage
    model = GraphTransformer(num_layers=2, d_model=128, num_heads=4, d_feedforward=512, input_dim=10)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, data_loader, criterion, optimizer, num_epochs=10)
"""

def train_model(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            inputs = inputs.float()
            targets = targets.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")