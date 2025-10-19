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

import torch
import logging
from torch.utils.data import DataLoader
from typing import Any

def train_model(model: torch.nn.Module,
               data_loader: DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               num_epochs: int,
               device: torch.device = torch.device('cpu'),
               gradient_clip_value: float = 1.0,
               early_stopping_patience: int = 5) -> None:
    """
    Trains the given model using the provided data loader, criterion, and optimizer.
    Includes comprehensive error handling, gradient clipping, and early stopping.

    Args:
        model (torch.nn.Module): The neural network model to train.
        data_loader (DataLoader): DataLoader providing the training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        num_epochs (int): Number of training epochs.
        device (torch.device, optional): Device to run the training on. Defaults to CPU.
        gradient_clip_value (float, optional): Value for gradient clipping. Defaults to 1.0.
        early_stopping_patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 5.
    """
    print("Starting training process")
    model.to(device)

    best_loss = float('inf')
    epochs_no_improve = 0

    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (inputs, batch_adj, targets) in enumerate(data_loader, 1):
                try:
                    # Move data to the specified device
                    inputs = inputs.to(device).float()
                    batch_adj = batch_adj.to(device).float()
                    targets = targets.to(device).float()

                    # Reshape inputs if necessary
                    if inputs.dim() == 2:
                        inputs = inputs.unsqueeze(-1)  # [batch_size, num_nodes, 1]

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass with all required arguments
                    outputs = model(inputs, batch_adj, batch_adj)  # Assuming graph_metrics = batch_adj

                    # Check output dimensions
                    if outputs.shape != targets.shape:
                        print(
                            f"Warning: Shape mismatch at Epoch {epoch + 1}, Batch {batch_idx}: "
                            f"outputs shape {outputs.shape} vs targets shape {targets.shape}"
                        )

                    # Compute loss
                    loss = criterion(outputs, targets)

                    # Backward pass
                    loss.backward()

                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

                    # Update parameters
                    optimizer.step()

                    # Accumulate loss
                    epoch_loss += loss.item()
                    batch_count += 1

                    # Print loss every 100 batches
                    if batch_idx % 100 == 0:
                        print(
                            f"Epoch [{epoch + 1}/{num_epochs}], "
                            f"Batch [{batch_idx}], Loss: {loss.item():.4f}"
                        )

                except Exception as batch_exc:
                    print(
                        f"Error in Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}]: {str(batch_exc)}"
                    )
                    # Optionally, continue training despite batch errors
                    continue

            # Average loss for the epoch
            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

                # Check for early stopping
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    epochs_no_improve = 0
                    # Save the best model
                    torch.save(model.state_dict(), 'model_best.pth')
                    print(f"New best loss: {best_loss:.4f}. Model saved.")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement in loss for {epochs_no_improve} epoch(s).")

                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
            else:
                print(f"Epoch [{epoch + 1}/{num_epochs}] skipped due to no successful batches.")

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    except Exception as e:
        print(f"Critical error during training: {str(e)}")
        raise e  # Re-raise exception after printing

    finally:
        # Save the final model state at the end of training
        try:
            torch.save(model.state_dict(), 'model_final.pth')
            print("Final model state saved successfully.")
        except Exception as save_exc:
            print(f"Failed to save the final model state: {str(save_exc)}")
