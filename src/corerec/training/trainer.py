"""
Unified Trainer

Standardized training infrastructure for all CoreRec models.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import List, Optional, Callable, Any
import torch
from torch.utils.data import DataLoader
from corerec.training.callbacks import Callback, EarlyStopping


class Trainer:
    """
    Unified trainer for all CoreRec models.

    Provides standardized training with callbacks, monitoring, and best practices.

    Example:
        from corerec.training import Trainer, EarlyStopping, ModelCheckpoint

        trainer = Trainer(
            model=ncf_model,
            optimizer=torch.optim.Adam(ncf_model.parameters()),
            loss_fn=torch.nn.BCELoss(),
            callbacks=[
                EarlyStopping(patience=5),
                ModelCheckpoint('best_model.pt')
            ]
        )

        trainer.train(train_loader, val_loader, epochs=100)

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        callbacks: Optional[List[Callback]] = None,
        device: str = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance
            loss_fn: Loss function
            callbacks: List of callbacks
            device: Device to train on ('cuda', 'cpu', or None for auto-detect)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Move model to device
        if hasattr(model, "to"):
            self.model = self.model.to(self.device)

    def train(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10
    ) -> Any:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs

        Returns:
            Trained model

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Callback: training begin
        for cb in self.callbacks:
            cb.on_train_begin()

        for epoch in range(epochs):
            # Callback: epoch begin
            logs = {}
            for cb in self.callbacks:
                cb.on_epoch_begin(epoch, logs)

            # Update learning rate if specified in logs
            if "learning_rate" in logs:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = logs["learning_rate"]

            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)
            logs["train_loss"] = train_loss

            # Validation phase
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                logs["val_loss"] = val_loss

            # Callback: epoch end
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs)

            # Print progress
            log_str = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}"
            if val_loader:
                log_str += f", Val Loss: {logs['val_loss']:.4f}"
            print(log_str)

            # Check early stopping
            early_stop_callbacks = [cb for cb in self.callbacks if isinstance(cb, EarlyStopping)]
            if any(cb.stop_training for cb in early_stop_callbacks):
                print("Early stopping triggered")
                break

        # Callback: training end
        for cb in self.callbacks:
            cb.on_train_end(logs)

        return self.model

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Callback: batch begin
            for cb in self.callbacks:
                cb.on_batch_begin(batch_idx)

            # Training step
            loss = self._train_step(batch)
            total_loss += loss.item()
            num_batches += 1

            # Callback: batch end
            batch_logs = {"batch_loss": loss.item()}
            for cb in self.callbacks:
                cb.on_batch_end(batch_idx, batch_logs)

        return total_loss / num_batches if num_batches > 0 else 0

    def _train_step(self, batch) -> torch.Tensor:
        """
        Single training step.

        Override this method for custom training logic.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Default implementation
        # Assumes batch is (inputs, targets)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch

            # Move to device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            return loss
        else:
            raise ValueError(
                "Batch format not recognized. Override _train_step() for custom logic."
            )

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Validate for one epoch.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Get inputs and targets
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch

                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(self.device)
                    if isinstance(targets, torch.Tensor):
                        targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0
