"""
Weights & Biases Integration

Experiment tracking with W&B.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import Dict, Any, Optional
import logging


class WandBTracker:
    """
    Weights & Biases integration for experiment tracking.

    Example:
        from corerec.integrations import WandBTracker

        tracker = WandBTracker(project="corerec-experiments")
        tracker.init(config={'embedding_dim': 64})

        # Training...
        tracker.log({'train_loss': 0.5, 'val_loss': 0.6})

        tracker.log_model('model.pt')
        tracker.finish()

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, project: str, entity: Optional[str] = None):
        """
        Initialize W&B tracker.

        Args:
            project: W&B project name
            entity: W&B entity (username or team)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.project = project
        self.entity = entity
        self.logger = logging.getLogger("WandBTracker")

        try:
            import wandb

            self.wandb = wandb
            self.enabled = True
        except ImportError:
            self.logger.warning("W&B not installed. Install with: pip install wandb")
            self.enabled = False
            self.wandb = None

    def init(self, config: Optional[Dict[str, Any]] = None, name: Optional[str] = None, **kwargs):
        """
        Initialize W&B run.

        Args:
            config: Configuration dict
            name: Run name
            **kwargs: Additional W&B init arguments

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.enabled:
            return

        self.wandb.init(
            project=self.project, entity=self.entity, config=config, name=name, **kwargs
        )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step number

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.enabled:
            return

        if step is not None:
            metrics["step"] = step

        self.wandb.log(metrics)

    def log_model(self, model_path: str, name: Optional[str] = None):
        """
        Log model artifact.

        Args:
            model_path: Path to model file
            name: Model name

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.enabled:
            return

        self.wandb.save(model_path, base_path=name)

    def watch(self, model, log: str = "gradients", log_freq: int = 100):
        """
        Watch model gradients and parameters.

        Args:
            model: PyTorch model
            log: What to log ('gradients', 'parameters', or 'all')
            log_freq: Logging frequency

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.enabled:
            return

        self.wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        """Finish W&B run."""
        if not self.enabled:
            return

        self.wandb.finish()
