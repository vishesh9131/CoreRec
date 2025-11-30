"""
MLflow Integration

Experiment tracking and model registry with MLflow.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import Dict, Any, Optional
import logging


class MLflowTracker:
    """
    MLflow integration for experiment tracking.

    Tracks parameters, metrics, and models with MLflow.

    Example:
        from corerec.integrations import MLflowTracker

        tracker = MLflowTracker(experiment_name="ncf_experiments")

        with tracker.start_run("run_1"):
            tracker.log_params({'embedding_dim': 64, 'num_layers': 3})

            # Training...
            tracker.log_metrics({'train_loss': 0.5, 'val_loss': 0.6}, step=1)

            tracker.log_model(model, "ncf_model")

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking server URI (optional)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.experiment_name = experiment_name
        self.logger = logging.getLogger("MLflowTracker")

        try:
            import mlflow

            self.mlflow = mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            mlflow.set_experiment(experiment_name)
            self.enabled = True

        except ImportError:
            self.logger.warning("MLflow not installed. Install with: pip install mlflow")
            self.enabled = False
            self.mlflow = None

    def start_run(self, run_name: Optional[str] = None):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run

        Returns:
            MLflow run context manager

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.enabled:
            # Return a no-op context manager
            from contextlib import nullcontext

            return nullcontext()

        return self.mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.

        Args:
            params: Dictionary of parameters

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.enabled:
            return

        self.mlflow.log_params(params)

    def log_param(self, key: str, value: Any):
        """Log single parameter."""
        if not self.enabled:
            return

        self.mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step number (epoch, iteration, etc.)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.enabled:
            return

        self.mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log single metric."""
        if not self.enabled:
            return

        self.mlflow.log_metric(key, value, step=step)

    def log_model(self, model, artifact_path: str = "model"):
        """
        Log model as artifact.

        Args:
            model: Model to log
            artifact_path: Path within artifact store

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.enabled:
            return

        try:
            self.mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            self.logger.warning(f"Could not log model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log file as artifact.

        Args:
            local_path: Local file path
            artifact_path: Path within artifact store

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.enabled:
            return

        self.mlflow.log_artifact(local_path, artifact_path)

    def set_tag(self, key: str, value: Any):
        """Set a tag for the run."""
        if not self.enabled:
            return

        self.mlflow.set_tag(key, value)
