"""
Configuration Manager

Unified configuration management for CoreRec models.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import Dict, Any, Optional, Union
import yaml
import json
from pathlib import Path
import os


class ModelConfig:
    """
    Model configuration container.

    Provides dot notation access to config values and validation.

    Example:
        config = ModelConfig({'embedding_dim': 64, 'num_layers': 3})
        print(config.embedding_dim)  # 64
        print(config.get('num_layers', 2))  # 3

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize with config dictionary."""
        self._config = config_dict or {}

    def __getattr__(self, name: str) -> Any:
        """Get config value via dot notation."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return self._config.get(name)

    def __setattr__(self, name: str, value: Any):
        """Set config value via dot notation."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._config[name] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self._config.get(key, default)

    def update(self, other: Dict[str, Any]):
        """Update config with another dict."""
        self._config.update(other)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()


class ConfigManager:
    """
    Unified configuration management system.

    Handles loading, validation, and merging of configurations
    from multiple sources (files, environment variables, defaults).

    Example:
        config = ConfigManager()
        config.load_from_file('config.yaml')
        config.load_from_env(prefix='COREREC_')
        config.set('model.embedding_dim', 128)

        model_config = config.get_section('model')

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to config file to load

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.config: Dict[str, Any] = {}

        if config_path:
            self.load_from_file(config_path)

    def load_from_file(self, path: Union[str, Path]) -> "ConfigManager":
        """
        Load configuration from YAML or JSON file.

        Args:
            path: Path to configuration file

        Returns:
            self for chaining

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            if path.suffix in [".yaml", ".yml"]:
                loaded_config = yaml.safe_load(f)
            elif path.suffix == ".json":
                loaded_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

        self.config.update(loaded_config)
        return self

    def load_from_env(self, prefix: str = "COREREC_") -> "ConfigManager":
        """
        Load configuration from environment variables.

        Args:
            prefix: Prefix for environment variables (e.g., COREREC_MODEL_DIM)

        Returns:
            self for chaining

        Example:
            # export COREREC_MODEL_EMBEDDING_DIM=128
            config.load_from_env('COREREC_')
            # config will have {'model': {'embedding_dim': 128}}

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(prefix):].lower().replace("_", ".")
                self.set(config_key, value)

        return self

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Supports nested keys with dot notation.

        Args:
            key: Configuration key (supports dot notation, e.g., 'model.embedding_dim')
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            dim = config.get('model.embedding_dim', 64)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Supports nested keys with dot notation.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set

        Example:
            config.set('model.embedding_dim', 128)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        keys = key.split(".")
        target = self.config

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

    def get_section(self, section: str) -> ModelConfig:
        """
        Get a configuration section as ModelConfig object.

        Args:
            section: Section name (e.g., 'model', 'training')

        Returns:
            ModelConfig object for the section

        Example:
            model_config = config.get_section('model')
            print(model_config.embedding_dim)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        section_dict = self.config.get(section, {})
        return ModelConfig(section_dict)

    def merge(self, other: Union[Dict, "ConfigManager"]) -> "ConfigManager":
        """
        Merge another configuration.

        Args:
            other: Dictionary or ConfigManager to merge

        Returns:
            self for chaining

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if isinstance(other, ConfigManager):
            other = other.config

        self._deep_merge(self.config, other)
        return self

    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(
                    base[key],
                    dict) and isinstance(
                    value,
                    dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def save(self, path: Union[str, Path], format: str = "yaml"):
        """
        Save configuration to file.

        Args:
            path: Path to save to
            format: Format ('yaml' or 'json')

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        path = Path(path)

        with open(path, "w") as f:
            if format == "yaml":
                yaml.dump(self.config, f, default_flow_style=False)
            elif format == "json":
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.config.copy()
