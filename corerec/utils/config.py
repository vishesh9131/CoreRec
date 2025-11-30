"""
Configuration utilities for CoreRec framework.

This module provides utilities for loading and parsing configuration files.
"""

import os
import yaml
import json
import argparse
from typing import Dict, Any, List, Union, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a file.

    Supports YAML and JSON formats.

    Args:
        config_path (str): Path to configuration file

    Returns:
        Dict[str, Any]: Configuration dictionary

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the configuration file format is not supported
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Get file extension
    _, ext = os.path.splitext(config_path)
    ext = ext.lower()

    # Load configuration
    if ext in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif ext == ".json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {ext}")

    return config


def merge_configs(
        base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.

    The override_config will override values in base_config.

    Args:
        base_config (Dict[str, Any]): Base configuration dictionary
        override_config (Dict[str, Any]): Override configuration dictionary

    Returns:
        Dict[str, Any]: Merged configuration dictionary
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(
                merged[key],
                dict) and isinstance(
                value,
                dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def load_configs(config_paths: List[str]) -> Dict[str, Any]:
    """Load and merge multiple configuration files.

    Args:
        config_paths (List[str]): List of paths to configuration files

    Returns:
        Dict[str, Any]: Merged configuration dictionary
    """
    if not config_paths:
        return {}

    # Load first configuration
    config = load_config(config_paths[0])

    # Merge with other configurations
    for path in config_paths[1:]:
        override_config = load_config(path)
        config = merge_configs(config, override_config)

    return config


def parse_cli_args() -> Dict[str, Any]:
    """Parse command line arguments into a configuration dictionary.

    Returns:
        Dict[str, Any]: Configuration dictionary from command line arguments
    """
    parser = argparse.ArgumentParser(description="CoreRec Configuration")

    # Add common arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--model_dir", type=str, help="Model directory")
    parser.add_argument("--log_dir", type=str, help="Log directory")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Parse arguments
    args = parser.parse_args()

    # Convert to dictionary
    config = vars(args)

    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}

    return config


def load_config_with_cli_args() -> Dict[str, Any]:
    """Load configuration from file and command line arguments.

    Command line arguments override values from configuration file.

    Returns:
        Dict[str, Any]: Merged configuration dictionary
    """
    # Parse command line arguments
    cli_config = parse_cli_args()

    # Load configuration from file if provided
    file_config = {}
    if "config" in cli_config:
        file_config = load_config(cli_config["config"])

    # Merge configurations
    config = merge_configs(file_config, cli_config)

    return config
