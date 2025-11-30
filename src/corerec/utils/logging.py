"""
Logging utilities for CoreRec framework.

This module provides utilities for setting up logging.
"""

import os
import logging
import sys
from typing import Optional, Dict, Any


def setup_logging(
    log_file: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_file (Optional[str]): Path to log file. If None, no file logging is set up.
        console_level (int): Logging level for console handler.
        file_level (int): Logging level for file handler.
        log_format (Optional[str]): Log message format. If None, a default format is used.

    Returns:
        logging.Logger: Root logger
    """
    # Create default log format if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_file is provided
    if log_file is not None:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name (str): Logger name
        log_file (Optional[str]): Path to log file. If None, no file logging is set up.
        level (int): Logging level

    Returns:
        logging.Logger: Logger
    """
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add file handler if log_file is provided and logger doesn't already have handlers
    if log_file is not None and not logger.handlers:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding contextual information to log messages.

    This adapter can be used to add additional information to log messages,
    such as run ID, user ID, etc.
    """

    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        """Initialize the logger adapter.

        Args:
            logger (logging.Logger): Logger to adapt
            extra (Dict[str, Any]): Extra contextual information to add to log messages
        """
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process the log message.

        Args:
            msg (str): Log message
            kwargs (Dict[str, Any]): Keyword arguments for the log message

        Returns:
            tuple: Processed log message and keyword arguments
        """
        # Add extra contextual information to log message
        extra_info = " ".join(f"{k}={v}" for k, v in self.extra.items())
        if extra_info:
            msg = f"{msg} [{extra_info}]"

        return msg, kwargs
