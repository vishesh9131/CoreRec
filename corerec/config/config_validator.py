"""
Configuration Validator

Validates configurations against schemas to prevent errors.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import Dict, Any, List, Optional


class ConfigValidator:
    """
    Configuration validation utility.

    Validates config dicts against expected schemas to catch errors early.

    Example:
        validator = ConfigValidator()
        validator.add_rule('embedding_dim', int, required=True)
        validator.add_rule('dropout', float, min_value=0.0, max_value=1.0)

        is_valid = validator.validate(config)

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self):
        """Initialize validator."""
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.errors: List[str] = []

    def add_rule(
        self,
        key: str,
        expected_type: type,
        required: bool = False,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        allowed_values: Optional[List] = None,
    ):
        """
        Add validation rule for a configuration key.

        Args:
            key: Configuration key
            expected_type: Expected Python type
            required: Whether key is required
            min_value: Minimum value (for numbers)
            max_value: Maximum value (for numbers)
            allowed_values: List of allowed values

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.rules[key] = {
            "type": expected_type,
            "required": required,
            "min": min_value,
            "max": max_value,
            "allowed": allowed_values,
        }

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration against rules.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if valid, False otherwise

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.errors = []

        for key, rule in self.rules.items():
            # Check if required key exists
            if rule["required"] and key not in config:
                self.errors.append(f"Required key '{key}' is missing")
                continue

            if key not in config:
                continue

            value = config[key]

            # Check type
            if not isinstance(value, rule["type"]):
                self.errors.append(
                    f"Key '{key}' has wrong type: expected {
                        rule['type'].__name__}, got {
                        type(value).__name__}")
                continue

            # Check range for numbers
            if rule["min"] is not None and value < rule["min"]:
                self.errors.append(
                    f"Key '{key}' value {value} is below minimum {
                        rule['min']}")

            if rule["max"] is not None and value > rule["max"]:
                self.errors.append(
                    f"Key '{key}' value {value} is above maximum {
                        rule['max']}")

            # Check allowed values
            if rule["allowed"] is not None and value not in rule["allowed"]:
                self.errors.append(
                    f"Key '{key}' value {value} not in allowed values {
                        rule['allowed']}")

        return len(self.errors) == 0

    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self.errors
