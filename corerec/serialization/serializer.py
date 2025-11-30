"""
Core Serialization/Deserialization Engine

Handles conversion between objects and various formats (JSON, YAML, Pickle).
Uses reflection and dynamic imports for class reconstruction.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import Any, Dict, Union
import json
import pickle
import os
from pathlib import Path

from corerec.serialization.serializable import Serializable
from corerec.serialization.registry import SerializableRegistry


def serialize(obj: Any) -> Dict[str, Any]:
    """
    Serialize an object to dictionary.

    Args:
        obj: Object to serialize (must be Serializable or have to_dict method)

    Returns:
        Dictionary representation

    Example:
        model = NCF(params)
        model_dict = serialize(model)

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    if isinstance(obj, Serializable):
        return obj.to_dict()
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    else:
        # Handle basic types
        return obj


def deserialize(data: Union[Dict[str, Any], str, Path]) -> Any:
    """
    Deserialize an object from dictionary or file.

    Uses reflection and dynamic imports to reconstruct objects
    without hardcoded class imports.

    Args:
        data: Dictionary, file path, or JSON string

    Returns:
        Reconstructed object

    Example:
        model_dict = {'_type': 'NCF', '_module': 'corerec.engines.nn_base.ncf', ...}
        model = deserialize(model_dict)  # Automatically imports and creates NCF!

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    # Handle file paths
    if isinstance(data, (str, Path)):
        if os.path.exists(data):
            return load_from_file(data)
        else:
            # Try parsing as JSON string
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Invalid input: not a file path or JSON string")

    # Must be a dictionary at this point
    if not isinstance(data, dict):
        return data

    # Check if this is a serialized object
    if "_type" not in data or "_module" not in data:
        # Not a serialized object, return as-is
        return data

    class_name = data["_type"]
    module_name = data["_module"]

    # Get class from registry or via dynamic import
    klass = SerializableRegistry.get_by_module(class_name, module_name)

    if klass is None:
        raise ValueError(f"Cannot deserialize {class_name} from {module_name}")

    # Use from_dict if available, otherwise use constructor
    if hasattr(klass, "from_dict"):
        return klass.from_dict(data)
    else:
        # Remove metadata and construct
        constructor_args = {
            k: v for k,
            v in data.items() if not k.startswith("_")}
        return klass(**constructor_args)


def save_to_file(obj: Any, file_path: str, format: str = "json"):
    """
    Save object to file in specified format.

    Args:
        obj: Object to save
        file_path: Path to save to
        format: Format ('json', 'yaml', or 'pickle')

    Example:
        model = NCF(params)
        save_to_file(model, "model.json", format="json")

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(
        file_path) else ".", exist_ok=True)

    # Serialize object
    if format in ["json", "yaml"]:
        serialized = serialize(obj)

    # Write to file
    if format == "json":
        with open(file_path, "w") as f:
            json.dump(serialized, f, indent=2, default=str)
    elif format == "yaml":
        try:
            import yaml

            with open(file_path, "w") as f:
                yaml.dump(serialized, f, default_flow_style=False)
        except ImportError:
            raise ImportError(
                "PyYAML not installed. Install with: pip install pyyaml")
    elif format == "pickle":
        with open(file_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(
            f"Unsupported format: {format}. Use 'json', 'yaml', or 'pickle'")


def load_from_file(file_path: str) -> Any:
    """
    Load object from file.

    Automatically detects format based on file extension.

    Args:
        file_path: Path to load from

    Returns:
        Reconstructed object

    Example:
        model = load_from_file("model.json")
        # Automatically detects format and reconstructs NCF!

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)
        return deserialize(data)
    elif file_ext in [".yaml", ".yml"]:
        try:
            import yaml

            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
            return deserialize(data)
        except ImportError:
            raise ImportError(
                "PyYAML not installed. Install with: pip install pyyaml")
    elif file_ext in [".pkl", ".pickle"]:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. Use .json, .yaml, or .pkl")
