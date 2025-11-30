"""
Base Serializable Class

Provides foundation for all serializable objects in CoreRec.
Similar to LangChain's Serializable base class.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from abc import ABC
from typing import Dict, Any, Optional, List
import inspect


class Serializable(ABC):
    """
    Base class for all serializable objects in CoreRec.

    Any class inheriting from this can be automatically serialized/deserialized
    with full config preservation and nested object support.

    Features:
        - Automatic __init__ parameter extraction
        - Nested Serializable object handling
        - Dynamic reconstruction via reflection
        - JSON/YAML compatible output

    Example:
        @register_serializable("my_model")
        class MyModel(Serializable):
            def __init__(self, param1, param2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        model = MyModel(10, 20)
        model_dict = model.to_dict()
        # {'_type': 'MyModel', '_module': '__main__', 'param1': 10, 'param2': 20}

        loaded = MyModel.from_dict(model_dict)
        assert loaded.param1 == 10

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self):
        """Initialize serializable object."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert object to dictionary representation.

        Automatically extracts all __init__ parameters and their current values.
        Handles nested Serializable objects recursively.

        Returns:
            Dictionary containing serializable representation with metadata

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Get class metadata
        class_name = self.__class__.__name__
        module_name = self.__class__.__module__

        # Extract __init__ parameters
        init_signature = inspect.signature(self.__class__.__init__)
        params = {}

        for param_name in init_signature.parameters:
            if param_name == "self":
                continue

            # Get current value of this parameter
            if hasattr(self, param_name):
                value = getattr(self, param_name)

                # Handle nested Serializable objects
                if isinstance(value, Serializable):
                    params[param_name] = value.to_dict()
                # Handle lists of Serializable objects
                elif (
                    isinstance(value, list)
                    and value
                    and all(isinstance(v, Serializable) for v in value)
                ):
                    params[param_name] = [v.to_dict() for v in value]
                # Handle dicts with Serializable values
                elif isinstance(value, dict):
                    params[param_name] = {
                        k: v.to_dict() if isinstance(v, Serializable) else v
                        for k, v in value.items()
                    }
                # Regular values (primitives, lists, dicts)
                else:
                    params[param_name] = value

        # Create serialized dict with metadata
        serialized = {"_type": class_name, "_module": module_name, **params}

        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """
        Reconstruct object from dictionary.

        Handles dynamic imports and nested object reconstruction automatically.

        Args:
            data: Dictionary containing serialized object with metadata

        Returns:
            Reconstructed object instance

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Remove metadata
        data = data.copy()
        data.pop("_type", None)
        data.pop("_module", None)

        # Reconstruct nested Serializable objects
        for key, value in data.items():
            if isinstance(value, dict) and "_type" in value and "_module" in value:
                # Dynamically import and reconstruct nested object
                from corerec.serialization.serializer import deserialize

                data[key] = deserialize(value)
            elif isinstance(value, list) and value:
                # Handle lists that might contain Serializable objects
                data[key] = [
                    deserialize(v) if isinstance(v, dict) and "_type" in v else v for v in value
                ]

        # Create instance with reconstructed parameters
        return cls(**data)

    def save(self, file_path: str, format: str = "json"):
        """
        Save object to file.

        Args:
            file_path: Path to save file
            format: Format to use ('json', 'yaml', or 'pickle')

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        from corerec.serialization.serializer import save_to_file

        save_to_file(self, file_path, format=format)

    @classmethod
    def load(cls, file_path: str) -> "Serializable":
        """
        Load object from file.

        Args:
            file_path: Path to load from

        Returns:
            Reconstructed object

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        from corerec.serialization.serializer import load_from_file

        return load_from_file(file_path)
