"""
Serializable Class Registry

Global registry for all serializable classes in CoreRec.
Enables dynamic class loading and reconstruction via reflection.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import Dict, Type, Optional, Any
import importlib


class SerializableRegistry:
    """
    Global registry for serializable classes.

    Maintains mapping of class names to class objects, enabling
    dynamic imports and object reconstruction at runtime.

    This uses Python's reflection capabilities (importlib) to load
    classes dynamically without hardcoded imports.

    Example:
        # Register a class
        SerializableRegistry.register("NCF", NCF, "corerec.engines.nn_base.ncf")

        # Get class dynamically
        klass = SerializableRegistry.get("NCF")
        model = klass(params...)

        # Or by module path
        klass = SerializableRegistry.get_by_module("NCF", "corerec.engines.nn_base.ncf")

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    _registry: Dict[str, Type] = {}
    _module_registry: Dict[str, str] = {}  # class_name -> module_path

    @classmethod
    def register(cls, name: str, klass: Type, module: Optional[str] = None):
        """
        Register a class in the registry.

        Args:
            name: Unique name for the class
            klass: The class object to register
            module: Optional module path for dynamic import

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        cls._registry[name] = klass
        if module is not None:
            cls._module_registry[name] = module

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """
        Get a class from the registry by name.

        Args:
            name: Name of the class

        Returns:
            Class object or None if not found

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return cls._registry.get(name)

    @classmethod
    def get_by_module(cls, class_name: str, module_name: str) -> Optional[Type]:
        """
        Get a class by dynamically importing it at runtime.

        This uses Python's reflection capabilities (importlib) to import
        a class without hardcoded imports - key feature for serialization!

        Args:
            class_name: Name of the class (e.g., 'NCF')
            module_name: Module path (e.g., 'corerec.engines.nn_base.ncf')

        Returns:
            Class object or None if import fails

        Example:
            NCF = SerializableRegistry.get_by_module("NCF", "corerec.engines.nn_base.ncf")
            model = NCF(params...)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # First check if already in registry
        if class_name in cls._registry:
            return cls._registry[class_name]

        # Try dynamic import using reflection
        try:
            module = importlib.import_module(module_name)
            klass = getattr(module, class_name)

            # Cache for future use
            cls.register(class_name, klass, module_name)

            return klass
        except (ImportError, AttributeError) as e:
            print(f"Failed to import {class_name} from {module_name}: {e}")
            return None

    @classmethod
    def list_registered(cls) -> Dict[str, Type]:
        """
        List all registered classes.

        Returns:
            Dict mapping class names to class objects

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return cls._registry.copy()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a class is registered.

        Args:
            name: Class name to check

        Returns:
            True if registered, False otherwise

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return name in cls._registry

    @classmethod
    def clear_registry(cls):
        """
        Clear the registry (useful for testing).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        cls._registry.clear()
        cls._module_registry.clear()


def register_serializable(name: Optional[str] = None):
    """
    Decorator to register a class as serializable.

    This is the easiest way to make a class serializable - just add the decorator!

    Usage:
        @register_serializable("my_recommender")
        class MyRecommender(Serializable):
            def __init__(self, param1, param2):
                self.param1 = param1
                self.param2 = param2

        # Now it's automatically registered and can be serialized/deserialized
        model = MyRecommender(10, 20)
        model_dict = serialize(model)
        loaded = deserialize(model_dict)  # Automatically reconstructs!

    Args:
        name: Optional name for registration (uses class name if not provided)

    Returns:
        Decorated class

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def decorator(klass):
        reg_name = name if name is not None else klass.__name__
        module_name = klass.__module__
        SerializableRegistry.register(reg_name, klass, module_name)
        return klass

    return decorator
