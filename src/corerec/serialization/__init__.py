"""
CoreRec Serialization Framework

LangChain-style serialization/deserialization for CoreRec models.
Enables dynamic import, registry pattern, and full config preservation.

Features:
    - Serialize complex models to JSON/YAML/Pickle
    - Deserialize models dynamically without hardcoded imports
    - Registry pattern for class management
    - Full state preservation including nested objects
    
Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from corerec.serialization.serializable import Serializable
from corerec.serialization.registry import SerializableRegistry, register_serializable
from corerec.serialization.serializer import serialize, deserialize, save_to_file, load_from_file

__all__ = [
    "Serializable",
    "SerializableRegistry",
    "register_serializable",
    "serialize",
    "deserialize",
    "save_to_file",
    "load_from_file",
]
