# corerec/csrc/__init__.py
try:
    from .tensor_ops import Tensor

    __all__ = ["Tensor"]
except ImportError as e:
    import warnings

    warnings.warn(
        f"Failed to import tensor_ops: {e}. Falling back to slower implementations.")

    # Define fallback Tensor implementation
    class Tensor:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "Fast Tensor implementation not available. Please compile the C++ extension."
            )

        def __repr__(self):
            return "Tensor(Not Implemented - C++ extension missing)"

    __all__ = ["Tensor"]
