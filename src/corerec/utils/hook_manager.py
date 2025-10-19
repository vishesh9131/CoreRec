
"""
This file is used to manage hooks for the model.

Author: Vishesh Yadav
Date: 2025-03-30
"""

class HookManager:
    """Manager for model hooks to inspect internal states."""
    
    def __init__(self):
        """Initialize the hook manager."""
        self.hooks = {}
        self.activations = {}
    
    def _get_activation(self, name):
        """Get activation for a specific layer."""
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def register_hook(self, model, layer_name, callback=None):
        """Register a hook for a specific layer."""
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            if callback is None:
                callback = self._get_activation(layer_name)
            handle = layer.register_forward_hook(callback)
            self.hooks[layer_name] = handle
            return True
        
        # Try to find the layer in submodules
        for name, module in model.named_modules():
            if name == layer_name:
                if callback is None:
                    callback = self._get_activation(layer_name)
                handle = module.register_forward_hook(callback)
                self.hooks[layer_name] = handle
                return True
        
        return False
    
    def remove_hook(self, layer_name):
        """Remove a hook for a specific layer."""
        if layer_name in self.hooks:
            self.hooks[layer_name].remove()
            del self.hooks[layer_name]
            return True
        return False
    
    def get_activation(self, layer_name):
        """Get the activation for a specific layer."""
        return self.activations.get(layer_name, None)
    
    def clear_activations(self):
        """Clear all stored activations."""
        self.activations.clear()

