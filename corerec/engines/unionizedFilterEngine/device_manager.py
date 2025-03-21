import os
import numpy as np
import warnings
from typing import Optional, Union, List, Dict

class DeviceManager:
    """
    Manages computation devices for recommendation algorithms.
    
    Supports various hardware acceleration platforms:
    - CPU
    - CUDA (NVIDIA GPUs)
    - MPS (Apple Metal Performance Shaders)
    - ROCm (AMD GPUs)
    - Metal (Apple GPUs)
    - DirectML (Windows ML)
    - OpenCL
    - SYCL
    - Vulkan Compute
    - Intel GPU
    - TPU (Tensor Processing Units)
    - IPU (Intelligence Processing Units)
    - FPGA
    - RISC-V GPU
    """
    
    SUPPORTED_DEVICES = [
        'cpu', 'cuda', 'mps', 'rocm', 'metal', 'directml', 'opencl', 
        'sycl', 'vulkan', 'level0', 'intel_gpu', 'xla', 'tensorrt', 
        'spirv', 'webgpu', 'plaidml', 'tpu', 'ipu', 'habana', 'fpga', 'riscv_gpu'
    ]
    
    def __init__(self, preferred_device: str = 'auto'):
        """
        Initialize the device manager.
        
        Args:
            preferred_device: Device to use for computation. If 'auto', will select the best available device.
        """
        self.preferred_device = preferred_device
        self._active_device = None
        self._torch_available = self._check_torch_available()
        self._tf_available = self._check_tensorflow_available()
        self._jax_available = self._check_jax_available()
        
        # Initialize device
        self.set_device(preferred_device)
    
    def _check_torch_available(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def _check_tensorflow_available(self) -> bool:
        """Check if TensorFlow is available."""
        try:
            import tensorflow as tf
            return True
        except ImportError:
            return False
    
    def _check_jax_available(self) -> bool:
        """Check if JAX is available."""
        try:
            import jax
            return True
        except ImportError:
            return False
    
    def get_available_devices(self) -> List[str]:
        """Get list of available computation devices."""
        available_devices = ['cpu']
        
        # Check PyTorch devices
        if self._torch_available:
            import torch
            if torch.cuda.is_available():
                available_devices.append('cuda')
            
            # Check for MPS (Apple Silicon)
            try:
                if torch.backends.mps.is_available():
                    available_devices.append('mps')
            except AttributeError:
                pass
        
        # Check for ROCm (AMD GPUs)
        if self._torch_available and os.environ.get('ROCM_HOME'):
            try:
                import torch
                if torch.cuda.is_available() and 'rocm' in torch.version.hip:
                    available_devices.append('rocm')
            except (AttributeError, ImportError):
                pass
        
        # Check for TPU
        if self._tf_available:
            import tensorflow as tf
            try:
                tpu_devices = tf.config.list_logical_devices('TPU')
                if tpu_devices:
                    available_devices.append('tpu')
            except:
                pass
        
        # Add other device checks as needed
        
        return available_devices
    
    def set_device(self, device: str = 'auto') -> str:
        """
        Set the active computation device.
        
        Args:
            device: Device to use. If 'auto', will select the best available device.
            
        Returns:
            The active device name
        """
        if device == 'auto':
            available_devices = self.get_available_devices()
            # Prioritize devices in order of typical performance
            for preferred in ['cuda', 'mps', 'rocm', 'tpu', 'cpu']:
                if preferred in available_devices:
                    device = preferred
                    break
            else:
                device = 'cpu'  # Default fallback
        
        if device not in self.SUPPORTED_DEVICES:
            warnings.warn(f"Device '{device}' not in supported devices list. Falling back to CPU.")
            device = 'cpu'
        
        # Check if requested device is available
        available_devices = self.get_available_devices()
        if device not in available_devices:
            warnings.warn(f"Requested device '{device}' is not available. Falling back to CPU.")
            device = 'cpu'
        
        self._active_device = device
        return self._active_device
    
    @property
    def active_device(self) -> str:
        """Get the currently active device."""
        return self._active_device
    
    def to_device(self, data, device: Optional[str] = None):
        """
        Move data to the specified device.
        
        Args:
            data: Data to move (numpy array, torch tensor, etc.)
            device: Target device. If None, uses the active device.
            
        Returns:
            Data on the target device
        """
        if device is None:
            device = self._active_device
        
        # Handle NumPy arrays
        if isinstance(data, np.ndarray):
            if device == 'cpu':
                return data
            
            # Convert to appropriate framework tensor
            if self._torch_available and device in ['cuda', 'mps', 'rocm']:
                import torch
                return torch.from_numpy(data).to(device)
            
            if self._tf_available and device in ['tpu', 'cuda']:
                import tensorflow as tf
                return tf.convert_to_tensor(data)
            
            # If no conversion is possible, return original array with warning
            warnings.warn(f"Cannot move NumPy array to device '{device}'. Keeping on CPU.")
            return data
        
        # Handle PyTorch tensors
        if self._torch_available:
            import torch
            if isinstance(data, torch.Tensor):
                return data.to(device)
        
        # Handle TensorFlow tensors
        if self._tf_available:
            import tensorflow as tf
            if isinstance(data, tf.Tensor):
                # TensorFlow handles device placement differently
                with tf.device(device):
                    return tf.identity(data)
        
        # If data type is not recognized, return as is
        return data
    
    def create_tensor(self, data, device: Optional[str] = None, dtype=None):
        """
        Create a tensor on the specified device.
        
        Args:
            data: Data to convert to tensor
            device: Target device. If None, uses the active device.
            dtype: Data type for the tensor
            
        Returns:
            Tensor on the target device
        """
        if device is None:
            device = self._active_device
        
        if self._torch_available and device in ['cpu', 'cuda', 'mps', 'rocm']:
            import torch
            torch_dtype = None
            if dtype is not None:
                # Map numpy/python dtypes to torch dtypes
                dtype_map = {
                    'float32': torch.float32,
                    'float64': torch.float64,
                    'int32': torch.int32,
                    'int64': torch.int64,
                    float: torch.float32,
                    int: torch.int64,
                }
                torch_dtype = dtype_map.get(dtype, dtype)
            
            return torch.tensor(data, device=device, dtype=torch_dtype)
        
        if self._tf_available and device in ['cpu', 'tpu', 'cuda']:
            import tensorflow as tf
            with tf.device(device):
                return tf.convert_to_tensor(data, dtype=dtype)
        
        # Default to NumPy array
        return np.array(data, dtype=dtype)
    
    def get_framework_for_device(self, device: Optional[str] = None):
        """
        Get the appropriate computation framework for the device.
        
        Args:
            device: Target device. If None, uses the active device.
            
        Returns:
            Module reference to appropriate framework (torch, tf, np, etc.)
        """
        if device is None:
            device = self._active_device
        
        if device in ['cuda', 'mps', 'rocm'] and self._torch_available:
            import torch
            return torch
        
        if device in ['tpu'] and self._tf_available:
            import tensorflow as tf
            return tf
        
        # Default to NumPy for CPU and unsupported devices
        return np