"""
Diagnostic utilities for CoreRec.

Provides tools to check system configuration, dependencies, and common issues.
"""

import sys
import warnings
from typing import Dict, Any, Optional


def check_numpy_backend() -> Dict[str, Any]:
    """
    Check which BLAS backend NumPy is using.
    
    Returns:
        Dict with 'backend' (str), 'has_mkl' (bool), 'warning' (str or None)
    """
    try:
        import numpy as np
        import io
        from contextlib import redirect_stdout
        
        # Get NumPy config
        f = io.StringIO()
        with redirect_stdout(f):
            np.show_config()
        config_str = f.getvalue()
        config_lower = config_str.lower()
        
        has_mkl = 'mkl' in config_lower
        has_openblas = 'openblas' in config_lower or ('blas' in config_lower and 'mkl' not in config_lower)
        
        if has_mkl:
            backend = 'Intel MKL'
            warning = (
                "NumPy is using Intel MKL, which may show SSE4.2 deprecation warnings.\n"
                "To fix: Reinstall NumPy/SciPy with OpenBLAS:\n"
                "  conda: conda install -c conda-forge numpy scipy scikit-learn\n"
                "  pip:   pip uninstall numpy scipy scikit-learn -y && pip install numpy scipy scikit-learn"
            )
        elif has_openblas:
            backend = 'OpenBLAS'
            warning = None
        else:
            backend = 'Unknown'
            warning = None
            
        return {
            'backend': backend,
            'has_mkl': has_mkl,
            'numpy_version': np.__version__,
            'warning': warning,
            'config': config_str
        }
    except ImportError:
        return {
            'backend': 'Not installed',
            'has_mkl': False,
            'numpy_version': None,
            'warning': 'NumPy is not installed',
            'config': None
        }


def check_mkl_warning() -> Optional[str]:
    """
    Check if Intel MKL warnings are likely to occur.
    
    Returns:
        Warning message if MKL is detected, None otherwise
    """
    result = check_numpy_backend()
    return result.get('warning')


def print_system_info():
    """Print comprehensive system and dependency information."""
    print("=" * 60)
    print("CoreRec System Diagnostics")
    print("=" * 60)
    
    # Python version
    print(f"\nPython: {sys.version}")
    
    # NumPy backend
    numpy_info = check_numpy_backend()
    print(f"\nNumPy Version: {numpy_info.get('numpy_version', 'Not installed')}")
    print(f"BLAS Backend: {numpy_info.get('backend', 'Unknown')}")
    
    if numpy_info.get('has_mkl'):
        print("\n⚠️  WARNING: Intel MKL detected")
        print(numpy_info.get('warning', ''))
    else:
        print("\n✓ NumPy backend looks good")
    
    # Check other key dependencies
    print("\n" + "-" * 60)
    print("Key Dependencies:")
    print("-" * 60)
    
    deps = ['numpy', 'scipy', 'pandas', 'torch', 'sklearn']
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  {dep:12s}: {version}")
        except ImportError:
            print(f"  {dep:12s}: Not installed")
    
    print("\n" + "=" * 60)


def fix_mkl_warning_instructions():
    """
    Print instructions to fix Intel MKL warnings.
    
    This is a helper function that provides clear instructions
    for users experiencing MKL warnings.
    """
    print("\n" + "=" * 60)
    print("How to Fix Intel MKL SSE4.2 Warning")
    print("=" * 60)
    print("""
The warning occurs because NumPy/SciPy are using Intel MKL, which
requires AVX instructions that your CPU may not support.

SOLUTION: Reinstall NumPy/SciPy with OpenBLAS instead of MKL

Option 1: Using Conda (Recommended)
-----------------------------------
    conda install -c conda-forge numpy scipy scikit-learn

Option 2: Using pip
-------------------
    pip uninstall numpy scipy scikit-learn -y
    pip install numpy scipy scikit-learn

Option 3: Force reinstall from source (if above doesn't work)
--------------------------------------------------------------
    pip uninstall numpy scipy scikit-learn -y
    pip install --no-binary numpy,scipy numpy scipy scikit-learn

After reinstalling, verify the fix:
    python -c "import numpy; numpy.show_config()"
    
Look for 'openblas' or 'blas' instead of 'mkl' in the output.
    """)
    print("=" * 60 + "\n")

