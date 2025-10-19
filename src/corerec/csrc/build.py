# corerec/csrc/build.py
import os
import sys
import subprocess
import numpy as np

def build_extension():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source files
    sources = [
        os.path.join(script_dir, 'tensor/tensor.cpp'),
        os.path.join(script_dir, 'ops/embedding_ops.cpp'),
        os.path.join(script_dir, 'python/module.cpp')
    ]
    
    # Output file
    output = os.path.join(script_dir, 'tensor_ops.so')
    
    # Build command
    cmd = ['c++', '-shared', '-std=c++14', '-O3', '-fPIC']
    
    # Add include directories
    cmd.extend(['-I', np.get_include()])
    cmd.extend(['-I', os.path.join(sys.prefix, 'include')])
    cmd.extend(['-I', os.path.join(sys.prefix, 'include/python' + sys.version[:3])])
    
    # Add source files
    cmd.extend(sources)
    
    # Add output file
    cmd.extend(['-o', output])
    
    # Add platform-specific flags
    if sys.platform == 'darwin':  # macOS
        cmd.extend(['-undefined', 'dynamic_lookup'])
        # For arm64 architecture
        if 'arm' in os.uname().machine:
            cmd.extend(['-arch', 'arm64'])
    
    # Execute the build command
    print("Executing:", ' '.join(cmd))
    subprocess.check_call(cmd)
    print(f"Successfully built extension: {output}")

if __name__ == "__main__":
    build_extension()