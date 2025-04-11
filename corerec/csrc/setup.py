# corerec/csrc/setup.py
from setuptools import setup, Extension, find_packages
import numpy as np
import os

# Define the extension module
fast_embedding_ops = Extension(
    'fast_embedding_ops',
    sources=[
        'tensor/tensor.cpp',
        'ops/embedding_ops.cpp',
        'python/module.cpp'
    ],
    include_dirs=[np.get_include()],
    extra_compile_args=['-std=c++14', '-O3'],
    language='c++'
)

# Setup the package
setup(
    name='corerec-csrc',
    version='0.1',
    description='Fast C++ operations for CoreRec',
    packages=find_packages(),
    ext_modules=[fast_embedding_ops],
    package_dir={'': '.'}
)