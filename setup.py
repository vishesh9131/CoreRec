from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import warnings


# Read the version from corerec/__init__.py
def get_version():
    init_file = os.path.join("corerec", "__init__.py")
    with open(init_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.5.1"


# Read long description from README
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()


# Custom build_ext to make C++ extensions optional
class OptionalBuildExt(build_ext):
    """
    Custom build_ext that doesn't fail if C++ extensions can't be compiled.
    This allows the package to install without C++ extensions, falling back to
    pure Python implementations.
    """

    def run(self):
        try:
            super().run()
        except Exception as e:
            warnings.warn(
                f"Failed to build C++ extensions: {e}\n"
                f"CoreRec will install without optimized C++ operations. "
                f"Performance may be reduced but all features will work."
            )

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            warnings.warn(
                f"Failed to build extension {ext.name}: {e}\n"
                f"Continuing without this extension..."
            )


# Define C++ extensions (optional)
def get_extensions():
    """
    Define C++ extensions for accelerated operations.
    These are optional - if they fail to build, the package will still install.
    """
    extensions = []

    # Check if csrc directory exists and has source files
    csrc_dir = os.path.join("corerec", "csrc")
    if os.path.exists(csrc_dir):
        # Only add extensions if we can find source files
        # This is a placeholder - customize based on actual C++ sources
        cpp_files = [
            os.path.join(csrc_dir, f)
            for f in os.listdir(csrc_dir)
            if f.endswith(".cpp") or f.endswith(".cc")
        ]

        if cpp_files:
            extensions.append(
                Extension(
                    "corerec.csrc.tensor_ops",
                    sources=cpp_files,
                    include_dirs=[csrc_dir],
                    extra_compile_args=["-std=c++14", "-O3"],
                    optional=True,
                )
            )

    return extensions


setup(
    name="corerec",
    version=get_version(),
    author="Vishesh Yadav",
    author_email="vishesh@corerec.tech",
    description="Advanced Recommendation Systems Library with State-of-the-Art Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vishesh9131/CoreRec",
    packages=find_packages(exclude=["tests*", "examples*", "docs*", "src*"]),
    # Optional C++ extensions for performance
    ext_modules=get_extensions(),
    cmdclass={"build_ext": OptionalBuildExt},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0,<3.0.0",
        "pandas>=1.3.0,<3.0.0",
        "numpy>=1.21.0,<2.0.0",
        "scikit-learn>=0.24.0,<2.0.0",
        "scipy>=1.7.0,<2.0.0",
        "tqdm>=4.62.0,<5.0.0",
        "argcomplete>=3.0.0,<4.0.0",
        "networkx>=2.5.0,<4.0.0",
        "matplotlib>=3.3.0,<4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "isort>=5.10",
            "pre-commit>=2.15",
        ],
        "docs": [
            "mkdocs>=1.4",
            "mkdocs-material>=9.0",
            "mkdocstrings[python]>=0.19",
        ],
        "transformers": [
            "transformers>=4.0",
            "gensim>=4.0",
        ],
        "all": [
            "transformers>=4.0",
            "gensim>=4.0",
            "pytest>=7.0",
            "pytest-cov>=3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "corerec=corerec.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

# Note: C++ extensions in corerec/csrc/ are optional.
# If compilation fails, CoreRec will install without them and use pure Python fallbacks.
# For CUDA support, ensure CUDA toolkit is installed before running pip
# install.
