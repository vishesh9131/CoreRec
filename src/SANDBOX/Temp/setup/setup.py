from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import shutil

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        # Define the destination directory on the user's Mac
        destination_directory = os.path.expanduser("~/your_library_files/")
        os.makedirs(destination_directory, exist_ok=True)
        
        # Define the source files to copy
        source_files = [
            'scripts/your_script.sh',
            # Add more files if needed
        ]
        
        # Copy each file to the destination directory
        for file in source_files:
            shutil.copy(file, destination_directory)
        
        print(f"Copied {source_files} to {destination_directory}")

setup(
    name='corerec',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your library dependencies here
        'matplotlib==3.7.5',
        'networkx==3.1',
        'scikit_learn==1.3.2',
        'torch==2.3.0',
        'tqdm==4.64.1',
        'memory_profiler==0.57.0',
        'pandas',
        'numpy',
        'torch',
        'networkx',
        'matplotlib',
        'torch_geometric',
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
    # Include non-Python files
    package_data={
        '': ['scripts/your_script.sh'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'your_command=your_library.your_module:main_function',
        ],
    },
)