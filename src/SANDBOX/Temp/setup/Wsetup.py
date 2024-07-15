from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import platform
import subprocess

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        if platform.system() == "Windows":
            self.add_to_env_variable_windows()
        else:
            self.add_to_env_variable_unix()

    def add_to_env_variable_unix(self):
        # Unix (Mac and Linux) environment variable setup
        home_dir = os.path.expanduser("~")
        if platform.system() == "Darwin":
            shell_profile = os.path.join(home_dir, ".zshrc")  # For zsh
            if not os.path.exists(shell_profile):
                shell_profile = os.path.join(home_dir, ".bash_profile")  # Fallback to bash
        else:
            shell_profile = os.path.join(home_dir, ".bashrc")  # Linux systems

        corerec_path = os.path.join(home_dir, "your_library_files")
        engine_path = os.path.join(home_dir, "engine")
        
        # Ensure the directories exist
        os.makedirs(corerec_path, exist_ok=True)
        os.makedirs(engine_path, exist_ok=True)

        # Add the library paths to the shell profile
        with open(shell_profile, 'a') as f:
            f.write(f'\n# Add Corerec and Engine to PATH\n')
            f.write(f'export PATH="$PATH:{corerec_path}:{engine_path}"\n')

        print(f"Added {corerec_path} and {engine_path} to PATH in {shell_profile}")

    def add_to_env_variable_windows(self):
        # Windows environment variable setup
        corerec_path = os.path.join(os.path.expanduser("~"), "your_library_files")
        engine_path = os.path.join(os.path.expanduser("~"), "engine")
        
        # Ensure the directories exist
        os.makedirs(corerec_path, exist_ok=True)
        os.makedirs(engine_path, exist_ok=True)

        # Add the library paths to the PATH environment variable
        subprocess.run([
            'setx', 'PATH', f'{corerec_path};{engine_path};%PATH%'
        ], check=True)

        print(f"Added {corerec_path} and {engine_path} to PATH")

setup(
    name='corerec',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your library dependencies here
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
    entry_points={
        'console_scripts': [
            'corerec=corerec:main_function',
        ],
    },
)