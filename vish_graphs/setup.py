from setuptools import setup, find_packages

setup(
    name='vish_graph',
    version='0.2.0',
    packages=find_packages(),
    description='Graph processing library',
    long_description='A Python library for processing and visualizing graphs.',
    author='Vishesh Yadav',
    author_email='sciencely98@gmail.com',
    url='https://github.com/visheshyadav/vish_graphs',
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'networkx',
        'scikit-learn',
        'torch',
        'torchvision'
    ],
)