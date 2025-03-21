from setuptools import setup, find_packages

setup(
    name="cr_learn",
    version="0.1.3",
    description="CoreRec Learning Library",
    author= 'vishesh yadav' ,
    description='A Library Provides Bunch of datasets to speed up your recsys learing.',
    author_email="vishesh@corerec.tech",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "gdown",
        "tqdm",
        "requests"
    ],
    entry_points={
        'console_scripts': [
            'cr=cr_learn.cli:main',
        ],
    },
    python_requires='>=3.7',
) 