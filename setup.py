from setuptools import find_packages, setup

setup(
    name="DD4ML",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "torchaudio>=2.5.1",
        "matplotlib>=3.10.0",
        "numpy>=1.24.4",
        "pandas>=1.5.3",
        "scipy>=1.10.1",
        "requests>=2.32.3",
        "wandb>=0.19.6",
        "pyyaml>=6.0.1",
    ],
)