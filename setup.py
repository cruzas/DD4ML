from setuptools import setup, find_packages

setup(
    name="DD4ML",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "requests",
        "wandb",
        "pyyaml",
    ],
)