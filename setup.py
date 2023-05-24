""" Setup
"""
from os import path

from setuptools import find_packages, setup

__version__ = "0.0.1"

setup(
    name="orsac_label_verification",
    version=__version__,
    description="Iterative Label Verification Model",
    author="ANON",
    author_email="anon",
    packages=["orsac_label_verification"],
    include_package_data=True,
    install_requires=[
        "torchvision",
        "natsort",
        "efficientnet_pytorch",
        "torch",
        "numpy",
        "opencv-python",
        "Pillow",
        "pandas",
        "scikit-image",
        "albumentations",
        "scikit-learn",
        "pretrainedmodels",
        "fastai",
        "python-dotenv",
        "torchsampler",
        "IPython",
        "seaborn"
    ],
    python_requires=">=3.9",
)
