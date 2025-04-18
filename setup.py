from setuptools import setup, find_packages

setup(
    name="sam2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "pillow",
        "opencv-python",
        "tqdm",
        "hydra-core",
    ],
) 