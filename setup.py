from setuptools import setup, find_namespace_packages

setup(
    name="sam2",
    version="0.1.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "gradio>=3.35.2",
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
    ],
    python_requires=">=3.8",
    description="SAM2 - Segment Anything Model 2",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/SAM2Mask",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 