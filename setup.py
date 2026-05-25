from setuptools import setup, find_packages

setup(
    name="smeta-zsl",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.29.0",
        "datasets>=2.18.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scipy>=1.13.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "matplotlib>=3.8.0",
    ],
)
