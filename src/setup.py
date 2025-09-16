from setuptools import setup, find_packages

setup(
    name="multimodal-fake-news-detection",
    version="1.0.0",
    description="Multimodal Fake News Detection with Vision Transformer and Text Encoders",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "transformers>=4.20.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "Pillow>=8.0.0",
        "requests>=2.25.0",
        "tqdm>=4.60.0",
        "kaggle>=1.5.0",
        "gdown>=4.0.0",
    ],
    python_requires=">=3.7",
)