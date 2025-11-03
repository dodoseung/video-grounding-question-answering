"""Setup script for VGQA package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="vgqa",
    version="1.0.0",
    author="VGQA Team",
    description="Video Grounding and Question Answering - A unified framework for spatio-temporal video grounding and video QA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/vgqa",
    packages=find_packages(exclude=["tests", "tools", "app", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "transformers>=4.30.0",
        "decord>=0.6.0",
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "pyyaml>=5.4.0",
        "yacs>=0.1.8",
        "tqdm>=4.60.0",
        "scipy>=1.5.0",
        "easydict>=1.9",
        "timm>=0.4.12",
    ],
    extras_require={
        "web": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "python-multipart>=0.0.6",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "all": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "python-multipart>=0.0.6",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vgqa-infer-grounding=tools.infer_grounding:main",
            "vgqa-infer-qa=tools.infer_qa:main",
            "vgqa-train=tools.train:main",
            "vgqa-evaluate=tools.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
