#!/usr/bin/env python3
"""
Setup script for the Chinese Genealogy Text Processor
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="genealogy-text-processor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A pipeline for processing Chinese genealogy texts from OCR to structured data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/genealogy-text-processor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Natural Language :: Chinese (Traditional)",
        "Natural Language :: Chinese (Simplified)",
    ],
    python_requires=">=3.8",
    install_requires=[
        "google-generativeai>=0.3.0",
        "PyYAML>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "enhanced": [
            "python-dotenv>=1.0.0",
            "tqdm>=4.66.0",
            "colorama>=0.4.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "genealogy-process=pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.txt"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/genealogy-text-processor/issues",
        "Source": "https://github.com/yourusername/genealogy-text-processor",
    },
)