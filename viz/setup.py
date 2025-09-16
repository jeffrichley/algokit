"""Setup script for AGLoViz - Algorithm Visualization CLI."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read version from __init__.py
init_file = Path(__file__).parent / "agloviz" / "__init__.py"
version = "0.1.0"
if init_file.exists():
    for line in init_file.read_text().splitlines():
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break

setup(
    name="agloviz",
    version=version,
    description="AGLoViz - Algorithm Visualization CLI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jeff Richley",
    author_email="jeffrichley@gmail.com",
    url="https://github.com/jeffrichley/algokit",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "typer[all]>=0.9.0",
        "rich>=13.0.0",
        "manim>=0.17.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agloviz=agloviz.main:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="algorithm visualization manim cli education computer-science",
    project_urls={
        "Bug Reports": "https://github.com/jeffrichley/algokit/issues",
        "Source": "https://github.com/jeffrichley/algokit",
        "Documentation": "https://algokit.readthedocs.io/",
    },
)
