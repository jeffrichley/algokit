"""Setup script for AlgoKit with MkDocs plugin support."""

from setuptools import find_packages, setup

setup(
    name="algokit",
    version="0.6.1",
    packages=find_packages(),
    entry_points={
        "mkdocs.plugins": [
            "dynamic-algorithm = macros.plugin:DynamicAlgorithmPlugin",
        ]
    },
    install_requires=[
        "mkdocs",
        "mkdocs-macros-plugin",
        "pyyaml",
    ],
)
