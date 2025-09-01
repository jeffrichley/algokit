# API Reference

Welcome to the Algorithm Kit API reference. This page provides comprehensive documentation for all public APIs and modules.

## Package Overview

Algorithm Kit is organized into logical modules for different types of algorithms and utilities.

## Core Modules

### `algokit`

The main package entry point.

```python
import algokit

# Check version
print(algokit.__version__)
```

## Development

This documentation is automatically generated from the source code. To contribute:

1. Add proper docstrings to your functions and classes
2. Use Google-style docstring format
3. Include type hints for all parameters and return values
4. Add examples in your docstrings

## Building Documentation

To build the documentation locally:

```bash
# Install dependencies
uv pip install -e .[docs]

# Build docs
mkdocs build

# Serve docs locally
mkdocs serve
```

## Contributing

See our [Contributing Guide](contributing.md) for detailed information on how to contribute to the project and documentation.
