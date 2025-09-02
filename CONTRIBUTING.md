# 🤝 Contributing to Algorithm Kit

Thank you for your interest in contributing to Algorithm Kit! This document provides guidelines and requirements for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Quality Requirements](#quality-requirements)
- [Contribution Process](#contribution-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Review Process](#review-process)

## 📜 Code of Conduct

By participating in this project, you agree to maintain a respectful environment for everyone. We are committed to providing a welcoming and inspiring community for all.

## 🛠️ Development Setup

### Prerequisites

- **Python 3.12+**: Required for modern type hints and features
- **uv**: Fast Python package manager (recommended)
- **Git**: Version control system

### Installation

```bash
# Clone the repository
git clone https://github.com/jeffrichley/algokit.git
cd algokit

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -r requirements.txt

# Install in editable mode for development
uv pip install -e ".[dev]"
```

### Development Environment

```bash
# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows

# Verify installation
uv run dev info
```

## 🎯 Quality Requirements

Algorithm Kit maintains strict quality standards. **All contributions must pass these quality gates:**

### Quality Gates

1. **🔍 Linting (Ruff)**: Code style and best practices
2. **🧠 Type Safety (MyPy)**: Strict type checking with zero errors
3. **📊 Coverage (≥80%)**: Comprehensive test coverage
4. **📈 Complexity (Xenon)**: Maintainable code complexity
5. **🧹 Dead Code (Vulture)**: No unused code or imports
6. **🔒 Security (pip-audit)**: No dependency vulnerabilities


### Pre-commit Checks

```bash
# Run all quality checks
uv run dev checkit

# Individual checks
uv run dev lint          # Ruff linting
uv run dev typecheck     # MyPy type checking
uv run dev test          # Pytest with coverage
uv run dev quality-gates # All quality gates
```

## 🔄 Contribution Process

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/algokit.git
cd algokit

# Add upstream remote
git remote add upstream https://github.com/jeffrichley/algokit.git
```

### 2. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/your-bug-description
```

### 3. Make Your Changes

- Follow the [Code Style Guidelines](#code-style-guidelines)
- Write comprehensive tests
- Update documentation as needed
- Ensure all quality gates pass

### 4. Test Your Changes

```bash
# Run all tests
uv run dev test

# Run quality gates
uv run dev quality-gates

# Run benchmarks (if applicable)
uv run dev benchmark
```

### 5. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add new data processing feature

- Implement crossfade transition
- Add comprehensive tests
- Update documentation
- Fixes #123"
```

### 6. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create a Pull Request on GitHub
```

## 📝 Code Style Guidelines

### Python Code Style

- **Black**: Automatic code formatting (line length: 88)
- **Ruff**: Linting and import sorting
- **Type Hints**: Required for all functions and methods
- **Docstrings**: Google-style docstrings for all public APIs

### Import Organization

```python
# Standard library imports
import os
import sys
from typing import List, Optional

# Third-party imports
import numpy as np
import pydantic

# Local imports
from algokit.models import ExampleModel
from algokit.core import CoreClass
```

### Type Annotations

```python
# Required for all functions
def process_data(spec: DataSpec, output_path: str) -> bool:
    """Process a data specification and save to output path.

    Args:
        spec: The data specification to process
        output_path: Path where the data will be saved

    Returns:
        True if processing was successful, False otherwise
    """
    pass
```

### Error Handling

```python
# Use specific exceptions
try:
    result = process_data(data)
except ValueError as e:
    logger.error(f"Invalid data format: {e}")
    raise
except FileNotFoundError as e:
    logger.error(f"Required file not found: {e}")
    raise
```

## 🧪 Testing Requirements

### Test Coverage

- **Minimum Coverage**: 80% overall
- **New Code**: 100% coverage required
- **Critical Paths**: 100% coverage required

### Test Structure

```python
# tests/unit/test_data_spec.py
import pytest
from algokit.models import ExampleModel

class TestDataSpec:
    def test_data_spec_creation(self):
        """Test DataSpec creation with valid parameters."""
        spec = DataSpec(
            format="json",
            encoding="utf-8",
            compression="gzip"
        )
        assert spec.format == "json"
        assert spec.encoding == "utf-8"
        assert spec.compression == "gzip"

    def test_data_spec_validation(self):
        """Test DataSpec validation with invalid parameters."""
        with pytest.raises(ValueError):
            DataSpec(format="invalid", encoding="utf-8", compression="gzip")
```

### Property-Based Testing

```python
# Use Hypothesis for property-based testing
from hypothesis import given, strategies as st

@given(st.sampled_from(["json", "csv", "xml", "yaml"]))
def test_data_format_validation(format_type):
    """Test that data format validation works for all valid values."""
    spec = DataSpec(format=format_type, encoding="utf-8", compression="gzip")
    assert spec.format == format_type
```

## 📚 Documentation Standards

### Docstring Requirements

- **All public APIs**: Must have Google-style docstrings
- **Parameters**: Document all parameters with types
- **Returns**: Document return values and types
- **Examples**: Include usage examples for complex functions

### Example Docstring

```python
def process_data_batch(
    data: List[Dict[str, Any]],
    processor: Optional[DataProcessor] = None,
    batch_size: Optional[int] = None
) -> ProcessedData:
    """Process a batch of data records.

    Args:
        data: List of data records to process
        processor: Optional custom processor (uses default if None)
        batch_size: Optional batch size (auto-calculated if None)

    Returns:
        A ProcessedData object containing the processed results

    Raises:
        ValueError: If data list is empty or invalid
        TypeError: If data contains invalid types

    Example:
        >>> data = [{"id": 1, "value": "test"}, {"id": 2, "value": "example"}]
        >>> result = process_data_batch(data)
        >>> result.save("output.json")
    """
    pass
```

### README Updates

- Update README.md for user-facing changes
- Update API documentation for technical changes
- Include examples for new features

## 👀 Review Process

### Pull Request Requirements

1. **Quality Gates**: All quality gates must pass
2. **Tests**: All tests must pass
3. **Documentation**: Updated as needed
4. **Description**: Clear description of changes
5. **Linked Issues**: Reference related issues

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Type hints are complete and correct
- [ ] Tests are comprehensive and pass
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

### Review Timeline

- **Initial Review**: Within 48 hours
- **Follow-up Reviews**: Within 24 hours
- **Merge**: After approval and CI passes

## 🚀 Getting Help

### Questions and Discussions

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the docs/ directory for detailed guides

### Development Resources

- **API Documentation**: `docs/api/`
- **Architecture Guide**: `docs/ARCHITECTURE_PATTERNS.md`
- **Quality Checks**: Run `nox -s lint type_check complexity security` for comprehensive quality validation
- **Project Status**: `docs/STATUS.md`

## 🎉 Recognition

Contributors will be recognized in:

- **README.md**: For significant contributions
- **CHANGELOG.md**: For all contributions
- **GitHub Contributors**: Automatic recognition

Thank you for contributing to Algorithm Kit! 🚀
