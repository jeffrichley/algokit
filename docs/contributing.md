# Contributing to Algorithm Kit

Thank you for your interest in contributing to Algorithm Kit! This guide will help you get started with contributing to the project.

## Development Setup

### Prerequisites

- Python 3.12+
- uv (Python package manager)
- just (command runner)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jeffrichley/algokit.git
   cd algokit
   ```

2. Install dependencies:
   ```bash
   uv sync --all-extras
   ```

3. Install the package in editable mode:
   ```bash
   uv pip install -e .
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
just test

# Run tests with coverage
just coverage
```

### Code Quality

```bash
# Run linting
just lint

# Run type checking
just type-check

# Run all quality checks
just quality
```

### Documentation

```bash
# Build documentation
just docs

# Check documentation links
just docs-linkcheck
```

## Code Standards

### Python Style

- Follow PEP 8 style guidelines
- Use Black for code formatting
- Use Ruff for linting
- Maximum line length: 88 characters

### Type Hints

- All public functions and methods must have type hints
- Use mypy for static type checking
- Avoid using `Any` type

### Testing

- Write tests for all new functionality
- Use pytest as the testing framework
- Aim for high test coverage
- Use descriptive test names

### Documentation

- Use Google-style docstrings
- Include examples in docstrings
- Keep documentation up to date

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation if needed
7. Submit a pull request

## Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

## Getting Help

- Open an issue for bugs or feature requests
- Join our discussions for questions
- Check existing documentation

Thank you for contributing to Algorithm Kit!
