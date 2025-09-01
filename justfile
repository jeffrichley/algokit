# Development shortcuts for Algorithm Kit
# Generated from the Seedling Copier template

# Run tests
test:
    nox -s tests

# Run linting
lint:
    nox -s lint

# Run type checking
type-check:
    nox -s type_check

# Build documentation
docs:
    nox -s docs

# Check documentation links
docs-linkcheck:
    nox -s docs_linkcheck

# Serve documentation locally
docs-serve:
    mkdocs serve

# Run all quality checks
quality: lint type-check docs-linkcheck

# Generate coverage report
coverage:
    nox -s coverage_html

# Run security audit
security:
    nox -s security

# Run complexity analysis
complexity:
    nox -s complexity

# Validate pyproject.toml
pyproject:
    nox -s pyproject

# Run pre-commit hooks
pre-commit:
    nox -s pre-commit

# Create a release PR
release:
    gh pr create --fill --title "chore: release"

# Install development dependencies
install:
    uv sync --all-extras

# Clean up generated files
clean:
    rm -rf .nox
    rm -rf htmlcov
    rm -rf site
    rm -rf .pytest_cache
    rm -rf .mypy_cache
    rm -rf .ruff_cache

# Show help
default:
    @just --list
