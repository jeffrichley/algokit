# Development shortcuts for Algorithm Kit
# Generated from the Seedling Copier template

# Run tests
test: install-dev
    nox -s "tests-3.12(mode='full')"

# Run linting
lint: install-dev
    nox -s lint

# Run type checking
type-check: install-dev
    nox -s type_check

# Build documentation
docs: install-docs
    nox -s docs

# Check documentation links
docs-linkcheck: install-docs
    nox -s docs_linkcheck

# Serve documentation locally
docs-serve: install-docs
    uv run mkdocs serve

# Serve documentation with live reload for development
docs-dev: install-docs
    uv run mkdocs serve --livereload --watch docs/

# Run all quality checks
quality: install-dev install-docs
    nox -s lint
    nox -s type_check
    nox -s docs_linkcheck
    nox -s spellcheck

# Generate coverage report
coverage: install-dev
    nox -s coverage_html

# Run security audit
security: install-dev
    nox -s security

# Run complexity analysis
complexity: install-dev
    nox -s complexity

# Validate pyproject.toml
pyproject: install-dev
    nox -s pyproject

# Run spell checking
spellcheck: install-dev
    nox -s spellcheck

# Run pre-commit hooks
pre-commit: install-dev
    nox -s pre-commit

# Create a release PR
release:
    gh pr create --fill --title "chore: release"

# Commitizen commands
commitizen-version:
    uv run python -m commitizen version --project

commitizen-bump:
    uv run python -m commitizen bump

commitizen-check:
    uv run python -m commitizen check --rev-range HEAD~1..HEAD

# Install core dependencies only (no groups)
install-core:
    uv sync

# Install development dependencies (dev group - includes testing)
install-dev:
    uv sync --extra dev

# Install documentation dependencies (docs group)
install-docs:
    uv sync --extra docs

# Install all dependencies including docs for full development
install-full:
    nox -s dev_full

# Install minimal dependencies (core only)
install-minimal:
    uv sync

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
