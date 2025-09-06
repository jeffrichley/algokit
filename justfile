# Development shortcuts for Algorithm Kit


# Run tests
test: install-dev
    uv run pytest tests

# Run tests with coverage
test-cov: install-dev
    uv run pytest tests --cov=src/algokit --cov-report=term-missing --cov-report=html

# Run linting and formatting
lint: install-dev
    uv run ruff check src tests --fix
    uv run ruff format src tests --check

# Format code
format: install-dev
    uv run ruff format src tests

# Run type checking
type-check: install-dev
    uv run mypy src tests

# Build documentation
docs: install-docs
    uv run mkdocs build

# Check documentation links
docs-linkcheck: install-docs
    uv run mkdocs build
    uv run python -m linkcheckmd docs

# Serve documentation locally
docs-serve: install-docs
    uv run mkdocs serve

# Serve documentation with live reload for development
docs-dev: install-docs
    uv run mkdocs serve --livereload --watch docs/

# Validate YAML files
validate-yaml: install-docs
    uv run --group docs python scripts/validate_yaml.py validate

# Validate YAML files with verbose output
validate-yaml-verbose: install-docs
    uv run --group docs python scripts/validate_yaml.py validate --verbose

# Run all quality checks
quality: install-dev install-docs
    uv run ruff check src tests --fix
    uv run ruff format src tests --check
    uv run mypy src tests
    uv run mkdocs build
    uv run python -m linkcheckmd docs
    uv run codespell src tests docs --ignore-words-list=algokit,jeffrichley

# Generate coverage report
coverage: install-dev
    uv run coverage html
    @echo "Generated coverage HTML at htmlcov/index.html"

# Run security audit
security: install-dev
    uv run pip-audit --progress-spinner=off

# Run complexity analysis
complexity: install-dev
    uv run xenon --max-absolute B src

# Validate pyproject.toml
pyproject: install-dev
    uv run validate-pyproject pyproject.toml

# Run spell checking
spellcheck: install-dev
    uv run codespell src tests docs --ignore-words-list=algokit,jeffrichley

# Run pre-commit hooks
pre-commit: install-dev
    uv run pre-commit run --all-files

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
    uv sync --group dev
    # uv sync --extra dev

# Install documentation dependencies (docs group)
install-docs:
    uv sync --group docs
    # uv sync --extra docs

# Install all dependencies including docs for full development
install-full:
    uv sync --group dev --group docs

# Install minimal dependencies (core only)
install-minimal:
    uv sync

# Clean up generated files
clean:
    rm -rf htmlcov
    rm -rf site
    rm -rf .pytest_cache
    rm -rf .mypy_cache
    rm -rf .ruff_cache

# Show help
default:
    @just --list
