"""Nox automation file for Algorithm Kit.

Key features:
- Uses `uv` for dependency installation.
- Supports Python 3.12+ (matrix-ready).
- Sessions: tests, lint, type_check, docs, precommit, coverage_html, complexity, security, pyproject.
- Reuses local virtualenvs for speed; CI passes `--force-python` to isolate.
- Parametrized "mode" for minimal vs full extras install.

Generated from the Seedling Copier template.
"""

from pathlib import Path

import nox

# -------- Global config -------- #
nox.options.sessions = [
    "tests",
    "lint",
    "type_check",
    "docs",
    "complexity",
    "security",
    "pyproject",
    "spellcheck",
]
# Reuse existing venvs locally for speed; CI can override with --no-reuse-existing-virtualenvs
nox.options.reuse_existing_virtualenvs = True

PROJECT_ROOT = Path(__file__).parent

PYTHON_VERSIONS = ["3.12"]  # Update when a stable 3.13 lands
INSTALL_MODES = ["minimal", "full"]  # "minimal" == core deps only; "full" == dev[all]


def install_project(session, mode: str = "minimal"):
    """Install the project with uv and the chosen extras mode."""
    assert mode in INSTALL_MODES

    if mode == "minimal":
        # Just core dependencies
        session.run("uv", "pip", "install", "-q", "-e", ".", external=True)
    elif mode == "full":
        # Development: main + dev dependencies
        session.run("uv", "pip", "install", "-q", "-e", ".[dev]", external=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# -------- Sessions -------- #


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("mode", INSTALL_MODES)
def tests(session, mode):
    """Run pytest with coverage."""
    install_project(session, mode)

    # For minimal mode, we need to install test dependencies separately
    if mode == "minimal":
        session.run("uv", "pip", "install", "-q", "pytest", "pytest-cov", external=True)

    session.run(
        "pytest",
        "tests",
        "--disable-warnings",
    )


@nox.session(python=PYTHON_VERSIONS[0])
def lint(session):
    """Run Ruff autofix + Black check."""
    install_project(session, "full")
    session.run("ruff", "check", "src", "tests", "--fix")
    session.run("black", "--check", "src", "tests")


@nox.session(python=PYTHON_VERSIONS[0])
def type_check(session):
    """Run MyPy type checks."""
    install_project(session, "full")
    session.run("mypy", "src", "tests")


@nox.session(python=PYTHON_VERSIONS[0])
def docs(session):
    """Build MkDocs docs."""
    install_project(session, "full")
    # Install docs dependencies on top of dev dependencies
    session.run("uv", "pip", "install", "-q", "-e", ".[docs]", external=True)
    session.run("mkdocs", "build", external=True)


@nox.session(python=PYTHON_VERSIONS[0])
def docs_linkcheck(session):
    """Check MkDocs docs links."""
    install_project(session, "full")
    # Install docs dependencies on top of dev dependencies
    session.run("uv", "pip", "install", "-e", ".[docs]", external=True)
    # Build docs to check for internal link issues
    session.run("mkdocs", "build", external=True)


@nox.session(python=PYTHON_VERSIONS[0], name="pre-commit")
def precommit_hooks(session):
    """Run pre-commit hooks on all files."""
    install_project(session, "full")
    session.run("pre-commit", "run", "--all-files")


@nox.session(python=PYTHON_VERSIONS[0])
def coverage_html(session):
    """Generate an HTML coverage report."""
    install_project(session, "full")
    session.run("coverage", "html")
    html_path = PROJECT_ROOT / "htmlcov" / "index.html"
    session.log(f"Generated coverage HTML at {html_path.as_uri()}")


# ---------------- Extra quality sessions ---------------- #


@nox.session(python=PYTHON_VERSIONS[0])
def complexity(session):
    """Fail if cyclomatic complexity exceeds score B."""
    install_project(session, "full")
    # Tweak --max-absolute (A=0, B=10, C=20) to your tolerance
    session.run("xenon", "--max-absolute", "B", "src")


@nox.session(python=PYTHON_VERSIONS[0])
def security(session):
    """Run pip-audit against project dependencies."""
    install_project(session, "full")
    # Audit direct + transitive deps pinned in uv.lock
    session.run("pip-audit", "--progress-spinner=off")


@nox.session(python=PYTHON_VERSIONS[0])
def pyproject(session):
    """Validate pyproject.toml configuration."""
    install_project(session, "full")
    session.run("validate-pyproject", "pyproject.toml")


@nox.session(python=PYTHON_VERSIONS[0])
def spellcheck(session):
    """Run codespell to check for spelling errors."""
    install_project(session, "full")
    session.run(
        "codespell", "src", "tests", "docs", "--ignore-words-list=algokit,jeffrichley"
    )


@nox.session(python=PYTHON_VERSIONS[0])
def dev_full(session):
    """Install all development dependencies including docs for full development setup."""
    install_project(session, "full")
    # Install docs dependencies on top of dev dependencies
    session.run("uv", "pip", "install", "-q", "-e", ".[docs]", external=True)
    session.log(
        "Full development environment installed with main + dev (includes testing, linting, type checking) + docs dependencies"
    )
