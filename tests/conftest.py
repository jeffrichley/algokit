"""Global test configuration for RL algorithms.

This module provides shared fixtures and configuration for all tests,
encouraging fixture reuse while maintaining test isolation.
"""

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import matplotlib
import pytest

# Configure matplotlib for testing (cross-platform)
matplotlib.use("Agg")

# Ensure src is in Python path for all platforms
if "src" not in sys.path:
    sys.path.insert(0, "src")


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest for RL testing."""
    # Markers are now defined in pytest.ini
    pass


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection for automatic marking."""
    for item in items:
        # Add RL marker to all RL tests
        if "rl" in str(item.fspath) or any(
            algo in str(item.fspath) for algo in ["q_learning", "dqn", "ppo"]
        ):
            item.add_marker(pytest.mark.rl)

        # Add algorithm-specific markers
        if "q_learning" in str(item.fspath):
            item.add_marker(pytest.mark.q_learning)
        elif "dqn" in str(item.fspath):
            item.add_marker(pytest.mark.dqn)
        elif "ppo" in str(item.fspath):
            item.add_marker(pytest.mark.ppo)

        # Mark slow tests
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark benchmark tests
        if "benchmark" in str(item.fspath) or "benchmark" in item.name:
            item.add_marker(pytest.mark.benchmark)


# Global fixtures
@pytest.fixture(scope="session")
def test_output_dir() -> Generator[Path, None, None]:
    """Session-scoped test output directory (cross-platform)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir).resolve()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Function-scoped temporary directory (cross-platform)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir).resolve()


@pytest.fixture
def test_data_dir() -> Path:
    """Test data directory (cross-platform)."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def cross_platform_path() -> Path:
    """Cross-platform path helper for tests."""
    return Path.cwd().resolve()


# Test environment setup
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment for all tests (cross-platform)."""
    # Use non-interactive backend for matplotlib
    matplotlib.use("Agg")

    # Ensure src is in Python path (cross-platform)
    src_path = Path("src").resolve()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Set environment variables for consistent testing
    os.environ["PYTHONPATH"] = str(src_path)
    os.environ["MATPLOTLIB_BACKEND"] = "Agg"

    # Platform-specific environment setup
    if os.name == "nt":  # Windows
        os.environ["PYTHONIOENCODING"] = "utf-8"

    yield

    # Cleanup if needed
    # Remove src from sys.path if we added it
    if str(src_path) in sys.path:
        sys.path.remove(str(src_path))


# Import fixtures from modules to make them available globally
# Import the mock environment factory function directly


@pytest.fixture(scope="session")
def fast_cartpole():
    """Fast CartPole environment for integration tests."""
    import gymnasium as gym

    return gym.make("CartPole-v1", render_mode=None)


# SARSA fixtures removed - no longer implemented


@pytest.fixture(scope="session")
def rl_assertions():
    """RL-specific assertion helpers."""
    from unittest.mock import Mock

    mock_assertions = Mock()
    mock_assertions.assert_training_results = Mock()
    mock_assertions.assert_convergence = Mock()
    mock_assertions.assert_q_table_valid = Mock()
    return mock_assertions




# CLI Testing fixtures
@pytest.fixture(scope="function")
def cli_runner():
    """CLI runner for testing Typer commands."""
    from typer.testing import CliRunner

    return CliRunner(echo_stdin=False, catch_exceptions=False)


@pytest.fixture(scope="function")
def app():
    """Main CLI app for testing."""
    from algokit.cli.main import app

    return app


# Additional cross-platform fixtures
@pytest.fixture
def mock_file_system(temp_dir):
    """Mock file system for testing (cross-platform)."""

    # Create a mock file structure
    mock_files = {
        "config.yaml": "test: true\n",
        "data.json": '{"test": "data"}\n',
        "output/": None,  # Directory
        "logs/": None,  # Directory
    }

    for path, content in mock_files.items():
        full_path = temp_dir / path
        if content is None:  # Directory
            full_path.mkdir(parents=True, exist_ok=True)
        else:  # File
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")

    yield temp_dir

    # Cleanup is handled by temp_dir fixture


@pytest.fixture
def platform_info():
    """Platform information for cross-platform testing."""
    import platform

    return {
        "system": platform.system(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture()[0],
        "is_windows": os.name == "nt",
        "is_unix": os.name == "posix",
    }


@pytest.fixture(scope="session")
def test_timeout():
    """Test timeout configuration (cross-platform)."""
    # Adjust timeouts based on platform
    if os.name == "nt":  # Windows
        return 60  # Windows can be slower
    else:
        return 30  # Unix-like systems


# Fixture for encouraging reuse without over-policing
@pytest.fixture
def shared_test_data():
    """Shared test data that can be reused across tests.

    This fixture encourages fixture reuse while maintaining
    test isolation. Use this for read-only test data.
    """
    return {
        "sample_config": {
            "episodes": 10,
            "learning_rate": 0.1,
            "discount_factor": 0.9,
        },
        "sample_observations": [0.1, 0.2, 0.3, 0.4],
        "sample_actions": [0, 1],
        "sample_rewards": [1.0, 0.0, 1.0],
    }
