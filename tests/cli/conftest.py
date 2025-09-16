"""Shared test configuration for CLI tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def cli_test_session() -> Generator[dict, None, None]:
    """Session-scoped fixture for CLI test configuration."""
    config = {
        "test_timeout": 30,  # seconds
        "max_episodes": 10,  # for faster tests
        "default_environment": "CartPole-v1",
    }
    yield config


@pytest.fixture
def cli_test_output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory for CLI tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def cli_test_environment() -> str:
    """Default test environment for CLI tests."""
    return "CartPole-v1"


@pytest.fixture
def cli_test_episodes() -> int:
    """Default number of episodes for CLI tests."""
    return 5


@pytest.fixture
def cli_test_parameters() -> dict:
    """Default test parameters for CLI tests."""
    return {
        "episodes": 5,
        "learning_rate": 0.1,
        "gamma": 0.9,
        "epsilon": 0.1,
        "epsilon_decay": 0.995,
        "mode": "tabular",
        "lambda_value": 0.9,
        "verbose": False,
    }


# Pytest configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest for CLI tests."""
    config.addinivalue_line("markers", "cli: mark test as CLI test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection for CLI tests."""
    for item in items:
        # Add cli marker to all CLI tests
        if "cli" in str(item.fspath):
            item.add_marker(pytest.mark.cli)

        # Algorithm-specific markers will be added here as algorithms are implemented

        # Mark slow tests
        if "slow" in item.name or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
