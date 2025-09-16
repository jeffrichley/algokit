"""Common mocks for RL testing."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_progress_tracker():
    """Mock progress tracker for testing."""
    mock_tracker = Mock()
    mock_task = Mock()
    mock_tracker.track_task.return_value.__enter__ = Mock(return_value=mock_task)
    mock_tracker.track_task.return_value.__exit__ = Mock(return_value=None)
    return mock_tracker


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    mock_logger = Mock()
    mock_logger.info = Mock()
    mock_logger.warning = Mock()
    mock_logger.error = Mock()
    mock_logger.debug = Mock()
    return mock_logger


@pytest.fixture
def mock_console():
    """Mock Rich console for testing."""
    mock_console = Mock()
    mock_console.print = Mock()
    mock_console.log = Mock()
    return mock_console


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing."""
    mock_ops = Mock()
    mock_ops.save_model = Mock()
    mock_ops.load_model = Mock()
    mock_ops.save_plots = Mock()
    mock_ops.create_directory = Mock()
    return mock_ops


@pytest.fixture
def mock_metrics():
    """Mock metrics for testing."""
    mock_metrics = Mock()
    mock_metrics.episode_rewards = [100, 120, 140, 150]
    mock_metrics.episode_lengths = [10, 12, 14, 15]
    mock_metrics.epsilon_values = [0.1, 0.0995, 0.099, 0.0985]
    mock_metrics.convergence_episode = None
    mock_metrics.final_epsilon = 0.0985
    mock_metrics.total_updates = 100
    return mock_metrics
