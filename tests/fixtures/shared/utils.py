"""Shared test utilities."""

import json
import pickle
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_model_file(temp_output_dir: Path) -> Path:
    """Create a temporary model file for testing."""
    model_file = temp_output_dir / "test_model.pkl"
    # Create a dummy model file
    dummy_data = {"test": "data", "q_table": [[0.1, 0.2], [0.3, 0.4]]}
    with open(model_file, "wb") as f:
        pickle.dump(dummy_data, f)
    return model_file


@pytest.fixture
def temp_config_file(temp_output_dir: Path) -> Path:
    """Create a temporary config file for testing."""
    config_file = temp_output_dir / "test_config.json"
    dummy_config = {
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "epsilon": 0.1,
        "episodes": 100,
    }
    with open(config_file, "w") as f:
        json.dump(dummy_config, f)
    return config_file


def create_test_data(data: dict[str, Any], file_path: Path) -> None:
    """Create test data file."""
    if file_path.suffix == ".json":
        with open(file_path, "w") as f:
            json.dump(data, f)
    elif file_path.suffix == ".pkl":
        with open(file_path, "wb") as f:
            pickle.dump(data, f)


def load_test_data(file_path: Path) -> dict[str, Any]:
    """Load test data file."""
    if file_path.suffix == ".json":
        with open(file_path) as f:
            return json.load(f)
    elif file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


@pytest.fixture
def test_data_manager():
    """Test data manager fixture."""
    return {"create": create_test_data, "load": load_test_data}
