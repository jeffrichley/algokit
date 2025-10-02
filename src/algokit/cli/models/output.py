"""Output artifact and run metadata models for CLI operations.

This module provides Pydantic models for tracking and managing output artifacts,
run metadata, and provenance information for CLI operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


@dataclass
class ArtifactCreationParameters:
    """Parameters for creating an artifact from a file."""

    file_path: Path
    algorithm: str
    family: str
    run_id: str
    artifact_type: ArtifactType
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class RunCreationParameters:
    """Parameters for creating a run metadata instance."""

    run_id: str
    run_type: RunType
    algorithm: str
    family: str
    output_dir: Path
    config: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None
    system_info: dict[str, Any] | None = None


class ArtifactType(str, Enum):
    """Output artifact type enumeration."""

    MODEL = "model"
    LOG = "log"
    PLOT = "plot"
    VIDEO = "video"
    METRIC = "metric"
    CONFIG = "config"
    CHECKPOINT = "checkpoint"
    RESULT = "result"
    REPORT = "report"
    OTHER = "other"


class RunStatus(str, Enum):
    """Run status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class RunType(str, Enum):
    """Run type enumeration."""

    TRAIN = "train"
    REPLAY = "replay"
    DEMO = "demo"
    TEST = "test"
    BENCHMARK = "benchmark"
    COMPARE = "compare"


class OutputArtifact(BaseModel):
    """Model for tracking output artifacts."""

    # Basic information
    name: str = Field(..., description="Artifact name")
    type: ArtifactType = Field(..., description="Artifact type")
    path: Path = Field(..., description="File path")
    size_bytes: int = Field(..., description="File size in bytes")

    # Metadata
    description: str | None = Field(default=None, description="Artifact description")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    modified_at: datetime = Field(
        default_factory=datetime.now, description="Last modification timestamp"
    )

    # Provenance
    algorithm: str = Field(..., description="Algorithm that generated this artifact")
    family: str = Field(..., description="Algorithm family")
    run_id: str = Field(..., description="Run ID that generated this artifact")

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("size_bytes")
    @classmethod
    def validate_size_bytes(cls, v: int) -> int:
        """Validate size is non-negative."""
        if v < 0:
            raise ValueError("size_bytes must be non-negative")
        return v

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str | Path) -> Path:
        """Validate path is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v

    def exists(self) -> bool:
        """Check if the artifact file exists.

        Returns:
            True if file exists, False otherwise
        """
        return self.path.exists()

    def get_relative_path(self, base_path: Path) -> Path:
        """Get relative path from base path.

        Args:
            base_path: Base path to calculate relative path from

        Returns:
            Relative path
        """
        try:
            return self.path.relative_to(base_path)
        except ValueError:
            return self.path

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "type": self.type.value,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "algorithm": self.algorithm,
            "family": self.family,
            "run_id": self.run_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_file(cls, params: ArtifactCreationParameters) -> OutputArtifact:
        """Create artifact from existing file.

        Args:
            params: Artifact creation parameters

        Returns:
            OutputArtifact instance

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not params.file_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {params.file_path}")

        stat = params.file_path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime)
        modified_at = datetime.fromtimestamp(stat.st_mtime)

        return cls(
            name=params.name or params.file_path.name,
            type=params.artifact_type,
            path=params.file_path,
            size_bytes=stat.st_size,
            description=params.description,
            created_at=created_at,
            modified_at=modified_at,
            algorithm=params.algorithm,
            family=params.family,
            run_id=params.run_id,
            metadata=params.metadata or {},
        )


class RunMetadata(BaseModel):
    """Model for tracking run metadata and information."""

    # Basic information
    run_id: str = Field(..., description="Unique run identifier")
    run_type: RunType = Field(..., description="Type of run")
    algorithm: str = Field(..., description="Algorithm name")
    family: str = Field(..., description="Algorithm family")

    # Timing information
    started_at: datetime = Field(
        default_factory=datetime.now, description="Run start time"
    )
    completed_at: datetime | None = Field(
        default=None, description="Run completion time"
    )
    duration_seconds: float | None = Field(
        default=None, description="Run duration in seconds"
    )

    # Status and results
    status: RunStatus = Field(default=RunStatus.PENDING, description="Run status")
    exit_code: int | None = Field(default=None, description="Process exit code")
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )

    # Configuration and parameters
    config: dict[str, Any] = Field(
        default_factory=dict, description="Run configuration"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm parameters"
    )

    # Output information
    output_dir: Path = Field(..., description="Output directory path")
    artifacts: list[OutputArtifact] = Field(
        default_factory=list, description="Generated artifacts"
    )

    # Performance metrics
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Performance metrics"
    )

    # System information
    system_info: dict[str, Any] = Field(
        default_factory=dict, description="System information"
    )

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, v: float | None) -> float | None:
        """Validate duration is non-negative."""
        if v is not None and v < 0:
            raise ValueError("duration_seconds must be non-negative")
        return v

    @field_validator("exit_code")
    @classmethod
    def validate_exit_code(cls, v: int | None) -> int | None:
        """Validate exit code is valid."""
        if v is not None and not isinstance(v, int):
            raise ValueError("exit_code must be an integer")
        return v

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str | Path) -> Path:
        """Validate output directory is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v

    def add_artifact(self, artifact: OutputArtifact) -> None:
        """Add an artifact to this run.

        Args:
            artifact: Artifact to add
        """
        self.artifacts.append(artifact)

    def get_artifacts_by_type(
        self, artifact_type: ArtifactType
    ) -> list[OutputArtifact]:
        """Get artifacts of a specific type.

        Args:
            artifact_type: Type of artifacts to retrieve

        Returns:
            List of artifacts of the specified type
        """
        return [
            artifact for artifact in self.artifacts if artifact.type == artifact_type
        ]

    def get_total_size_bytes(self) -> int:
        """Get total size of all artifacts in bytes.

        Returns:
            Total size in bytes
        """
        return sum(artifact.size_bytes for artifact in self.artifacts)

    def get_artifact_count(self) -> int:
        """Get total number of artifacts.

        Returns:
            Number of artifacts
        """
        return len(self.artifacts)

    def is_completed(self) -> bool:
        """Check if the run is completed.

        Returns:
            True if completed, False otherwise
        """
        return self.status == RunStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if the run failed.

        Returns:
            True if failed, False otherwise
        """
        return self.status in [RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.TIMEOUT]

    def is_running(self) -> bool:
        """Check if the run is currently running.

        Returns:
            True if running, False otherwise
        """
        return self.status == RunStatus.RUNNING

    def mark_completed(self, exit_code: int = 0) -> None:
        """Mark the run as completed.

        Args:
            exit_code: Process exit code
        """
        self.status = RunStatus.COMPLETED
        self.completed_at = datetime.now()
        self.exit_code = exit_code
        if self.started_at:
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()

    def mark_failed(self, error_message: str, exit_code: int = 1) -> None:
        """Mark the run as failed.

        Args:
            error_message: Error message
            exit_code: Process exit code
        """
        self.status = RunStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
        self.exit_code = exit_code
        if self.started_at:
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()

    def mark_cancelled(self) -> None:
        """Mark the run as cancelled."""
        self.status = RunStatus.CANCELLED
        self.completed_at = datetime.now()
        self.exit_code = -1
        if self.started_at:
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()

    def mark_timeout(self) -> None:
        """Mark the run as timed out."""
        self.status = RunStatus.TIMEOUT
        self.completed_at = datetime.now()
        self.exit_code = -2
        if self.started_at:
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()

    def get_run_directory(self) -> Path:
        """Get the run directory path.

        Returns:
            Run directory path
        """
        return self.output_dir

    def get_config_file_path(self) -> Path:
        """Get the configuration file path.

        Returns:
            Configuration file path
        """
        return self.output_dir / "config.yaml"

    def get_logs_directory(self) -> Path:
        """Get the logs directory path.

        Returns:
            Logs directory path
        """
        return self.output_dir / "logs"

    def get_models_directory(self) -> Path:
        """Get the models directory path.

        Returns:
            Models directory path
        """
        return self.output_dir / "models"

    def get_plots_directory(self) -> Path:
        """Get the plots directory path.

        Returns:
            Plots directory path
        """
        return self.output_dir / "plots"

    def get_videos_directory(self) -> Path:
        """Get the videos directory path.

        Returns:
            Videos directory path
        """
        return self.output_dir / "videos"

    def get_metrics_directory(self) -> Path:
        """Get the metrics directory path.

        Returns:
            Metrics directory path
        """
        return self.output_dir / "metrics"

    def create_directories(self) -> None:
        """Create all necessary directories for this run."""
        directories = [
            self.output_dir,
            self.get_logs_directory(),
            self.get_models_directory(),
            self.get_plots_directory(),
            self.get_videos_directory(),
            self.get_metrics_directory(),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def save_metadata(self) -> None:
        """Save run metadata to file."""
        metadata_file = self.output_dir / "run_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {
            "run_id": self.run_id,
            "run_type": self.run_type.value,
            "algorithm": self.algorithm,
            "family": self.family,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
            "config": self.config,
            "parameters": self.parameters,
            "output_dir": str(self.output_dir),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metrics": self.metrics,
            "system_info": self.system_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMetadata:
        """Create RunMetadata from dictionary.

        Args:
            data: Dictionary data

        Returns:
            RunMetadata instance
        """
        # Convert artifacts
        artifacts = []
        for artifact_data in data.get("artifacts", []):
            artifact = OutputArtifact(**artifact_data)
            artifacts.append(artifact)

        # Convert timestamps
        started_at = datetime.fromisoformat(data["started_at"])
        completed_at = None
        if data.get("completed_at"):
            completed_at = datetime.fromisoformat(data["completed_at"])

        return cls(
            run_id=data["run_id"],
            run_type=RunType(data["run_type"]),
            algorithm=data["algorithm"],
            family=data["family"],
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=data.get("duration_seconds"),
            status=RunStatus(data["status"]),
            exit_code=data.get("exit_code"),
            error_message=data.get("error_message"),
            config=data.get("config", {}),
            parameters=data.get("parameters", {}),
            output_dir=Path(data["output_dir"]),
            artifacts=artifacts,
            metrics=data.get("metrics", {}),
            system_info=data.get("system_info", {}),
        )

    @classmethod
    def from_file(cls, metadata_file: Path) -> RunMetadata:
        """Load run metadata from file.

        Args:
            metadata_file: Path to metadata file

        Returns:
            RunMetadata instance

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def create_run(cls, params: RunCreationParameters) -> RunMetadata:
        """Create a new run metadata instance.

        Args:
            params: Run creation parameters

        Returns:
            New RunMetadata instance
        """
        return cls(
            run_id=params.run_id,
            run_type=params.run_type,
            algorithm=params.algorithm,
            family=params.family,
            output_dir=params.output_dir,
            config=params.config or {},
            parameters=params.parameters or {},
            system_info=params.system_info or {},
        )
