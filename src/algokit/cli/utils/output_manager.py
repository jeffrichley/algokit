"""Output directory management and artifact tracking for CLI operations."""

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from algokit.cli.models.config import Config
from algokit.cli.models.output import ArtifactType, OutputArtifact, RunMetadata
from algokit.cli.utils.logging import get_logger


@dataclass
class ArtifactTrackingParameters:
    """Parameters for tracking an artifact."""

    run_dir: Path
    artifact_path: str | Path
    artifact_type: str
    algorithm: str
    family: str
    run_id: str
    metadata: dict[str, Any] | None = None


class OutputManager:
    """Manages output directory structure and artifact tracking for CLI operations.

    This class provides comprehensive output management including directory creation,
    artifact tracking, cleanup utilities, and export/import functionality for
    sharing results and maintaining organized output.

    Attributes:
        config: Configuration instance with output settings
        output_dir: Base output directory path
        logger: Logger instance for output management operations
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the output manager with configuration.

        Args:
            config: Configuration instance with output settings. If None,
                   uses default configuration.
        """
        self.config = config or Config()
        self.output_dir = Path(self.config.global_.output_dir)
        self.logger = get_logger("output_manager")

        # Ensure output directory exists
        self._ensure_output_directory()

    def _ensure_output_directory(self) -> None:
        """Ensure the output directory and subdirectories exist."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            subdirs = [
                "config",
                "runs",
                "replays",
                "demos",
                "benchmarks",
                "exports",
                "logs",
            ]

            for subdir in subdirs:
                (self.output_dir / subdir).mkdir(exist_ok=True)

            self.logger.debug(
                f"Output directory structure ensured at: {self.output_dir}"
            )

        except OSError as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise

    def create_run_directory(
        self,
        family: str,
        algorithm: str,
        run_type: str = "run",
        run_id: str | None = None,
    ) -> Path:
        """Create a timestamped directory for a specific run.

        Args:
            family: Algorithm family name
            algorithm: Algorithm name
            run_type: Type of run (run, replay, demo, benchmark)
            run_id: Optional custom run ID. If None, generates timestamp-based ID.

        Returns:
            Path to the created run directory.
        """
        if run_id is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = f"{timestamp}_{family}_{algorithm}_{run_type}_001"

        # Determine base directory based on run type
        if run_type == "replay":
            base_dir = self.output_dir / "replays"
        elif run_type == "demo":
            base_dir = self.output_dir / "demos"
        elif run_type == "benchmark":
            base_dir = self.output_dir / "benchmarks"
        else:
            base_dir = self.output_dir / "runs"

        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different artifact types
        artifact_dirs = [
            "config",
            "logs",
            "models",
            "metrics",
            "plots",
            "videos",
            "artifacts",
        ]

        for artifact_dir in artifact_dirs:
            (run_dir / artifact_dir).mkdir(exist_ok=True)

        self.logger.info(f"Created {run_type} directory: {run_dir}")
        return run_dir

    def track_artifact(self, params: ArtifactTrackingParameters) -> OutputArtifact:
        """Track an artifact with metadata.

        Args:
            params: Artifact tracking parameters

        Returns:
            OutputArtifact instance with tracking information.
        """
        artifact_path = Path(params.artifact_path)

        # Ensure artifact path is relative to run directory
        if artifact_path.is_absolute():
            try:
                artifact_path = artifact_path.relative_to(params.run_dir)
            except ValueError:
                # If not relative to run_dir, copy to appropriate subdirectory
                artifact_type_dir = params.run_dir / params.artifact_type
                artifact_type_dir.mkdir(exist_ok=True)
                target_path = artifact_type_dir / artifact_path.name
                shutil.copy2(artifact_path, target_path)
                artifact_path = target_path.relative_to(params.run_dir)

        # Create artifact metadata
        artifact = OutputArtifact(
            name=artifact_path.name,
            type=ArtifactType(params.artifact_type),
            path=artifact_path,
            size_bytes=artifact_path.stat().st_size if artifact_path.exists() else 0,
            algorithm=params.algorithm,
            family=params.family,
            run_id=params.run_id,
            metadata=params.metadata or {},
        )

        # Save artifact metadata
        metadata_file = (
            params.run_dir / "artifacts" / f"{artifact_path.stem}_metadata.json"
        )
        metadata_file.parent.mkdir(exist_ok=True)

        with open(metadata_file, "w") as f:
            json.dump(artifact.model_dump(), f, indent=2, default=str)

        self.logger.debug(f"Tracked artifact: {artifact_path} ({params.artifact_type})")
        return artifact

    def save_run_metadata(self, run_dir: Path, metadata: RunMetadata) -> None:
        """Save run metadata to the run directory.

        Args:
            run_dir: Run directory path
            metadata: RunMetadata instance to save
        """
        metadata_file = run_dir / "config" / "run_metadata.json"

        with open(metadata_file, "w") as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)

        self.logger.debug(f"Saved run metadata: {metadata_file}")

    def load_run_metadata(self, run_dir: Path) -> RunMetadata | None:
        """Load run metadata from the run directory.

        Args:
            run_dir: Run directory path

        Returns:
            RunMetadata instance if found, None otherwise.
        """
        metadata_file = run_dir / "config" / "run_metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file) as f:
                data = json.load(f)
            return RunMetadata.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to load run metadata: {e}")
            return None

    def list_runs(
        self,
        family: str | None = None,
        algorithm: str | None = None,
        run_type: str | None = None,
    ) -> list[Path]:
        """List all run directories matching the specified criteria.

        Args:
            family: Filter by algorithm family
            algorithm: Filter by algorithm name
            run_type: Filter by run type (run, replay, demo, benchmark)

        Returns:
            List of run directory paths matching the criteria.
        """
        runs = []

        # Search in all run type directories
        search_dirs = ["runs", "replays", "demos", "benchmarks"]
        if run_type:
            search_dirs = [run_type]

        for search_dir in search_dirs:
            base_dir = self.output_dir / search_dir
            if not base_dir.exists():
                continue

            for run_dir in base_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                # Parse run directory name: timestamp_family_algorithm_type_id
                parts = run_dir.name.split("_")
                if len(parts) < 4:
                    continue

                run_family = parts[1] if len(parts) > 1 else None
                run_algorithm = parts[2] if len(parts) > 2 else None
                run_type_name = parts[3] if len(parts) > 3 else None

                # Apply filters
                if family and run_family != family:
                    continue
                if algorithm and run_algorithm != algorithm:
                    continue
                if run_type and run_type_name != run_type:
                    continue

                runs.append(run_dir)

        # Sort by creation time (newest first)
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return runs

    def get_run_info(self, run_dir: Path) -> dict[str, Any]:
        """Get comprehensive information about a run directory.

        Args:
            run_dir: Run directory path

        Returns:
            Dictionary with run information and statistics.
        """
        if not run_dir.exists():
            return {}

        # Load metadata
        metadata = self.load_run_metadata(run_dir)

        # Calculate directory size
        total_size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())

        # Count artifacts by type
        artifact_counts = {}
        for artifact_dir in [
            "models",
            "logs",
            "metrics",
            "plots",
            "videos",
            "artifacts",
        ]:
            artifact_path = run_dir / artifact_dir
            if artifact_path.exists():
                artifact_counts[artifact_dir] = len(list(artifact_path.iterdir()))

        info = {
            "run_directory": str(run_dir),
            "run_name": run_dir.name,
            "created_at": datetime.fromtimestamp(run_dir.stat().st_mtime),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "artifact_counts": artifact_counts,
            "metadata": metadata.model_dump() if metadata else None,
        }

        return info

    def cleanup_old_runs(
        self,
        days_to_keep: int = 30,
        family: str | None = None,
        algorithm: str | None = None,
    ) -> list[Path]:
        """Clean up old run directories based on retention policy.

        Args:
            days_to_keep: Number of days to keep run directories
            family: Only clean up runs for specific family
            algorithm: Only clean up runs for specific algorithm

        Returns:
            List of cleaned up run directory paths.
        """
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        cleaned_runs = []

        runs = self.list_runs(family=family, algorithm=algorithm)

        for run_dir in runs:
            if run_dir.stat().st_mtime < cutoff_time:
                try:
                    shutil.rmtree(run_dir)
                    cleaned_runs.append(run_dir)
                    self.logger.info(f"Cleaned up old run: {run_dir.name}")
                except OSError as e:
                    self.logger.warning(f"Failed to clean up run {run_dir.name}: {e}")

        return cleaned_runs

    def export_run(
        self, run_dir: Path, export_path: Path, include_artifacts: bool = True
    ) -> None:
        """Export a run directory to a compressed archive.

        Args:
            run_dir: Run directory to export
            export_path: Path for the exported archive
            include_artifacts: Whether to include all artifacts in the export
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if include_artifacts:
            # Export entire directory
            shutil.make_archive(
                str(export_path.with_suffix("")), "zip", run_dir.parent, run_dir.name
            )
        else:
            # Export only metadata and configuration
            temp_dir = Path(f"/tmp/algokit_export_{run_dir.name}")
            temp_dir.mkdir(exist_ok=True)

            try:
                # Copy only config and metadata
                if (run_dir / "config").exists():
                    shutil.copytree(run_dir / "config", temp_dir / "config")

                shutil.make_archive(
                    str(export_path.with_suffix("")),
                    "zip",
                    temp_dir.parent,
                    temp_dir.name,
                )
            finally:
                shutil.rmtree(temp_dir)

        self.logger.info(f"Exported run to: {export_path}")

    def import_run(self, archive_path: Path, target_family: str | None = None) -> Path:
        """Import a run from a compressed archive.

        Args:
            archive_path: Path to the archive file
            target_family: Optional target family for the imported run

        Returns:
            Path to the imported run directory.
        """
        archive_path = Path(archive_path)

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        # Extract to temporary directory
        temp_dir = Path(f"/tmp/algokit_import_{archive_path.stem}")
        temp_dir.mkdir(exist_ok=True)

        try:
            shutil.unpack_archive(archive_path, temp_dir)

            # Find the extracted run directory
            extracted_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
            if not extracted_dirs:
                raise ValueError("No run directory found in archive")

            source_run_dir = extracted_dirs[0]

            # Determine target directory
            if target_family:
                # Modify run name to include target family
                run_name = source_run_dir.name
                parts = run_name.split("_")
                if len(parts) >= 2:
                    parts[1] = target_family
                    run_name = "_".join(parts)
            else:
                run_name = source_run_dir.name

            # Determine target base directory from run type
            if "replay" in run_name:
                target_base = self.output_dir / "replays"
            elif "demo" in run_name:
                target_base = self.output_dir / "demos"
            elif "benchmark" in run_name:
                target_base = self.output_dir / "benchmarks"
            else:
                target_base = self.output_dir / "runs"

            target_run_dir = target_base / run_name

            # Move to target location
            shutil.move(str(source_run_dir), str(target_run_dir))

            self.logger.info(f"Imported run to: {target_run_dir}")
            return target_run_dir

        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def get_disk_usage(self) -> dict[str, Any]:
        """Get disk usage statistics for the output directory.

        Returns:
            Dictionary with disk usage information.
        """
        if not self.output_dir.exists():
            return {"total_size_bytes": 0, "total_size_mb": 0}

        total_size = sum(
            f.stat().st_size for f in self.output_dir.rglob("*") if f.is_file()
        )

        # Get usage by subdirectory
        usage_by_dir = {}
        for subdir in ["runs", "replays", "demos", "benchmarks", "exports", "logs"]:
            subdir_path = self.output_dir / subdir
            if subdir_path.exists():
                subdir_size = sum(
                    f.stat().st_size for f in subdir_path.rglob("*") if f.is_file()
                )
                usage_by_dir[subdir] = {
                    "size_bytes": subdir_size,
                    "size_mb": round(subdir_size / (1024 * 1024), 2),
                }

        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "usage_by_directory": usage_by_dir,
        }

    def optimize_storage(self, max_size_mb: int = 1000) -> list[Path]:
        """Optimize storage by cleaning up old runs when size limit is exceeded.

        Args:
            max_size_mb: Maximum allowed size in MB

        Returns:
            List of cleaned up run directory paths.
        """
        usage = self.get_disk_usage()
        current_size_mb = usage["total_size_mb"]

        if current_size_mb <= max_size_mb:
            return []

        # Calculate how much to clean up
        excess_mb = current_size_mb - max_size_mb

        # Get all runs sorted by age (oldest first)
        all_runs = self.list_runs()
        all_runs.sort(key=lambda p: p.stat().st_mtime)  # Oldest first

        cleaned_runs = []
        cleaned_size_mb = 0

        for run_dir in all_runs:
            if cleaned_size_mb >= excess_mb:
                break

            try:
                run_size = sum(
                    f.stat().st_size for f in run_dir.rglob("*") if f.is_file()
                )
                run_size_mb = run_size / (1024 * 1024)

                shutil.rmtree(run_dir)
                cleaned_runs.append(run_dir)
                cleaned_size_mb += int(run_size_mb)

                self.logger.info(
                    f"Optimized storage by removing: {run_dir.name} ({run_size_mb:.2f} MB)"
                )

            except OSError as e:
                self.logger.warning(f"Failed to remove run {run_dir.name}: {e}")

        return cleaned_runs


# Global output manager instance
_output_manager: OutputManager | None = None


def get_output_manager(config: Config | None = None) -> OutputManager:
    """Get the global output manager instance.

    Args:
        config: Configuration instance. If None, uses existing configuration.

    Returns:
        Global OutputManager instance.
    """
    global _output_manager

    if _output_manager is None or config is not None:
        _output_manager = OutputManager(config)

    return _output_manager


def setup_output_management(config: Config | None = None) -> OutputManager:
    """Set up output management for the CLI application.

    Args:
        config: Configuration instance with output settings.

    Returns:
        Configured OutputManager instance.
    """
    return get_output_manager(config)
