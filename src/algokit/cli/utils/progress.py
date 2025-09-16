"""Progress tracking utilities with Rich integration for CLI operations."""

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from algokit.cli.models.config import Config
from algokit.cli.utils.logging import get_logger


class ProgressTracker:
    """Manages progress tracking with Rich integration for CLI operations.

    This class provides comprehensive progress tracking including multi-task
    progress support, progress persistence, time estimation, and status updates
    for different types of operations (training, validation, testing, etc.).

    Attributes:
        console: Rich console instance for progress display
        progress: Rich Progress instance for progress bars
        logger: Logger instance for progress tracking operations
        config: Configuration instance with progress settings
        active_tasks: Dictionary of active progress tasks
        progress_history: List of completed progress sessions
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the progress tracker with configuration.

        Args:
            config: Configuration instance with progress settings. If None,
                   uses default configuration.
        """
        self.config = config or Config()
        self.console = Console()
        self.logger = get_logger("progress_tracker")

        # Initialize progress tracking
        self.progress = self._create_progress_bar()
        self.active_tasks: dict[str, Any] = {}
        self.progress_history: list[dict[str, Any]] = []

        # Progress persistence
        self.progress_file = (
            Path(self.config.global_.output_dir) / "progress" / "current_progress.json"
        )
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

    def _create_progress_bar(self) -> Progress:
        """Create a Rich Progress instance with custom columns.

        Returns:
            Configured Progress instance with custom columns.
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

    @contextmanager
    def track_task(
        self,
        task_name: str,
        total: int = 100,
        description: str | None = None,
        task_type: str = "general",
    ):
        """Context manager for tracking a single task with progress.

        Args:
            task_name: Unique name for the task
            total: Total number of steps for the task
            description: Description to display for the task
            task_type: Type of task (training, validation, testing, etc.)

        Yields:
            Task ID for updating progress
        """
        if description is None:
            description = task_name

        # Start progress tracking
        with self.progress:
            task_id = self.progress.add_task(description=description, total=total)

            # Store task information
            self.active_tasks[task_name] = {
                "task_id": task_id,
                "task_type": task_type,
                "total": total,
                "start_time": datetime.now(),
                "description": description,
            }

            self.logger.info(f"Started progress tracking for task: {task_name}")

            try:
                yield task_id
            finally:
                # Clean up task
                if task_name in self.active_tasks:
                    task_info = self.active_tasks[task_name]
                    task_info["end_time"] = datetime.now()
                    task_info["duration"] = (
                        task_info["end_time"] - task_info["start_time"]
                    )

                    # Add to history
                    self.progress_history.append(task_info.copy())

                    # Remove from active tasks
                    del self.active_tasks[task_name]

                    self.logger.info(
                        f"Completed progress tracking for task: {task_name}"
                    )

    def update_progress(
        self, task_name: str, advance: int = 1, description: str | None = None
    ) -> None:
        """Update progress for a specific task.

        Args:
            task_name: Name of the task to update
            advance: Number of steps to advance
            description: Optional new description for the task
        """
        if task_name not in self.active_tasks:
            self.logger.warning(f"Task not found for progress update: {task_name}")
            return

        task_info = self.active_tasks[task_name]
        task_id = task_info["task_id"]

        # Update progress
        self.progress.update(
            task_id,
            advance=advance,
            description=description or task_info["description"],
        )

        # Update task info
        if description:
            task_info["description"] = description

    def set_progress(
        self, task_name: str, completed: int, description: str | None = None
    ) -> None:
        """Set absolute progress for a specific task.

        Args:
            task_name: Name of the task to update
            completed: Number of completed steps
            description: Optional new description for the task
        """
        if task_name not in self.active_tasks:
            self.logger.warning(f"Task not found for progress update: {task_name}")
            return

        task_info = self.active_tasks[task_name]
        task_id = task_info["task_id"]

        # Update progress
        self.progress.update(
            task_id,
            completed=completed,
            description=description or task_info["description"],
        )

        # Update task info
        if description:
            task_info["description"] = description

    def add_subtask(
        self,
        parent_task: str,
        subtask_name: str,
        total: int = 100,
        description: str | None = None,
    ) -> str:
        """Add a subtask to an existing parent task.

        Args:
            parent_task: Name of the parent task
            subtask_name: Name of the subtask
            total: Total number of steps for the subtask
            description: Description to display for the subtask

        Returns:
            Full subtask name for future updates
        """
        full_subtask_name = f"{parent_task}.{subtask_name}"

        if description is None:
            description = f"{parent_task} - {subtask_name}"

        # Add subtask to progress
        with self.progress:
            task_id = self.progress.add_task(description=description, total=total)

            # Store subtask information
            self.active_tasks[full_subtask_name] = {
                "task_id": task_id,
                "task_type": "subtask",
                "parent_task": parent_task,
                "subtask_name": subtask_name,
                "total": total,
                "start_time": datetime.now(),
                "description": description,
            }

        self.logger.debug(f"Added subtask: {full_subtask_name}")
        return full_subtask_name

    def remove_subtask(self, subtask_name: str) -> None:
        """Remove a subtask from progress tracking.

        Args:
            subtask_name: Full name of the subtask to remove
        """
        if subtask_name in self.active_tasks:
            task_info = self.active_tasks[subtask_name]
            task_info["end_time"] = datetime.now()
            task_info["duration"] = task_info["end_time"] - task_info["start_time"]

            # Add to history
            self.progress_history.append(task_info.copy())

            # Remove from active tasks
            del self.active_tasks[subtask_name]

            self.logger.debug(f"Removed subtask: {subtask_name}")

    def get_task_status(self, task_name: str) -> dict[str, Any] | None:
        """Get current status of a specific task.

        Args:
            task_name: Name of the task

        Returns:
            Dictionary with task status information, or None if not found
        """
        if task_name not in self.active_tasks:
            return None

        task_info = self.active_tasks[task_name]
        task_id = task_info["task_id"]

        # Get current progress from Rich Progress
        task = (
            self.progress.tasks[task_id] if task_id < len(self.progress.tasks) else None
        )

        status = {
            "task_name": task_name,
            "task_type": task_info["task_type"],
            "description": task_info["description"],
            "start_time": task_info["start_time"],
            "total": task_info["total"],
            "completed": task.completed if task else 0,
            "percentage": task.percentage if task else 0.0,
            "elapsed_time": datetime.now() - task_info["start_time"],
        }

        # Calculate estimated remaining time
        if task and task.completed > 0:
            elapsed_seconds = status["elapsed_time"].total_seconds()
            rate = task.completed / elapsed_seconds
            remaining_steps = task_info["total"] - task.completed
            estimated_remaining = (
                timedelta(seconds=remaining_steps / rate) if rate > 0 else timedelta(0)
            )
            status["estimated_remaining"] = estimated_remaining

        return status

    def get_all_task_status(self) -> list[dict[str, Any]]:
        """Get status of all active tasks.

        Returns:
            List of dictionaries with task status information
        """
        return [self.get_task_status(task_name) for task_name in self.active_tasks]

    def display_progress_summary(self) -> None:
        """Display a summary of current progress in a Rich table."""
        if not self.active_tasks:
            self.console.print("[yellow]No active progress tasks[/yellow]")
            return

        table = Table(title="Active Progress Tasks")
        table.add_column("Task", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Progress", style="green")
        table.add_column("Elapsed", style="blue")
        table.add_column("Remaining", style="red")

        for task_name in self.active_tasks:
            status = self.get_task_status(task_name)
            if status:
                table.add_row(
                    status["description"],
                    status["task_type"],
                    f"{status['completed']}/{status['total']} ({status['percentage']:.1f}%)",
                    str(status["elapsed_time"]).split(".")[0],  # Remove microseconds
                    str(status.get("estimated_remaining", "N/A")).split(".")[0],
                )

        self.console.print(table)

    def save_progress(self) -> None:
        """Save current progress to file for persistence."""
        progress_data = {
            "active_tasks": {
                name: {
                    "task_type": info["task_type"],
                    "total": info["total"],
                    "start_time": info["start_time"].isoformat(),
                    "description": info["description"],
                }
                for name, info in self.active_tasks.items()
            },
            "progress_history": [
                {
                    "task_name": task.get("task_name", "unknown"),
                    "task_type": task["task_type"],
                    "total": task["total"],
                    "start_time": task["start_time"].isoformat(),
                    "end_time": task["end_time"].isoformat(),
                    "duration": str(task["duration"]),
                    "description": task["description"],
                }
                for task in self.progress_history[-10:]  # Keep last 10 completed tasks
            ],
            "saved_at": datetime.now().isoformat(),
        }

        with open(self.progress_file, "w") as f:
            json.dump(progress_data, f, indent=2)

        self.logger.debug(f"Saved progress to: {self.progress_file}")

    def load_progress(self) -> None:
        """Load progress from file for persistence."""
        if not self.progress_file.exists():
            return

        try:
            with open(self.progress_file) as f:
                progress_data = json.load(f)

            # Restore progress history
            self.progress_history = []
            for task_data in progress_data.get("progress_history", []):
                self.progress_history.append(
                    {
                        "task_name": task_data.get("task_name", "unknown"),
                        "task_type": task_data["task_type"],
                        "total": task_data["total"],
                        "start_time": datetime.fromisoformat(task_data["start_time"]),
                        "end_time": datetime.fromisoformat(task_data["end_time"]),
                        "duration": timedelta(
                            seconds=float(task_data["duration"].split(":")[-1])
                        ),
                        "description": task_data["description"],
                    }
                )

            self.logger.debug(f"Loaded progress from: {self.progress_file}")

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to load progress: {e}")

    def clear_progress(self) -> None:
        """Clear all progress tracking data."""
        self.active_tasks.clear()
        self.progress_history.clear()

        # Clear progress file
        if self.progress_file.exists():
            self.progress_file.unlink()

        self.logger.info("Cleared all progress tracking data")

    def get_progress_statistics(self) -> dict[str, Any]:
        """Get comprehensive progress statistics.

        Returns:
            Dictionary with progress statistics and metrics.
        """
        total_tasks = len(self.progress_history)
        active_tasks = len(self.active_tasks)

        # Calculate average task duration
        completed_tasks = [task for task in self.progress_history if "duration" in task]
        avg_duration = None
        if completed_tasks:
            total_duration = sum(
                task["duration"].total_seconds() for task in completed_tasks
            )
            avg_duration = timedelta(seconds=total_duration / len(completed_tasks))

        # Group by task type
        task_types = {}
        for task in self.progress_history:
            task_type = task["task_type"]
            if task_type not in task_types:
                task_types[task_type] = 0
            task_types[task_type] += 1

        return {
            "total_completed_tasks": total_tasks,
            "active_tasks": active_tasks,
            "average_task_duration": str(avg_duration) if avg_duration else None,
            "task_types": task_types,
            "progress_file": str(self.progress_file),
            "progress_file_exists": self.progress_file.exists(),
        }


# Global progress tracker instance
_progress_tracker: ProgressTracker | None = None


def get_progress_tracker(config: Config | None = None) -> ProgressTracker:
    """Get the global progress tracker instance.

    Args:
        config: Configuration instance. If None, uses existing configuration.

    Returns:
        Global ProgressTracker instance.
    """
    global _progress_tracker  # noqa: PLW0603

    if _progress_tracker is None or config is not None:
        _progress_tracker = ProgressTracker(config)
        _progress_tracker.load_progress()

    return _progress_tracker


def track_progress(
    task_name: str,
    total: int = 100,
    description: str | None = None,
    task_type: str = "general",
):
    """Context manager for tracking progress of a single task.

    Args:
        task_name: Unique name for the task
        total: Total number of steps for the task
        description: Description to display for the task
        task_type: Type of task (training, validation, testing, etc.)

    Yields:
        Task ID for updating progress
    """
    tracker = get_progress_tracker()
    return tracker.track_task(task_name, total, description, task_type)


def update_progress(
    task_name: str, advance: int = 1, description: str | None = None
) -> None:
    """Update progress for a specific task.

    Args:
        task_name: Name of the task to update
        advance: Number of steps to advance
        description: Optional new description for the task
    """
    tracker = get_progress_tracker()
    tracker.update_progress(task_name, advance, description)


def set_progress(
    task_name: str, completed: int, description: str | None = None
) -> None:
    """Set absolute progress for a specific task.

    Args:
        task_name: Name of the task to update
        completed: Number of completed steps
        description: Optional new description for the task
    """
    tracker = get_progress_tracker()
    tracker.set_progress(task_name, completed, description)


def setup_progress_tracking(config: Config | None = None) -> ProgressTracker:
    """Set up progress tracking for the CLI application.

    Args:
        config: Configuration instance with progress settings.

    Returns:
        Configured ProgressTracker instance.
    """
    return get_progress_tracker(config)
