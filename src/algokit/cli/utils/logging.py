"""Rich logging configuration and management for CLI operations."""

import logging
import logging.handlers
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

from algokit.cli.models.config import Config


class LoggingManager:
    """Manages Rich logging configuration and console output for CLI operations.

    This class provides a centralized logging system that integrates with Rich
    for beautiful console output, file logging with rotation, and structured
    logging for debugging and analysis.

    Attributes:
        console: Rich console instance for formatted output
        logger: Main logger instance for CLI operations
        file_logger: File logger instance for persistent logging
        config: Configuration instance for logging settings
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the logging manager with configuration.

        Args:
            config: Configuration instance with logging settings. If None,
                   uses default configuration.
        """
        self.config = config or Config()
        self.console = Console()

        # Install Rich traceback handler for better error display
        install(show_locals=True)

        # Initialize loggers
        self.logger = self._setup_main_logger()
        self.file_logger = self._setup_file_logger()

        # Set log level from configuration
        self.set_log_level(self.config.global_.log_level)

    def _setup_main_logger(self) -> logging.Logger:
        """Set up the main console logger with Rich formatting.

        Returns:
            Configured logger instance for console output.
        """
        logger = logging.getLogger("algokit.cli")
        logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create Rich handler for console output
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )

        # Set format for Rich handler
        rich_handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%X]"))

        logger.addHandler(rich_handler)
        return logger

    def _setup_file_logger(self) -> logging.Logger:
        """Set up the file logger with rotation and retention.

        Returns:
            Configured logger instance for file output.
        """
        logger = logging.getLogger("algokit.cli.file")
        logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create logs directory if it doesn't exist
        log_dir = Path(self.config.global_.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        log_file = log_dir / "algokit.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )

        # Set detailed format for file logging
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)
        return logger

    def set_log_level(self, level: str | int) -> None:
        """Set the logging level for both console and file loggers.

        Args:
            level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  or integer (10, 20, 30, 40, 50).
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        self.logger.setLevel(level)
        self.file_logger.setLevel(level)

        # Update configuration
        self.config.global_.log_level = logging.getLevelName(level)

    def get_logger(self, name: str | None = None) -> logging.Logger:
        """Get a logger instance for a specific module or component.

        Args:
            name: Logger name. If None, returns the main CLI logger.

        Returns:
            Logger instance configured with Rich formatting.
        """
        if name is None:
            return self.logger

        # Create child logger
        child_logger = logging.getLogger(f"algokit.cli.{name}")
        child_logger.setLevel(self.logger.level)

        # Add handlers if not already present
        if not child_logger.handlers:
            for handler in self.logger.handlers:
                child_logger.addHandler(handler)

        return child_logger

    def log_structured(self, level: str, message: str, **kwargs: Any) -> None:
        """Log a structured message with additional context.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            **kwargs: Additional context data to include in the log
        """
        log_level = getattr(logging, level.upper(), logging.INFO)

        # Format structured message
        if kwargs:
            context_str = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            structured_message = f"{message} | {context_str}"
        else:
            structured_message = message

        # Log to both console and file
        self.logger.log(log_level, structured_message)
        self.file_logger.log(log_level, structured_message)

    def log_performance(self, operation: str, duration: float, **metrics: Any) -> None:
        """Log performance metrics for an operation.

        Args:
            operation: Name of the operation being measured
            duration: Duration in seconds
            **metrics: Additional performance metrics
        """
        self.log_structured(
            "INFO", f"Performance: {operation}", duration=f"{duration:.3f}s", **metrics
        )

    def log_error_with_context(self, error: Exception, context: dict[str, Any]) -> None:
        """Log an error with additional context information.

        Args:
            error: Exception that occurred
            context: Additional context information
        """
        self.log_structured(
            "ERROR", f"Error: {type(error).__name__}: {str(error)}", **context
        )

        # Log full traceback to file
        self.file_logger.exception("Full traceback:")

    def create_progress_logger(self, task_name: str) -> logging.Logger:
        """Create a logger specifically for progress tracking.

        Args:
            task_name: Name of the task being tracked

        Returns:
            Logger instance configured for progress logging.
        """
        progress_logger = self.get_logger(f"progress.{task_name}")
        return progress_logger

    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """Clean up old log files based on retention policy.

        Args:
            days_to_keep: Number of days to keep log files
        """
        log_dir = Path(self.config.global_.output_dir) / "logs"
        if not log_dir.exists():
            return

        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)

        for log_file in log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.logger.info(f"Cleaned up old log file: {log_file.name}")
                except OSError as e:
                    self.logger.warning(
                        f"Failed to clean up log file {log_file.name}: {e}"
                    )

    def get_log_stats(self) -> dict[str, Any]:
        """Get statistics about current logging configuration.

        Returns:
            Dictionary with logging statistics and configuration.
        """
        log_dir = Path(self.config.global_.output_dir) / "logs"

        stats = {
            "log_level": logging.getLevelName(self.logger.level),
            "console_handler": len(self.logger.handlers),
            "file_handler": len(self.file_logger.handlers),
            "log_directory": str(log_dir),
            "log_directory_exists": log_dir.exists(),
        }

        if log_dir.exists():
            log_files = list(log_dir.glob("*.log*"))
            stats.update(
                {
                    "log_files_count": len(log_files),
                    "log_files": [f.name for f in log_files],
                    "total_log_size": sum(f.stat().st_size for f in log_files),
                }
            )

        return stats


# Global logging manager instance
_logging_manager: LoggingManager | None = None


def get_logging_manager(config: Config | None = None) -> LoggingManager:
    """Get the global logging manager instance.

    Args:
        config: Configuration instance. If None, uses existing configuration.

    Returns:
        Global LoggingManager instance.
    """
    global _logging_manager  # noqa: PLW0603

    if _logging_manager is None or config is not None:
        _logging_manager = LoggingManager(config)

    return _logging_manager


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance for the specified name.

    Args:
        name: Logger name. If None, returns the main CLI logger.

    Returns:
        Logger instance configured with Rich formatting.
    """
    return get_logging_manager().get_logger(name)


def setup_logging(config: Config | None = None) -> LoggingManager:
    """Set up logging for the CLI application.

    Args:
        config: Configuration instance with logging settings.

    Returns:
        Configured LoggingManager instance.
    """
    return get_logging_manager(config)


def cleanup_logging() -> None:
    """Clean up logging resources and handlers."""
    global _logging_manager  # noqa: PLW0603

    if _logging_manager is not None:
        # Close all handlers
        for logger in [_logging_manager.logger, _logging_manager.file_logger]:
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)

        _logging_manager = None
