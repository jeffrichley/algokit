"""Algorithm execution engine for CLI operations.

This module provides the core execution engine that handles algorithm execution,
resource management, progress tracking, and result collection for all CLI operations.
"""

from __future__ import annotations

import asyncio
import builtins
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from algokit.cli.models.algorithm import Algorithm
from algokit.cli.models.config import Config
from algokit.cli.models.output import OutputArtifact, RunMetadata, RunStatus, RunType


class ExecutionMode(str, Enum):
    """Algorithm execution mode enumeration."""

    TRAIN = "train"
    REPLAY = "replay"
    DEMO = "demo"
    TEST = "test"
    BENCHMARK = "benchmark"


@dataclass
class ExecutionParameters:
    """Parameters for algorithm execution."""

    algorithm: Algorithm
    mode: ExecutionMode
    parameters: dict[str, Any] | None = None
    output_dir: Path | None = None
    timeout: timedelta | None = None
    progress_callback: Callable[[str, float], None] | None = None
    log_callback: Callable[[str, str], None] | None = None


class ExecutionResult(str, Enum):
    """Execution result enumeration."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass
class ExecutionContext:
    """Context for algorithm execution."""

    algorithm: Algorithm
    config: Config
    mode: ExecutionMode
    parameters: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None
    timeout: timedelta | None = None
    max_memory_mb: int | None = None
    progress_callback: Callable[[str, float], None] | None = None
    log_callback: Callable[[str, str], None] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    result: ExecutionResult = ExecutionResult.SUCCESS
    error_message: str | None = None
    artifacts: list[OutputArtifact] = field(default_factory=list)
    metadata: RunMetadata | None = None


class AlgorithmWrapper(Protocol):
    """Protocol for algorithm wrapper classes."""

    def train(self, **kwargs: Any) -> dict[str, Any]:
        """Train the algorithm."""
        ...

    def replay(self, **kwargs: Any) -> dict[str, Any]:
        """Replay the algorithm."""
        ...

    def demo(self, **kwargs: Any) -> dict[str, Any]:
        """Demo the algorithm."""
        ...

    def test(self, **kwargs: Any) -> dict[str, Any]:
        """Test the algorithm."""
        ...

    def benchmark(self, **kwargs: Any) -> dict[str, Any]:
        """Benchmark the algorithm."""
        ...


class ExecutionError(Exception):
    """Base exception for execution errors."""

    def __init__(self, message: str, context: ExecutionContext | None = None) -> None:
        """Initialize execution error.

        Args:
            message: Error message
            context: Execution context
        """
        super().__init__(message)
        self.context = context


class TimeoutError(ExecutionError):
    """Exception raised when execution times out."""

    pass


class ResourceError(ExecutionError):
    """Exception raised when resource limits are exceeded."""

    pass


class AlgorithmExecutor:
    """Base class for executing algorithms with proper resource management and error handling."""

    def __init__(self, config: Config) -> None:
        """Initialize the algorithm executor.

        Args:
            config: CLI configuration object
        """
        self.config = config
        self._active_executions: dict[str, ExecutionContext] = {}
        self._executor_pool: ThreadPoolExecutor | None = None
        self._shutdown_event = asyncio.Event()

    def __enter__(self) -> AlgorithmExecutor:
        """Enter the executor context."""
        self._executor_pool = ThreadPoolExecutor(
            max_workers=self.config.execution.max_workers,
            thread_name_prefix="algokit-executor",
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the executor context."""
        self._shutdown_event.set()
        if self._executor_pool:
            self._executor_pool.shutdown(wait=True)
            self._executor_pool = None

    async def execute_algorithm(self, params: ExecutionParameters) -> ExecutionContext:
        """Execute an algorithm with proper resource management and error handling.

        Args:
            params: Execution parameters

        Returns:
            Execution context with results and metadata

        Raises:
            ExecutionError: If execution fails
            TimeoutError: If execution times out
            ResourceError: If resource limits are exceeded
        """
        if params.parameters is None:
            params.parameters = {}

        if params.output_dir is None:
            params.output_dir = Path(self.config.global_.output_dir)

        if params.timeout is None:
            params.timeout = timedelta(seconds=self.config.execution.timeout_seconds)

        # Add output_dir to parameters so it gets passed to the algorithm
        params.parameters["output_dir"] = str(params.output_dir)

        # Create execution context
        context = ExecutionContext(
            algorithm=params.algorithm,
            config=self.config,
            mode=params.mode,
            parameters=params.parameters,
            output_dir=params.output_dir,
            timeout=params.timeout,
            max_memory_mb=self.config.execution.memory_limit_gb * 1024,
            progress_callback=params.progress_callback,
            log_callback=params.log_callback,
            start_time=datetime.now(),
        )

        # Generate unique execution ID
        execution_id = self._generate_execution_id(params.algorithm, params.mode)
        self._active_executions[execution_id] = context

        try:
            # Log execution start
            self._log_execution_start(context, execution_id)

            # Execute the algorithm
            await self._execute_with_timeout(context, params.timeout)

            # Mark as completed
            context.end_time = datetime.now()
            context.result = ExecutionResult.SUCCESS

            # Log execution completion
            self._log_execution_completion(context, execution_id)

        except builtins.TimeoutError:
            context.result = ExecutionResult.TIMEOUT
            context.end_time = datetime.now()
            context.error_message = f"Execution timed out after {params.timeout}"
            self._log_execution_timeout(context, execution_id)
            raise TimeoutError(context.error_message, context) from None

        except KeyboardInterrupt:
            context.result = ExecutionResult.INTERRUPTED
            context.end_time = datetime.now()
            context.error_message = "Execution interrupted by user"
            self._log_execution_interruption(context, execution_id)
            raise ExecutionError(context.error_message, context) from None

        except Exception as e:
            context.result = ExecutionResult.FAILURE
            context.end_time = datetime.now()
            context.error_message = str(e)
            self._log_execution_error(context, execution_id, e)
            raise ExecutionError(context.error_message, context) from e

        finally:
            # Clean up execution context
            self._cleanup_execution(execution_id)

        return context

    async def _execute_with_timeout(
        self, context: ExecutionContext, _timeout: timedelta
    ) -> None:
        """Execute algorithm with timeout protection.

        Args:
            context: Execution context
            timeout: Execution timeout
        """
        # Create the algorithm wrapper
        algorithm_wrapper = self._create_algorithm_wrapper(context.algorithm)

        # Execute based on mode
        if context.mode == ExecutionMode.TRAIN:
            result = await self._execute_train(algorithm_wrapper, context)
        elif context.mode == ExecutionMode.REPLAY:
            result = await self._execute_replay(algorithm_wrapper, context)
        elif context.mode == ExecutionMode.DEMO:
            result = await self._execute_demo(algorithm_wrapper, context)
        elif context.mode == ExecutionMode.TEST:
            result = await self._execute_test(algorithm_wrapper, context)
        elif context.mode == ExecutionMode.BENCHMARK:
            result = await self._execute_benchmark(algorithm_wrapper, context)
        else:
            raise ExecutionError(f"Unknown execution mode: {context.mode}")

        # Process results
        self._process_execution_results(context, result)

    async def _execute_train(
        self, algorithm_wrapper: AlgorithmWrapper, context: ExecutionContext
    ) -> dict[str, Any]:
        """Execute training mode.

        Args:
            algorithm_wrapper: Algorithm wrapper instance
            context: Execution context

        Returns:
            Training results
        """
        if context.log_callback:
            context.log_callback(
                "info", f"Starting training for {context.algorithm.name}"
            )

        # Execute training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def train_wrapper():
            return algorithm_wrapper.train(**context.parameters)

        result = await loop.run_in_executor(self._executor_pool, train_wrapper)

        if context.log_callback:
            context.log_callback(
                "info", f"Training completed for {context.algorithm.name}"
            )

        return result

    async def _execute_replay(
        self, algorithm_wrapper: AlgorithmWrapper, context: ExecutionContext
    ) -> dict[str, Any]:
        """Execute replay mode.

        Args:
            algorithm_wrapper: Algorithm wrapper instance
            context: Execution context

        Returns:
            Replay results
        """
        if context.log_callback:
            context.log_callback(
                "info", f"Starting replay for {context.algorithm.name}"
            )

        # Execute replay in thread pool
        loop = asyncio.get_event_loop()

        def replay_wrapper():
            return algorithm_wrapper.replay(**context.parameters)

        result = await loop.run_in_executor(self._executor_pool, replay_wrapper)

        if context.log_callback:
            context.log_callback(
                "info", f"Replay completed for {context.algorithm.name}"
            )

        return result

    async def _execute_demo(
        self, algorithm_wrapper: AlgorithmWrapper, context: ExecutionContext
    ) -> dict[str, Any]:
        """Execute demo mode.

        Args:
            algorithm_wrapper: Algorithm wrapper instance
            context: Execution context

        Returns:
            Demo results
        """
        if context.log_callback:
            context.log_callback("info", f"Starting demo for {context.algorithm.name}")

        # Execute demo in thread pool
        loop = asyncio.get_event_loop()

        def demo_wrapper():
            return algorithm_wrapper.demo(**context.parameters)

        result = await loop.run_in_executor(self._executor_pool, demo_wrapper)

        if context.log_callback:
            context.log_callback("info", f"Demo completed for {context.algorithm.name}")

        return result

    async def _execute_test(
        self, algorithm_wrapper: AlgorithmWrapper, context: ExecutionContext
    ) -> dict[str, Any]:
        """Execute test mode.

        Args:
            algorithm_wrapper: Algorithm wrapper instance
            context: Execution context

        Returns:
            Test results
        """
        if context.log_callback:
            context.log_callback("info", f"Starting test for {context.algorithm.name}")

        # Execute test in thread pool
        loop = asyncio.get_event_loop()

        def test_wrapper():
            return algorithm_wrapper.test(**context.parameters)

        result = await loop.run_in_executor(self._executor_pool, test_wrapper)

        if context.log_callback:
            context.log_callback("info", f"Test completed for {context.algorithm.name}")

        return result

    async def _execute_benchmark(
        self, algorithm_wrapper: AlgorithmWrapper, context: ExecutionContext
    ) -> dict[str, Any]:
        """Execute benchmark mode.

        Args:
            algorithm_wrapper: Algorithm wrapper instance
            context: Execution context

        Returns:
            Benchmark results
        """
        if context.log_callback:
            context.log_callback(
                "info", f"Starting benchmark for {context.algorithm.name}"
            )

        # Execute benchmark in thread pool
        loop = asyncio.get_event_loop()

        def benchmark_wrapper():
            return algorithm_wrapper.benchmark(**context.parameters)

        result = await loop.run_in_executor(self._executor_pool, benchmark_wrapper)

        if context.log_callback:
            context.log_callback(
                "info", f"Benchmark completed for {context.algorithm.name}"
            )

        return result

    def _create_algorithm_wrapper(self, algorithm: Algorithm) -> AlgorithmWrapper:
        """Create algorithm wrapper instance.

        Args:
            algorithm: Algorithm configuration

        Returns:
            Algorithm wrapper instance

        Raises:
            ExecutionError: If algorithm wrapper cannot be created
        """
        try:
            # Import the algorithm module dynamically
            module_path = (
                f"algokit.cli.algorithms.{algorithm.family_id}.{algorithm.slug}"
            )
            module = __import__(module_path, fromlist=[algorithm.name])

            # Get the algorithm class
            algorithm_class = getattr(module, algorithm.name)

            # Create instance
            return algorithm_class()

        except ImportError as e:
            raise ExecutionError(
                f"Could not import algorithm {algorithm.name}: {e}"
            ) from e
        except AttributeError as e:
            raise ExecutionError(
                f"Could not find algorithm class {algorithm.name}: {e}"
            ) from e
        except Exception as e:
            raise ExecutionError(f"Could not create algorithm wrapper: {e}") from e

    def _process_execution_results(
        self, context: ExecutionContext, result: dict[str, Any]
    ) -> None:
        """Process execution results and create artifacts.

        Args:
            context: Execution context
            result: Execution results
        """
        # Create run metadata
        context.metadata = RunMetadata(
            run_id=self._generate_execution_id(context.algorithm, context.mode),
            algorithm=context.algorithm.name,
            family=context.algorithm.family_id,
            run_type=RunType(context.mode.value),
            status=RunStatus.COMPLETED,
            started_at=context.start_time,
            completed_at=context.end_time,
            duration_seconds=(context.end_time - context.start_time).total_seconds()
            if context.end_time and context.start_time
            else 0,
            parameters=context.parameters,
            output_dir=Path(context.output_dir),
            metrics=result.get("metrics", {}),
            artifacts=[],
        )

        # Create output artifacts from results
        for artifact_type, artifact_data in result.get("artifacts", {}).items():
            artifact = OutputArtifact(
                artifact_id=f"{context.metadata.run_id}_{artifact_type}",
                artifact_type=artifact_type,
                file_path=str(
                    context.output_dir
                    / f"{artifact_type}.{artifact_data.get('format', 'json')}"
                ),
                size_bytes=len(str(artifact_data)),
                created_at=datetime.now(),
                metadata=artifact_data.get("metadata", {}),
                checksum=None,  # Will be calculated if needed
                run_id=context.metadata.run_id,
            )
            context.artifacts.append(artifact)
            context.metadata.artifacts.append(artifact.artifact_id)

    def _generate_execution_id(self, algorithm: Algorithm, mode: ExecutionMode) -> str:
        """Generate unique execution ID.

        Args:
            algorithm: Algorithm configuration
            mode: Execution mode

        Returns:
            Unique execution ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{algorithm.family_id}_{algorithm.slug}_{mode.value}"

    def _log_execution_start(
        self, context: ExecutionContext, execution_id: str
    ) -> None:
        """Log execution start.

        Args:
            context: Execution context
            execution_id: Execution ID
        """
        if context.log_callback:
            context.log_callback(
                "info",
                f"Starting execution {execution_id}: {context.algorithm.name} ({context.mode.value})",
            )

    def _log_execution_completion(
        self, context: ExecutionContext, execution_id: str
    ) -> None:
        """Log execution completion.

        Args:
            context: Execution context
            execution_id: Execution ID
        """
        if context.log_callback:
            duration = (
                (context.end_time - context.start_time).total_seconds()
                if context.end_time and context.start_time
                else 0
            )
            context.log_callback(
                "info",
                f"Completed execution {execution_id}: {context.algorithm.name} ({context.mode.value}) in {duration:.2f}s",
            )

    def _log_execution_timeout(
        self, context: ExecutionContext, execution_id: str
    ) -> None:
        """Log execution timeout.

        Args:
            context: Execution context
            execution_id: Execution ID
        """
        if context.log_callback:
            context.log_callback(
                "error",
                f"Timeout execution {execution_id}: {context.algorithm.name} ({context.mode.value})",
            )

    def _log_execution_interruption(
        self, context: ExecutionContext, execution_id: str
    ) -> None:
        """Log execution interruption.

        Args:
            context: Execution context
            execution_id: Execution ID
        """
        if context.log_callback:
            context.log_callback(
                "warning",
                f"Interrupted execution {execution_id}: {context.algorithm.name} ({context.mode.value})",
            )

    def _log_execution_error(
        self, context: ExecutionContext, execution_id: str, error: Exception
    ) -> None:
        """Log execution error.

        Args:
            context: Execution context
            execution_id: Execution ID
            error: Exception that occurred
        """
        if context.log_callback:
            context.log_callback(
                "error",
                f"Failed execution {execution_id}: {context.algorithm.name} ({context.mode.value}) - {error}",
            )
            context.log_callback("debug", traceback.format_exc())

    def _cleanup_execution(self, execution_id: str) -> None:
        """Clean up execution context.

        Args:
            execution_id: Execution ID to clean up
        """
        if execution_id in self._active_executions:
            del self._active_executions[execution_id]

    def get_active_executions(self) -> dict[str, ExecutionContext]:
        """Get currently active executions.

        Returns:
            Dictionary of active execution contexts
        """
        return self._active_executions.copy()

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution.

        Args:
            execution_id: Execution ID to cancel

        Returns:
            True if execution was cancelled, False if not found
        """
        if execution_id in self._active_executions:
            context = self._active_executions[execution_id]
            context.result = ExecutionResult.CANCELLED
            context.end_time = datetime.now()
            context.error_message = "Execution cancelled by user"
            return True
        return False

    def get_execution_status(self, execution_id: str) -> ExecutionContext | None:
        """Get execution status.

        Args:
            execution_id: Execution ID

        Returns:
            Execution context if found, None otherwise
        """
        return self._active_executions.get(execution_id)
