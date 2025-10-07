"""ParameterDefinition validation system for CLI operations.

This module provides comprehensive parameter validation for algorithm parameters,
environment compatibility, and input sanitization for all CLI operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import gymnasium as gym

from algokit.cli.models.algorithm import Algorithm, ParameterDefinition, ParameterType
from algokit.cli.models.config import Config


class ValidationSeverity(str, Enum):
    """Validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of parameter validation."""

    is_valid: bool
    severity: ValidationSeverity
    message: str
    parameter_name: str | None = None
    suggested_value: Any | None = None
    error_code: str | None = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    is_valid: bool
    results: list[ValidationResult]
    warnings: list[ValidationResult]
    errors: list[ValidationResult]
    critical_errors: list[ValidationResult]

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        self.warnings = [
            r for r in self.results if r.severity == ValidationSeverity.WARNING
        ]
        self.errors = [
            r for r in self.results if r.severity == ValidationSeverity.ERROR
        ]
        self.critical_errors = [
            r for r in self.results if r.severity == ValidationSeverity.CRITICAL
        ]
        self.is_valid = len(self.errors) == 0 and len(self.critical_errors) == 0


class ParameterDefinitionValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        """Initialize parameter validation error.

        Args:
            message: Error message
            report: Validation report
        """
        super().__init__(message)
        self.report = report


class ParameterDefinitionValidator:
    """Comprehensive parameter validation system for algorithm parameters and environment compatibility."""

    def __init__(self, config: Config) -> None:
        """Initialize the parameter validator.

        Args:
            config: CLI configuration object
        """
        self.config = config
        self._supported_environments: set[str] = set()
        self._environment_cache: dict[str, Any] = {}
        self._parameter_cache: dict[str, dict[str, Any]] = {}

        # Initialize supported environments
        self._initialize_supported_environments()

    def validate_algorithm_parameters(
        self, algorithm: Algorithm, parameters: dict[str, Any], mode: str = "train"
    ) -> ValidationReport:
        """Validate algorithm parameters against the algorithm specification.

        Args:
            algorithm: Algorithm configuration
            parameters: ParameterDefinitions to validate
            mode: Execution mode (train, replay, demo, test, benchmark)

        Returns:
            Validation report with results and recommendations
        """
        results: list[ValidationResult] = []

        # Validate required parameters
        results.extend(self._validate_required_parameters(algorithm, parameters, mode))

        # Validate parameter types and ranges
        results.extend(self._validate_parameter_types_and_ranges(algorithm, parameters))

        # Validate parameter dependencies
        results.extend(self._validate_parameter_dependencies(algorithm, parameters))

        # Validate mode-specific parameters
        results.extend(
            self._validate_mode_specific_parameters(algorithm, parameters, mode)
        )

        # Validate environment compatibility
        if "env" in parameters:
            results.extend(
                self._validate_environment_compatibility(algorithm, parameters["env"])
            )

        # Algorithm-specific parameter validation will be added here as algorithms are implemented

        return ValidationReport(
            is_valid=True,  # Will be updated in __post_init__
            results=results,
            warnings=[],
            errors=[],
            critical_errors=[],
        )

    def _validate_required_parameters(
        self, algorithm: Algorithm, parameters: dict[str, Any], mode: str
    ) -> list[ValidationResult]:
        """Validate that all required parameters are provided.

        Args:
            algorithm: Algorithm configuration
            parameters: ParameterDefinitions to validate
            mode: Execution mode

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        # Get required parameters for the mode
        required_params = self._get_required_parameters(algorithm, mode)

        results.extend(
            [
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required parameter '{param_name}' is missing for mode '{mode}'",
                    parameter_name=param_name,
                    error_code="MISSING_REQUIRED_PARAMETER",
                )
                for param_name in required_params
                if param_name not in parameters
            ]
        )

        return results

    def _validate_parameter_types_and_ranges(
        self, algorithm: Algorithm, parameters: dict[str, Any]
    ) -> list[ValidationResult]:
        """Validate parameter types and ranges.

        Args:
            algorithm: Algorithm configuration
            parameters: ParameterDefinitions to validate

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        for param_name, param_value in parameters.items():
            # Get parameter specification
            param_spec = self._get_parameter_specification(algorithm, param_name)

            if param_spec is None:
                # Unknown parameter - this might be a warning or error depending on strictness
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Unknown parameter '{param_name}' for algorithm '{algorithm.name}'",
                        parameter_name=param_name,
                        error_code="UNKNOWN_PARAMETER",
                    )
                )
                continue

            # Validate type
            type_result = self._validate_parameter_type(
                param_name, param_value, param_spec
            )
            if type_result:
                results.append(type_result)

            # Validate range/constraints
            range_result = self._validate_parameter_range(
                param_name, param_value, param_spec
            )
            if range_result:
                results.append(range_result)

        return results

    def _validate_parameter_dependencies(
        self, algorithm: Algorithm, parameters: dict[str, Any]
    ) -> list[ValidationResult]:
        """Validate parameter dependencies and constraints.

        Args:
            algorithm: Algorithm configuration
            parameters: ParameterDefinitions to validate

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        # Algorithm-specific dependency validation will be added here as algorithms are implemented

        # Validate general algorithm dependencies
        results.extend(self._validate_general_dependencies(algorithm, parameters))

        return results

    def _validate_mode_specific_parameters(
        self, _algorithm: Algorithm, parameters: dict[str, Any], mode: str
    ) -> list[ValidationResult]:
        """Validate mode-specific parameters.

        Args:
            algorithm: Algorithm configuration
            parameters: ParameterDefinitions to validate
            mode: Execution mode

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        if mode == "replay":
            # Replay mode requires model path
            if "model_path" not in parameters:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="Replay mode requires 'model_path' parameter",
                        parameter_name="model_path",
                        error_code="MISSING_MODEL_PATH",
                    )
                )

        elif (
            mode == "demo"
            and "interactive" in parameters
            and not isinstance(parameters["interactive"], bool)
        ):
            # Demo mode might have different requirements
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="'interactive' parameter must be a boolean",
                    parameter_name="interactive",
                    error_code="INVALID_PARAMETER_TYPE",
                )
            )

        return results

    def _validate_environment_compatibility(
        self, algorithm: Algorithm, env_name: str
    ) -> list[ValidationResult]:
        """Validate environment compatibility with algorithm.

        Args:
            algorithm: Algorithm configuration
            env_name: Environment name

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        try:
            # Check if environment exists
            if env_name not in self._supported_environments:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Environment '{env_name}' is not supported or not installed",
                        parameter_name="env",
                        error_code="UNSUPPORTED_ENVIRONMENT",
                    )
                )
                return results

            # Get environment info
            env_info = self._get_environment_info(env_name)

            # Validate algorithm compatibility
            if algorithm.family_id == "rl":
                results.extend(
                    self._validate_rl_environment_compatibility(algorithm, env_info)
                )

        except Exception as e:
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Error validating environment '{env_name}': {e}",
                    parameter_name="env",
                    error_code="ENVIRONMENT_VALIDATION_ERROR",
                )
            )

        return results

    def _validate_learning_rate(
        self, parameters: dict[str, Any]
    ) -> list[ValidationResult]:
        """Validate learning rate parameter.

        Args:
            parameters: ParameterDefinitions to validate

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []
        if "learning_rate" in parameters:
            lr = parameters["learning_rate"]
            if not isinstance(lr, int | float) or lr <= 0 or lr > 1:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="Learning rate must be a number between 0 and 1",
                        parameter_name="learning_rate",
                        suggested_value=0.1,
                        error_code="INVALID_LEARNING_RATE",
                    )
                )
        return results

    def _validate_discount_factor(
        self, parameters: dict[str, Any]
    ) -> list[ValidationResult]:
        """Validate discount factor parameter.

        Args:
            parameters: ParameterDefinitions to validate

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []
        if "gamma" in parameters:
            gamma = parameters["gamma"]
            if not isinstance(gamma, int | float) or gamma < 0 or gamma > 1:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="Discount factor (gamma) must be a number between 0 and 1",
                        parameter_name="gamma",
                        suggested_value=0.9,
                        error_code="INVALID_DISCOUNT_FACTOR",
                    )
                )
        return results

    def _validate_epsilon_parameters(
        self, parameters: dict[str, Any]
    ) -> list[ValidationResult]:
        """Validate epsilon and epsilon decay parameters.

        Args:
            parameters: ParameterDefinitions to validate

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        # Validate epsilon
        if "epsilon" in parameters:
            epsilon = parameters["epsilon"]
            if not isinstance(epsilon, int | float) or epsilon < 0 or epsilon > 1:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="Epsilon must be a number between 0 and 1",
                        parameter_name="epsilon",
                        suggested_value=0.1,
                        error_code="INVALID_EPSILON",
                    )
                )

        # Validate epsilon decay
        if "epsilon_decay" in parameters:
            epsilon_decay = parameters["epsilon_decay"]
            if (
                not isinstance(epsilon_decay, int | float)
                or epsilon_decay <= 0
                or epsilon_decay > 1
            ):
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="Epsilon decay must be a number between 0 and 1",
                        parameter_name="epsilon_decay",
                        suggested_value=0.995,
                        error_code="INVALID_EPSILON_DECAY",
                    )
                )

        return results

    def _validate_episodes(self, parameters: dict[str, Any]) -> list[ValidationResult]:
        """Validate episodes parameter.

        Args:
            parameters: ParameterDefinitions to validate

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []
        if "episodes" in parameters:
            episodes = parameters["episodes"]
            if not isinstance(episodes, int) or episodes <= 0:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="Episodes must be a positive integer",
                        parameter_name="episodes",
                        suggested_value=1000,
                        error_code="INVALID_EPISODES",
                    )
                )
        return results

    # SARSA-specific validation methods removed - no longer implemented

    def _validate_general_dependencies(
        self, _algorithm: Algorithm, _parameters: dict[str, Any]
    ) -> list[ValidationResult]:
        """Validate general algorithm parameter dependencies.

        Args:
            algorithm: Algorithm configuration
            parameters: ParameterDefinitions to validate

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        # Add general dependency validation logic here
        # This can be extended for other algorithms

        return results

    def _validate_parameter_type(
        self, param_name: str, param_value: Any, param_spec: ParameterDefinition
    ) -> ValidationResult | None:
        """Validate parameter type.

        Args:
            param_name: ParameterDefinition name
            param_value: ParameterDefinition value
            param_spec: ParameterDefinition specification

        Returns:
            Validation result if type is invalid, None otherwise
        """
        expected_type = param_spec.type

        # Type validation logic
        if expected_type == "float" and not isinstance(param_value, int | float):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"ParameterDefinition '{param_name}' must be a number",
                parameter_name=param_name,
                error_code="INVALID_PARAMETER_TYPE",
            )

        elif expected_type == "int" and not isinstance(param_value, int):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"ParameterDefinition '{param_name}' must be an integer",
                parameter_name=param_name,
                error_code="INVALID_PARAMETER_TYPE",
            )

        elif expected_type == "str" and not isinstance(param_value, str):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"ParameterDefinition '{param_name}' must be a string",
                parameter_name=param_name,
                error_code="INVALID_PARAMETER_TYPE",
            )

        elif expected_type == "bool" and not isinstance(param_value, bool):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"ParameterDefinition '{param_name}' must be a boolean",
                parameter_name=param_name,
                error_code="INVALID_PARAMETER_TYPE",
            )

        return None

    def _validate_parameter_range(
        self, param_name: str, param_value: Any, param_spec: ParameterDefinition
    ) -> ValidationResult | None:
        """Validate parameter range and constraints.

        Args:
            param_name: ParameterDefinition name
            param_value: ParameterDefinition value
            param_spec: ParameterDefinition specification

        Returns:
            Validation result if range is invalid, None otherwise
        """
        # Check minimum value
        if param_spec.min_value is not None and param_value < param_spec.min_value:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"ParameterDefinition '{param_name}' must be >= {param_spec.min_value}",
                parameter_name=param_name,
                suggested_value=param_spec.min_value,
                error_code="PARAMETER_BELOW_MINIMUM",
            )

        # Check maximum value
        if param_spec.max_value is not None and param_value > param_spec.max_value:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"ParameterDefinition '{param_name}' must be <= {param_spec.max_value}",
                parameter_name=param_name,
                suggested_value=param_spec.max_value,
                error_code="PARAMETER_ABOVE_MAXIMUM",
            )

        # Check allowed values
        if param_spec.choices and param_value not in param_spec.choices:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"ParameterDefinition '{param_name}' must be one of {param_spec.choices}",
                parameter_name=param_name,
                suggested_value=param_spec.choices[0],
                error_code="PARAMETER_NOT_ALLOWED",
            )

        return None

    def _get_required_parameters(self, algorithm: Algorithm, mode: str) -> list[str]:
        """Get required parameters for a specific mode.

        Args:
            algorithm: Algorithm configuration
            mode: Execution mode

        Returns:
            List of required parameter names
        """
        required_params = []

        # Base required parameters
        if mode == "train":
            required_params.extend(["env", "episodes"])
        elif mode == "replay":
            required_params.extend(["model_path"])
        elif mode == "demo":
            required_params.extend(["env"])

        # Algorithm-specific required parameters will be added here as algorithms are implemented

        return required_params

    def _get_parameter_specification(
        self, _algorithm: Algorithm, param_name: str
    ) -> ParameterDefinition | None:
        """Get parameter specification from algorithm configuration.

        Args:
            algorithm: Algorithm configuration
            param_name: ParameterDefinition name

        Returns:
            ParameterDefinition specification if found, None otherwise
        """
        # This would typically come from the algorithm's parameter definitions
        # For now, we'll use a simple mapping based on common parameters

        common_params = {
            "env": ParameterDefinition(
                name="env",
                type=ParameterType.STRING,
                description="Environment name",
                required=True,
                default="CartPole-v1",
            ),
            "episodes": ParameterDefinition(
                name="episodes",
                type=ParameterType.INTEGER,
                description="Number of training episodes",
                required=True,
                default=1000,
                min_value=1,
                max_value=1000000,
            ),
            "learning_rate": ParameterDefinition(
                name="learning_rate",
                type=ParameterType.FLOAT,
                description="Learning rate (alpha)",
                required=True,
                default=0.1,
                min_value=0.0,
                max_value=1.0,
            ),
            "gamma": ParameterDefinition(
                name="gamma",
                type=ParameterType.FLOAT,
                description="Discount factor",
                required=True,
                default=0.9,
                min_value=0.0,
                max_value=1.0,
            ),
            "epsilon": ParameterDefinition(
                name="epsilon",
                type=ParameterType.FLOAT,
                description="Exploration rate",
                required=True,
                default=0.1,
                min_value=0.0,
                max_value=1.0,
            ),
            "epsilon_decay": ParameterDefinition(
                name="epsilon_decay",
                type=ParameterType.FLOAT,
                description="Epsilon decay rate",
                required=False,
                default=0.995,
                min_value=0.0,
                max_value=1.0,
            ),
        }

        return common_params.get(param_name)

    def _initialize_supported_environments(self) -> None:
        """Initialize the list of supported environments."""
        try:
            # Get all registered environments from gymnasium
            # Use getattr to safely access registry attribute
            registry = getattr(gym.envs, "registry", None)
            if registry:
                env_specs = registry.values()
                self._supported_environments = {spec.id for spec in env_specs}
            else:
                raise AttributeError("Registry not found")
        except Exception:
            # Fallback to common environments if gymnasium is not available
            self._supported_environments = {
                "CartPole-v1",
                "MountainCar-v0",
                "FrozenLake-v1",
                "Acrobot-v1",
                "LunarLander-v2",
            }

    def _get_environment_info(self, env_name: str) -> dict[str, Any]:
        """Get environment information.

        Args:
            env_name: Environment name

        Returns:
            Environment information dictionary
        """
        if env_name in self._environment_cache:
            return self._environment_cache[env_name]

        try:
            env = gym.make(env_name)
            env_info = {
                "name": env_name,
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "reward_range": getattr(env, "reward_range", None),
                "spec": env.spec,
            }
            env.close()

            self._environment_cache[env_name] = env_info
            return env_info

        except Exception as e:
            raise ValueError(
                f"Could not get information for environment '{env_name}': {e}"
            ) from e

    def _validate_rl_environment_compatibility(
        self, algorithm: Algorithm, env_info: dict[str, Any]
    ) -> list[ValidationResult]:
        """Validate RL algorithm compatibility with environment.

        Args:
            algorithm: Algorithm configuration
            env_info: Environment information

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        # Check if environment has discrete action space (required for tabular methods)
        if algorithm.slug in ["q-learning"]:
            action_space = env_info["action_space"]
            if not hasattr(action_space, "n") or not isinstance(action_space.n, int):
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Algorithm '{algorithm.name}' requires discrete action space",
                        parameter_name="env",
                        error_code="INCOMPATIBLE_ACTION_SPACE",
                    )
                )

        return results

    def normalize_parameters(
        self, algorithm: Algorithm, parameters: dict[str, Any], _mode: str = "train"
    ) -> dict[str, Any]:
        """Normalize and convert parameters to appropriate types.

        Args:
            algorithm: Algorithm configuration
            parameters: ParameterDefinitions to normalize
            mode: Execution mode

        Returns:
            Normalized parameters dictionary
        """
        normalized = {}

        for param_name, param_value in parameters.items():
            param_spec = self._get_parameter_specification(algorithm, param_name)

            if param_spec is None:
                # Keep unknown parameters as-is
                normalized[param_name] = param_value
            else:
                normalized[param_name] = self._convert_parameter_value(
                    param_spec.type, param_value
                )

        return normalized

    @staticmethod
    def _convert_parameter_value(param_type: str, value: Any) -> Any:
        """Convert parameter value to appropriate type.

        Args:
            param_type: Expected parameter type
            value: Value to convert

        Returns:
            Converted value, or original if conversion fails
        """
        if not isinstance(value, str):
            return value

        if param_type == "float":
            return ParameterDefinitionValidator._try_convert(value, float)
        if param_type == "int":
            return ParameterDefinitionValidator._try_convert(value, int)
        if param_type == "bool":
            return value.lower() in ("true", "1", "yes", "on")

        return value

    @staticmethod
    def _try_convert(value: str, converter: type) -> Any:
        """Attempt to convert value using converter function.

        Args:
            value: String value to convert
            converter: Conversion function (e.g., int, float)

        Returns:
            Converted value, or original if conversion fails
        """
        try:
            return converter(value)
        except ValueError:
            return value

    def get_parameter_suggestions(
        self, algorithm: Algorithm, mode: str = "train"
    ) -> dict[str, Any]:
        """Get suggested parameter values for an algorithm and mode.

        Args:
            algorithm: Algorithm configuration
            mode: Execution mode

        Returns:
            Dictionary of suggested parameter values
        """
        suggestions = {}

        # Get default values from parameter specifications
        required_params = self._get_required_parameters(algorithm, mode)

        for param_name in required_params:
            param_spec = self._get_parameter_specification(algorithm, param_name)
            if param_spec and param_spec.default is not None:
                suggestions[param_name] = param_spec.default

        # Add mode-specific suggestions
        if mode == "train":
            suggestions.update(
                {"episodes": 1000, "learning_rate": 0.1, "gamma": 0.9, "epsilon": 0.1}
            )
        elif mode == "replay":
            suggestions.update({"episodes": 10, "visualize": True})
        elif mode == "demo":
            suggestions.update({"episodes": 5, "interactive": True})

        return suggestions
