"""Configuration data model for CLI operations.

This module provides Pydantic models for managing CLI configuration with
hierarchical structure, validation, and persistence capabilities.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class LogLevel(str, Enum):
    """Logging level enumeration."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogFormat(str, Enum):
    """Log format enumeration."""

    RICH = "rich"
    SIMPLE = "simple"
    JSON = "json"
    PLAIN = "plain"


class PlotFormat(str, Enum):
    """Plot format enumeration."""

    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    SVG = "svg"
    PDF = "pdf"


class GlobalConfig(BaseModel):
    """Global CLI configuration."""

    output_dir: str = Field(default="output", description="Output directory path")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: LogFormat = Field(default=LogFormat.RICH, description="Log format")
    auto_cleanup: bool = Field(default=True, description="Enable automatic cleanup")
    max_runs: int = Field(default=100, description="Maximum number of runs to keep")

    @field_validator("max_runs")
    @classmethod
    def validate_max_runs(cls, v: int) -> int:
        """Validate max_runs is positive."""
        if v <= 0:
            raise ValueError("max_runs must be positive")
        return v


class RLConfig(BaseModel):
    """Reinforcement Learning family configuration."""

    default_env: str = Field(default="CartPole-v1", description="Default environment")
    default_episodes: int = Field(
        default=1000, description="Default number of episodes"
    )
    default_learning_rate: float = Field(
        default=0.01, description="Default learning rate"
    )
    default_gamma: float = Field(default=0.99, description="Default discount factor")
    default_epsilon: float = Field(
        default=0.1, description="Default epsilon for exploration"
    )
    default_epsilon_decay: float = Field(
        default=0.995, description="Default epsilon decay"
    )

    @field_validator("default_episodes")
    @classmethod
    def validate_episodes(cls, v: int) -> int:
        """Validate episodes is positive."""
        if v <= 0:
            raise ValueError("default_episodes must be positive")
        return v

    @field_validator("default_learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning rate is in valid range."""
        if not 0 < v <= 1:
            raise ValueError("default_learning_rate must be in range (0, 1]")
        return v

    @field_validator("default_gamma")
    @classmethod
    def validate_gamma(cls, v: float) -> float:
        """Validate gamma is in valid range."""
        if not 0 <= v <= 1:
            raise ValueError("default_gamma must be in range [0, 1]")
        return v

    @field_validator("default_epsilon")
    @classmethod
    def validate_epsilon(cls, v: float) -> float:
        """Validate epsilon is in valid range."""
        if not 0 <= v <= 1:
            raise ValueError("default_epsilon must be in range [0, 1]")
        return v

    @field_validator("default_epsilon_decay")
    @classmethod
    def validate_epsilon_decay(cls, v: float) -> float:
        """Validate epsilon decay is in valid range."""
        if not 0 < v <= 1:
            raise ValueError("default_epsilon_decay must be in range (0, 1]")
        return v


class DMPsConfig(BaseModel):
    """Dynamic Movement Primitives family configuration."""

    default_duration: float = Field(
        default=5.0, description="Default movement duration"
    )
    default_trajectory: str = Field(
        default="circle", description="Default trajectory type"
    )
    default_scaling: float = Field(default=1.0, description="Default scaling factor")
    default_alpha: float = Field(default=25.0, description="Default alpha parameter")
    default_beta: float = Field(default=6.25, description="Default beta parameter")

    @field_validator("default_duration")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        """Validate duration is positive."""
        if v <= 0:
            raise ValueError("default_duration must be positive")
        return v

    @field_validator("default_scaling")
    @classmethod
    def validate_scaling(cls, v: float) -> float:
        """Validate scaling is positive."""
        if v <= 0:
            raise ValueError("default_scaling must be positive")
        return v

    @field_validator("default_alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        """Validate alpha is positive."""
        if v <= 0:
            raise ValueError("default_alpha must be positive")
        return v

    @field_validator("default_beta")
    @classmethod
    def validate_beta(cls, v: float) -> float:
        """Validate beta is positive."""
        if v <= 0:
            raise ValueError("default_beta must be positive")
        return v


class ControlConfig(BaseModel):
    """Control Systems family configuration."""

    default_system: str = Field(
        default="second_order", description="Default system type"
    )
    default_sampling_rate: int = Field(
        default=100, description="Default sampling rate (Hz)"
    )
    default_pid_kp: float = Field(default=1.0, description="Default PID Kp gain")
    default_pid_ki: float = Field(default=0.1, description="Default PID Ki gain")
    default_pid_kd: float = Field(default=0.01, description="Default PID Kd gain")

    @field_validator("default_sampling_rate")
    @classmethod
    def validate_sampling_rate(cls, v: float) -> float:
        """Validate sampling rate is positive."""
        if v <= 0:
            raise ValueError("default_sampling_rate must be positive")
        return v


class FamilyConfigs(BaseModel):
    """Family-specific configurations."""

    rl: RLConfig = Field(
        default_factory=RLConfig, description="RL family configuration"
    )
    dmps: DMPsConfig = Field(
        default_factory=DMPsConfig, description="DMPs family configuration"
    )
    control: ControlConfig = Field(
        default_factory=ControlConfig, description="Control family configuration"
    )


class OutputConfig(BaseModel):
    """Output preferences configuration."""

    save_models: bool = Field(default=True, description="Save trained models")
    save_logs: bool = Field(default=True, description="Save training logs")
    save_videos: bool = Field(default=True, description="Save training videos")
    save_plots: bool = Field(default=True, description="Save training plots")
    video_fps: int = Field(default=30, description="Video frame rate")
    plot_format: PlotFormat = Field(
        default=PlotFormat.PNG, description="Plot file format"
    )
    log_retention_days: int = Field(
        default=30, description="Log retention period in days"
    )

    @field_validator("video_fps")
    @classmethod
    def validate_video_fps(cls, v: int) -> int:
        """Validate video FPS is positive."""
        if v <= 0:
            raise ValueError("video_fps must be positive")
        return v

    @field_validator("log_retention_days")
    @classmethod
    def validate_retention_days(cls, v: int) -> int:
        """Validate retention days is non-negative."""
        if v < 0:
            raise ValueError("log_retention_days must be non-negative")
        return v


class ExecutionConfig(BaseModel):
    """Execution preferences configuration."""

    max_workers: int = Field(
        default=4, description="Maximum number of worker processes"
    )
    timeout_seconds: int = Field(default=3600, description="Default timeout in seconds")
    memory_limit_gb: int = Field(default=8, description="Memory limit in GB")
    gpu_enabled: bool = Field(default=False, description="Enable GPU acceleration")

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v: int) -> int:
        """Validate max workers is positive."""
        if v <= 0:
            raise ValueError("max_workers must be positive")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("timeout_seconds must be positive")
        return v

    @field_validator("memory_limit_gb")
    @classmethod
    def validate_memory_limit(cls, v: int) -> int:
        """Validate memory limit is positive."""
        if v <= 0:
            raise ValueError("memory_limit_gb must be positive")
        return v


class AlgorithmConfig(BaseModel):
    """Algorithm-specific configuration."""

    # Algorithm-specific configurations will be added here as algorithms are implemented

    # Q-Learning configuration
    q_learning: dict[str, Any] = Field(
        default_factory=dict, description="Q-Learning algorithm configuration"
    )

    # DQN configuration
    dqn: dict[str, Any] = Field(
        default_factory=dict, description="DQN algorithm configuration"
    )

    # Policy Gradient configuration
    policy_gradient: dict[str, Any] = Field(
        default_factory=dict, description="Policy Gradient algorithm configuration"
    )

    # Actor-Critic configuration
    actor_critic: dict[str, Any] = Field(
        default_factory=dict, description="Actor-Critic algorithm configuration"
    )

    # PPO configuration
    ppo: dict[str, Any] = Field(
        default_factory=dict, description="PPO algorithm configuration"
    )


class Config(BaseModel):
    """Main CLI configuration model."""

    global_: GlobalConfig = Field(
        default_factory=GlobalConfig, alias="global", description="Global configuration"
    )
    families: FamilyConfigs = Field(
        default_factory=FamilyConfigs, description="Family configurations"
    )
    algorithms: AlgorithmConfig = Field(
        default_factory=AlgorithmConfig, description="Algorithm configurations"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig, description="Output configuration"
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig, description="Execution configuration"
    )

    model_config = ConfigDict(populate_by_name=True, validate_assignment=True)

    @classmethod
    def from_yaml_file(cls, file_path: str | Path) -> Config:
        """Load configuration from YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Config instance loaded from YAML

        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the YAML is invalid
            ValidationError: If the data doesn't match the schema
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml_file(self, file_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            file_path: Path to save the YAML file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.dict(by_alias=True), f, default_flow_style=False, sort_keys=False
            )

    @classmethod
    def get_default_config(cls) -> Config:
        """Get default configuration.

        Returns:
            Default configuration instance
        """
        return cls()

    def get_family_config(self, family_id: str) -> dict[str, Any] | None:
        """Get configuration for a specific family.

        Args:
            family_id: Family identifier

        Returns:
            Family configuration if found, None otherwise
        """
        if family_id == "rl":
            return self.families.rl.dict()
        elif family_id == "dmps":
            return self.families.dmps.dict()
        elif family_id == "control":
            return self.families.control.dict()
        else:
            return None

    def get_algorithm_config(self, algorithm_id: str) -> dict[str, Any] | None:
        """Get configuration for a specific algorithm.

        Args:
            algorithm_id: Algorithm identifier

        Returns:
            Algorithm configuration if found, None otherwise
        """
        return getattr(self.algorithms, algorithm_id, None)

    def set_family_config(self, family_id: str, config: dict[str, Any]) -> None:
        """Set configuration for a specific family.

        Args:
            family_id: Family identifier
            config: Configuration dictionary
        """
        if family_id == "rl":
            self.families.rl = RLConfig(**config)
        elif family_id == "dmps":
            self.families.dmps = DMPsConfig(**config)
        elif family_id == "control":
            self.families.control = ControlConfig(**config)
        else:
            raise ValueError(f"Unknown family: {family_id}")

    def set_algorithm_config(self, algorithm_id: str, config: dict[str, Any]) -> None:
        """Set configuration for a specific algorithm.

        Args:
            algorithm_id: Algorithm identifier
            config: Configuration dictionary
        """
        if hasattr(self.algorithms, algorithm_id):
            setattr(self.algorithms, algorithm_id, config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_id}")

    def merge_config(self, other: Config) -> Config:
        """Merge another configuration into this one.

        Args:
            other: Configuration to merge

        Returns:
            New merged configuration
        """
        # Create a copy of this config
        merged = self.copy()

        # Merge global config
        merged.global_ = GlobalConfig(**{**self.global_.dict(), **other.global_.dict()})

        # Merge family configs
        merged.families.rl = RLConfig(
            **{**self.families.rl.dict(), **other.families.rl.dict()}
        )
        merged.families.dmps = DMPsConfig(
            **{**self.families.dmps.dict(), **other.families.dmps.dict()}
        )
        merged.families.control = ControlConfig(
            **{**self.families.control.dict(), **other.families.control.dict()}
        )

        # Merge algorithm configs
        for algorithm_id in [
            "q_learning",
            "dqn",
            "policy_gradient",
            "actor_critic",
            "ppo",
        ]:
            current_config = getattr(self.algorithms, algorithm_id, {})
            other_config = getattr(other.algorithms, algorithm_id, {})
            merged_config = {**current_config, **other_config}
            setattr(merged.algorithms, algorithm_id, merged_config)

        # Merge output and execution configs
        merged.output = OutputConfig(**{**self.output.dict(), **other.output.dict()})
        merged.execution = ExecutionConfig(
            **{**self.execution.dict(), **other.execution.dict()}
        )

        return merged

    def get_config_path(
        self, family_id: str | None = None, algorithm_id: str | None = None
    ) -> str:
        """Get configuration path for hierarchical access.

        Args:
            family_id: Optional family identifier
            algorithm_id: Optional algorithm identifier

        Returns:
            Configuration path string
        """
        if algorithm_id and family_id:
            return f"families.{family_id}.algorithms.{algorithm_id}"
        elif family_id:
            return f"families.{family_id}"
        else:
            return "global"

    def get_nested_value(self, path: str) -> Any:
        """Get nested configuration value by path.

        Args:
            path: Dot-separated path to the value

        Returns:
            Configuration value

        Raises:
            KeyError: If the path doesn't exist
        """
        keys = path.split(".")
        value = self.dict(by_alias=True)

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Configuration path not found: {path}")

        return value

    def set_nested_value(self, path: str, value: Any) -> None:
        """Set nested configuration value by path.

        Args:
            path: Dot-separated path to the value
            value: Value to set

        Raises:
            KeyError: If the path doesn't exist
        """
        keys = path.split(".")
        config_dict = self.dict(by_alias=True)
        current = config_dict

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                raise KeyError(f"Configuration path not found: {path}")
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

        # Update the config object with new values
        for key, value in config_dict.items():
            setattr(self, key, value)

    def validate_config(self) -> list[str]:
        """Validate the entire configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            # Validate the entire config
            self.validate(self.dict(by_alias=True))
        except Exception as e:
            errors.append(f"Configuration validation failed: {str(e)}")

        return errors

    # SARSA defaults method removed - no longer implemented

    def get_output_directory(self) -> Path:
        """Get the output directory path.

        Returns:
            Output directory path
        """
        return Path(self.global_.output_dir)

    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled.

        Returns:
            True if GPU is enabled, False otherwise
        """
        return self.execution.gpu_enabled

    def get_memory_limit_bytes(self) -> int:
        """Get memory limit in bytes.

        Returns:
            Memory limit in bytes
        """
        return self.execution.memory_limit_gb * 1024 * 1024 * 1024
