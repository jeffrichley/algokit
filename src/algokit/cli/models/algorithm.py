"""Algorithm data model for CLI operations.

This module provides Pydantic models for loading and validating algorithm
configurations from YAML files, with support for algorithm metadata,
parameter validation, and type checking.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class AlgorithmStatus(str, Enum):
    """Algorithm implementation status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    DEPRECATED = "deprecated"


class ImplementationQuality(str, Enum):
    """Algorithm implementation quality."""

    NONE = "none"
    BASIC = "basic"
    GOOD = "good"
    EXCELLENT = "excellent"


class CoverageLevel(str, Enum):
    """Test coverage level."""

    NONE = "none"
    PARTIAL = "partial"
    GOOD = "good"
    COMPREHENSIVE = "comprehensive"


class DocumentationQuality(str, Enum):
    """Documentation quality level."""

    NONE = "none"
    BASIC = "basic"
    GOOD = "good"
    EXCELLENT = "excellent"
    PLANNED = "planned"


class ParameterType(str, Enum):
    """Parameter type enumeration."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    PATH = "path"


class ParameterDefinition(BaseModel):
    """Definition of an algorithm parameter."""

    name: str = Field(..., description="Parameter name")
    type: ParameterType = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: str | int | float | bool | list[Any] | dict[str, Any] | None = Field(
        default=None, description="Default value"
    )
    min_value: int | float | None = Field(default=None, description="Minimum value")
    max_value: int | float | None = Field(default=None, description="Maximum value")
    choices: list[Any] | None = Field(default=None, description="Allowed values")
    validation_regex: str | None = Field(
        default=None, description="Validation regex pattern"
    )

    @field_validator("default")
    @classmethod
    def validate_default_type(
        cls, v: str | int | float | bool | list[Any] | dict[str, Any] | None, info: Any
    ) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
        """Validate that default value matches parameter type."""
        if v is None:
            return v

        # Get the type field from the model data
        if hasattr(info, "data") and "type" in info.data:
            param_type = info.data["type"]
        else:
            return v

        if param_type == ParameterType.STRING and not isinstance(v, str):
            raise ValueError(
                f"Default value for string parameter must be string, got {type(v)}"
            )
        elif param_type == ParameterType.INTEGER and not isinstance(v, int):
            raise ValueError(
                f"Default value for integer parameter must be int, got {type(v)}"
            )
        elif param_type == ParameterType.FLOAT and not isinstance(v, int | float):
            raise ValueError(
                f"Default value for float parameter must be number, got {type(v)}"
            )
        elif param_type == ParameterType.BOOLEAN and not isinstance(v, bool):
            raise ValueError(
                f"Default value for boolean parameter must be bool, got {type(v)}"
            )
        elif param_type == ParameterType.LIST and not isinstance(v, list):
            raise ValueError(
                f"Default value for list parameter must be list, got {type(v)}"
            )
        elif param_type == ParameterType.DICT and not isinstance(v, dict):
            raise ValueError(
                f"Default value for dict parameter must be dict, got {type(v)}"
            )
        elif param_type == ParameterType.PATH and not isinstance(v, str):
            raise ValueError(
                f"Default value for path parameter must be string, got {type(v)}"
            )

        return v


class ImplementationApproach(BaseModel):
    """Implementation approach for an algorithm."""

    type: str = Field(..., description="Implementation type identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Implementation description")
    complexity: dict[str, str] | None = Field(
        default=None, description="Complexity analysis"
    )
    code: str | None = Field(default=None, description="Code example")
    advantages: list[str] | None = Field(
        default=None, description="Implementation advantages"
    )
    disadvantages: list[str] | None = Field(
        default=None, description="Implementation disadvantages"
    )


class AlgorithmProperty(BaseModel):
    """Algorithm property or characteristic."""

    name: str = Field(..., description="Property name")
    description: str = Field(..., description="Property description")
    importance: Literal["fundamental", "implementation", "optimization"] = Field(
        ..., description="Property importance level"
    )


class KeyProperty(BaseModel):
    """Key mathematical property of the algorithm."""

    name: str = Field(..., description="Property name")
    formula: str | None = Field(default=None, description="Mathematical formula")
    description: str = Field(..., description="Property description")


class ProblemFormulation(BaseModel):
    """Problem formulation and mathematical details."""

    problem_definition: str = Field(
        ..., description="Problem definition in mathematical terms"
    )
    key_properties: list[KeyProperty] = Field(
        default_factory=list, description="Key mathematical properties"
    )


class ComplexityAnalysis(BaseModel):
    """Complexity analysis for different approaches."""

    approach: str = Field(..., description="Approach name")
    time: str = Field(..., description="Time complexity")
    space: str = Field(..., description="Space complexity")
    notes: str | None = Field(default=None, description="Additional complexity notes")


class ApplicationCategory(BaseModel):
    """Application category with examples."""

    category: str = Field(..., description="Application category name")
    examples: list[str] = Field(..., description="Example applications")


class Reference(BaseModel):
    """Reference or resource."""

    author: str | None = Field(default=None, description="Author name")
    year: str | None = Field(default=None, description="Publication year")
    title: str = Field(..., description="Title")
    publisher: str | None = Field(default=None, description="Publisher")
    note: str | None = Field(default=None, description="Additional notes")
    url: str | None = Field(default=None, description="URL for online resources")
    bib_key: str | None = Field(default=None, description="Bibliography key")


class ReferenceCategory(BaseModel):
    """Category of references."""

    category: str = Field(..., description="Reference category name")
    items: list[Reference] = Field(..., description="References in this category")


class RelatedAlgorithm(BaseModel):
    """Related algorithm reference."""

    slug: str = Field(..., description="Related algorithm slug")
    relationship: Literal["same_family", "alternative", "extension", "foundation"] = (
        Field(..., description="Relationship type")
    )
    description: str = Field(..., description="Relationship description")


class SourceFile(BaseModel):
    """Source file reference."""

    path: str = Field(..., description="File path")
    description: str = Field(..., description="File description")


class AlgorithmStatusModel(BaseModel):
    """Algorithm implementation status."""

    current: AlgorithmStatus = Field(..., description="Current implementation status")
    implementation_quality: ImplementationQuality = Field(
        ..., description="Implementation quality"
    )
    test_coverage: CoverageLevel = Field(..., description="Test coverage level")
    documentation_quality: DocumentationQuality = Field(
        ..., description="Documentation quality"
    )
    source_files: list[SourceFile] = Field(
        default_factory=list, description="Source file references"
    )


class Algorithm(BaseModel):
    """Algorithm data model for loading and validating algorithm configurations."""

    # Basic metadata
    slug: str = Field(..., description="Algorithm slug identifier")
    name: str = Field(..., description="Algorithm name")
    family_id: str = Field(..., description="Algorithm family identifier")
    summary: str = Field(..., description="Brief one-sentence summary")
    description: str = Field(..., description="Detailed algorithm description")

    # Problem formulation
    formulation: ProblemFormulation | None = Field(
        default=None, description="Problem formulation"
    )

    # Algorithm properties
    properties: list[AlgorithmProperty] = Field(
        default_factory=list, description="Algorithm properties"
    )

    # Implementation approaches
    implementations: list[ImplementationApproach] = Field(
        default_factory=list, description="Implementation approaches"
    )

    # Complexity analysis
    complexity: dict[str, list[ComplexityAnalysis]] | None = Field(
        default=None, description="Complexity analysis"
    )

    # Applications
    applications: list[ApplicationCategory] = Field(
        default_factory=list, description="Applications"
    )

    # Educational value
    educational_value: list[str] = Field(
        default_factory=list, description="Educational value points"
    )

    # Status
    status: AlgorithmStatusModel | None = Field(
        default=None, description="Implementation status"
    )

    # References
    references: list[ReferenceCategory] = Field(
        default_factory=list, description="References"
    )

    # Tags
    tags: list[str] = Field(default_factory=list, description="Algorithm tags")

    # Related algorithms
    related_algorithms: list[RelatedAlgorithm] = Field(
        default_factory=list, description="Related algorithms"
    )

    # CLI-specific fields
    parameters: list[ParameterDefinition] = Field(
        default_factory=list, description="CLI parameters for this algorithm"
    )

    @classmethod
    def from_yaml_file(cls, file_path: str | Path) -> Algorithm:
        """Load algorithm configuration from YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Algorithm instance loaded from YAML

        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the YAML is invalid
            ValidationError: If the data doesn't match the schema
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Algorithm file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml_file(self, file_path: str | Path) -> None:
        """Save algorithm configuration to YAML file.

        Args:
            file_path: Path to save the YAML file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)

    def get_parameter(self, name: str) -> ParameterDefinition | None:
        """Get parameter definition by name.

        Args:
            name: Parameter name

        Returns:
            Parameter definition if found, None otherwise
        """
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def get_required_parameters(self) -> list[ParameterDefinition]:
        """Get list of required parameters.

        Returns:
            List of required parameter definitions
        """
        return [param for param in self.parameters if param.required]

    def get_optional_parameters(self) -> list[ParameterDefinition]:
        """Get list of optional parameters.

        Returns:
            List of optional parameter definitions
        """
        return [param for param in self.parameters if not param.required]

    def validate_parameter_value(self, name: str, value: Any) -> bool:
        """Validate a parameter value against its definition.

        Args:
            name: Parameter name
            value: Value to validate

        Returns:
            True if valid, False otherwise
        """
        param = self.get_parameter(name)
        if param is None:
            return False

        # Type validation
        if (
            param.type == ParameterType.STRING
            and not isinstance(value, str)
            or param.type == ParameterType.INTEGER
            and not isinstance(value, int)
            or param.type == ParameterType.FLOAT
            and not isinstance(value, int | float)
            or param.type == ParameterType.BOOLEAN
            and not isinstance(value, bool)
            or param.type == ParameterType.LIST
            and not isinstance(value, list)
            or param.type == ParameterType.DICT
            and not isinstance(value, dict)
            or param.type == ParameterType.PATH
            and not isinstance(value, str)
        ):
            return False

        # Range validation
        if (
            param.min_value is not None
            and isinstance(value, int | float)
            and value < param.min_value
        ):
            return False

        if (
            param.max_value is not None
            and isinstance(value, int | float)
            and value > param.max_value
        ):
            return False

        # Choices validation
        return param.choices is None or value in param.choices

    def get_family_directory(self) -> str:
        """Get the family directory name for this algorithm.

        Returns:
            Family directory name
        """
        return self.family_id

    def get_algorithm_file_path(self) -> str:
        """Get the expected algorithm file path.

        Returns:
            Expected file path for this algorithm
        """
        return f"{self.family_id}/algorithms/{self.slug}.yaml"

    def is_implemented(self) -> bool:
        """Check if the algorithm is implemented.

        Returns:
            True if algorithm is implemented, False otherwise
        """
        if self.status is None:
            return False
        return self.status.current in [
            AlgorithmStatus.COMPLETE,
            AlgorithmStatus.IN_PROGRESS,
        ]

    def get_implementation_quality(self) -> ImplementationQuality:
        """Get the implementation quality.

        Returns:
            Implementation quality level
        """
        if self.status is None:
            return ImplementationQuality.NONE
        return self.status.implementation_quality

    def get_test_coverage(self) -> CoverageLevel:
        """Get the test coverage level.

        Returns:
            Test coverage level
        """
        if self.status is None:
            return CoverageLevel.NONE
        return self.status.test_coverage

    def get_documentation_quality(self) -> DocumentationQuality:
        """Get the documentation quality level.

        Returns:
            Documentation quality level
        """
        if self.status is None:
            return DocumentationQuality.NONE
        return self.status.documentation_quality
