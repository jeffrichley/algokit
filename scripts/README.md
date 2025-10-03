# AlgoKit Scripts

This directory contains utility scripts for the AlgoKit project.

## YAML Validation

### `validate_yaml.py`

A beautiful CLI tool for validating YAML files against their schemas to ensure data integrity and consistency. Built with Typer and Rich for an excellent user experience.

#### Installation

The validation tool requires the docs dependencies:

```bash
uv sync --group docs
```

#### Usage

```bash
# Show help and available commands
uv run --group docs python scripts/validate_yaml.py --help

# Validate all YAML files
uv run --group docs python scripts/validate_yaml.py validate

# Validate only family files
uv run --group docs python scripts/validate_yaml.py validate --families

# Validate only algorithm files
uv run --group docs python scripts/validate_yaml.py validate --algorithms

# Validate a specific file
uv run --group docs python scripts/validate_yaml.py validate mkdocs_plugins/data/dp/family.yaml

# Verbose output with detailed information
uv run --group docs python scripts/validate_yaml.py validate --families --verbose

# Show validator information
uv run --group docs python scripts/validate_yaml.py info
```

#### Features

- **🎨 Beautiful CLI**: Built with Typer and Rich for an excellent user experience
- **📊 Rich Output**: Colorful tables, progress bars, and detailed formatting
- **🔍 Family Schema Validation**: Validates `family.yaml` files against the complete AlgoKit family schema
- **🧮 Algorithm Schema Validation**: Validates algorithm YAML files (when implemented)
- **✅ YAML Syntax Checking**: Ensures files have valid YAML syntax
- **🛡️ Flexible Validation**: Uses Cerberus for robust schema validation with detailed error reporting
- **📦 Batch Processing**: Can validate all files at once or specific files individually
- **📋 Summary Tables**: Beautiful summary tables showing validation results
- **⚡ Progress Indicators**: Real-time progress bars for batch operations

#### Schema Coverage

The validator checks for:

**Family Files (`family.yaml`)**:
- **Required fields**: `id`, `name`, `slug`, `summary`, `description`
- **Content sections**: `key_characteristics`, `common_applications`, `concepts`, `complexity`, `domain_sections`
- **Algorithm management**: `algorithms` configuration (list of algorithm slugs)
- **Cross-references**: `related_families` (list of family objects with `id`, `relationship`, `description`)
- **Documentation**: `references` (list of reference objects), `tags` (list of tag objects)
- **Rendering**: `template_options` (display preferences for the documentation system)

**Algorithm Files**:
- **Required fields**: `slug`, `name`, `family_id`, `summary`, `description`
- **Content sections**: `formulation`, `properties`, `implementations`, `complexity`, `applications`, `educational_value`
- **Status**: `status` (development state)
- **Cross-references**: `related_algorithms` (list of algorithm objects with `slug`, `relationship`, `description`)
- **Documentation**: `references` (list of reference objects), `tags` (list of tag objects)

#### Field Usage Status

**✅ Used Fields** (rendered in documentation):
- **Family**: All fields except `meta` are used in templates
- **Algorithm**: All fields except `aliases`, `order`, `template_options`, `meta` are used in templates

**❌ Removed Fields** (cleaned from YAML files):
- **Family**: `meta` (metadata not displayed in templates)
- **Algorithm**: `aliases`, `order`, `template_options`, `meta` (not used in generation system)

**🔧 Schema Evolution**:
- **Algorithm complexity**: Standardized to `analysis` array format (list of objects with `approach`, `time`, `space`, `notes`)
- **Cross-references**: Updated to use object format with `slug`/`id`, `relationship`, `description` fields
- **Field naming**: `family` → `family_id`, `id` → `slug` to match generation system expectations

#### Cleanup Status

**✅ Completed Phases**:
- **Phase 1**: Updated validator schemas to match actual YAML structure
- **Phase 2**: Cleaned algorithm files (removed unused fields: `aliases`, `order`, `template_options`, `meta`)
- **Phase 3**: Cleaned family files (removed unused field: `meta`)
- **Phase 4**: Validated all files pass schema validation and documentation builds successfully

**📊 Current State**:
- **2 family files**: All validated and cleaned
- **4 algorithm files**: All validated and cleaned
- **Schema coverage**: 100% of used fields validated
- **Documentation**: Builds successfully with all content intact

#### Quick Reference

**Current Schema Structure**:

```yaml
# Family file structure
id: string                    # Family identifier (e.g., "dp", "rl")
name: string                  # Display name
slug: string                  # URL slug
summary: string               # Brief description
description: string           # Detailed description
key_characteristics: [string] # List of key features
common_applications: [string] # Use cases
concepts: [string]            # Key concepts
complexity: object            # Complexity analysis
domain_sections: [object]     # Custom sections
algorithms: [string]          # List of algorithm slugs
related_families: [object]    # Cross-references
references: [object]          # Bibliography entries
tags: [object]                # Categorization tags
template_options: object      # Display preferences

# Algorithm file structure
slug: string                  # Algorithm identifier
name: string                  # Display name
family_id: string             # Parent family ID
summary: string               # Brief description
description: string           # Detailed description
formulation: string           # Mathematical formulation
properties: [string]          # Key properties
implementations: [object]     # Implementation approaches
complexity: object            # Complexity analysis (analysis array)
applications: [string]        # Use cases
educational_value: string     # Learning benefits
status: string                # Development status
related_algorithms: [object]  # Cross-references
references: [object]          # Bibliography entries
tags: [object]                # Categorization tags
```

#### Integration

This script can be integrated into:
- **Pre-commit hooks**: Validate YAML before commits
- **CI/CD pipelines**: Ensure data integrity in automated builds
- **Development workflow**: Quick validation during development

#### Dependencies

- `typer`: Modern CLI framework with automatic help generation
- `rich`: Beautiful terminal output with colors, tables, and progress bars
- `cerberus`: Schema validation library
- `pyyaml`: YAML parsing
- `pathlib`: File system operations

All dependencies are managed through the project's `uv` configuration and are part of the `docs` dependency group.
