# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-09-01

### Added
- Initial project setup
- Basic project structure with src layout
- Development tooling (pytest, ruff, mypy, etc.)
- Documentation setup with MkDocs
- CI/CD workflows
- Pre-commit hooks including commitizen

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## v0.13.0 (2025-10-07)

### Feat

- Complete state-of-the-art hierarchical RL implementations with refinements
- add hierarchical reinforcement learning algorithms

### Fix

- add missing rl markers to pytest-algorithms.ini

### Refactor

- ActorCritic to Pydantic config pattern

## v0.12.0 (2025-10-03)

### Feat

- Add comprehensive reinforcement learning algorithms (#7)

## v0.11.1 (2025-10-03)

### Fix

- use getattr for safe gymnasium registry access
- resolve mypy yaml import issues
- resolve pre-commit mypy issues
- align CI workflow with local just quality setup
- move decorators to viz-source and exclude from coverage
- remove unused decorators.py to improve coverage
- use --install-types to automatically install missing type stubs
- update pre-commit mypy to use pyproject.toml and match local version
- disable mypy in pre-commit, use GitHub Actions instead
- create dedicated mypy.ini for pre-commit hooks
- configure pre-commit mypy to use same config as local
- ignore viz_tests and remove pydantic mypy plugin
- simplify pre-commit configuration
- remove uv run commands from pre-commit config for CI compatibility
- make pre-commit configuration cross-platform compatible
- resolve pre-commit CI failures

### Refactor

- remove agloviz visualization system and clean up codebase

## v0.11.0 (2025-10-02)

### Feat

- Implement A* pathfinding algorithm with shared distance utilities
- restructure algorithms under algokit.algorithms package
- Add CLI render command with timing infrastructure

### Fix

- remove duplicate tag to allow Commitizen bump
- update remaining imports and linting issues
- replace custom SVG icons with Material Design icons

## v0.10.0 (2025-09-06)

### Feat

- add missing algorithm pages to navigation

## v0.9.1 (2025-09-06)

### Fix

- correct icon paths in algorithm page template

## v0.9.0 (2025-09-06)

### Feat

- add Google Analytics tracking with feedback widget

## v0.8.0 (2025-09-06)

### Feat

- add missing algorithms for Gaussian Process, Planning, and Real-time Control families

## v0.7.2 (2025-09-06)

### Fix

- restore missing SVG icons for documentation

## v0.7.1 (2025-09-06)

### Fix

- resolve CI build failures

## v0.7.0 (2025-09-05)

### Feat

- update mkdocs automation system and documentation structure
- enhance references section with Amazon affiliate links and improved styling
- Create comprehensive algorithms.yaml data structure
- Install and configure MkDocs Macros plugin
- Restructure documentation with new algorithm organization and navigation fixes

### Fix

- ChatGPT buttons with unique prompts and section-specific button text
- improve algorithm family status badge logic

## v0.6.1 (2025-09-02)

### Fix

- **pre-commit**: use nox sessions for local hooks to maintain proper dependency management

## v0.6.0 (2025-09-02)

### Feat

- **quality**: add codespell integration for spell checking

### Fix

- **pre-commit**: use python language for local hooks to ensure proper uv environment

### Refactor

- **deps**: consolidate dependency groups and update tooling

## v0.5.0 (2025-09-02)

### Feat

- **structure**: complete project restructuring and fibonacci implementation

## v0.4.0 (2025-09-02)

### Feat

- **rules**: import comprehensive development standards from redwing_core

## v0.3.0 (2025-09-02)

### Feat

- upgrade to Python 3.12+ and add py.typed support

## v0.2.2 (2025-09-01)

### Fix

- **docs**: Fix Mermaid diagram rendering in documentation

## v0.2.1 (2025-09-01)

### Fix

- resolve pytest not found error in minimal mode tests

## v0.2.0 (2025-09-01)

### Feat

- improve dependency management and fix commitizen setup
- update all GitHub Actions to latest versions
- configure GitHub Pages deployment
- initial commit with MkDocs documentation system

### Fix

- enable Material icons in MkDocs documentation
- enable Material icons in MkDocs documentation
- add --system flag to uv pip install in deploy workflow
- remove unsupported uv cache from setup-python actions
- resolve CI build issues and configure proper GitHub Pages deployment
- simplify mkdocs configuration to resolve build issues

### Refactor

- remove duplicate docs.yml workflow
