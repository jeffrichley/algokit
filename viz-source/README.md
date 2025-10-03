# Visualization Source Code

This directory contains visualization-related code that depends on `manim` and other visualization libraries.

## Contents

- `agloviz/` - Algorithm visualization components using Manim
- `viz/` - Visualization adapters and utilities
- `bfs_with_events.py` - BFS implementation with event tracking for visualization
- `decorators.py` - Visualization-specific decorators (with_event_tracking)

## Dependencies

This code requires the `viz` optional dependency group to be installed:

```bash
pip install algokit[viz]
```

Or with uv:

```bash
uv sync --extra viz
```

## Purpose

This separation allows the core algorithm library to be used without visualization dependencies, which is important for:

- CI/CD environments that don't have system-level dependencies for manim
- Users who only need the algorithms without visualization
- Reducing installation complexity for core functionality

The visualization components are still available when needed by installing the optional dependencies.
