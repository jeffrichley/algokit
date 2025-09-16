# AGLoViz - Algorithm Visualization CLI

AGLoViz is a standalone CLI tool for rendering algorithm visualizations using Manim. It provides a clean, professional interface for generating visualizations of various algorithms with the HarborNet: Search & Rescue narrative.

## Features

- **Multiple Algorithms**: Support for BFS, DFS, A*, and more (extensible)
- **Multiple Output Formats**: MP4, GIF, and image sequences
- **Quality Settings**: Low, medium, and high quality rendering
- **Scenario Management**: Easy loading and validation of HarborNet scenarios
- **Rich CLI Interface**: Beautiful terminal output with progress bars and formatting
- **Auto-Discovery**: Automatically finds and registers new algorithm scenes

## Installation

```bash
# Install AGLoViz
cd viz/
pip install -e .

# Verify installation
agloviz --version
```

## Quick Start

```bash
# List available algorithms
agloviz list algorithms

# List available scenarios
agloviz list scenarios

# Render BFS visualization
agloviz render bfs --scenario data/examples/scenarios/harbor_storm.yaml

# Render with different settings
agloviz render bfs --scenario harbor_flood.yaml --format gif --quality high

# Validate scenario file
agloviz validate scenario my_scenario.yaml
```

## Commands

### Render Commands

```bash
# Render specific algorithms
agloviz render bfs --scenario scenario.yaml
agloviz render dfs --scenario scenario.yaml
agloviz render astar --scenario scenario.yaml

# Render any algorithm
agloviz render algorithm bfs --scenario scenario.yaml

# Custom output settings
agloviz render bfs --scenario scenario.yaml --format gif --quality high --output my_video.mp4
```

### List Commands

```bash
# List available algorithms
agloviz list algorithms

# List available scenarios
agloviz list scenarios

# List algorithm families
agloviz list families
```

### Validate Commands

```bash
# Validate a single scenario
agloviz validate scenario my_scenario.yaml

# Validate all scenarios in directory
agloviz validate all --dir data/examples/scenarios
```

## Output Formats

- **MP4** (default): High-quality video file
- **GIF**: Animated GIF for web sharing
- **Images**: PNG image sequence for frame-by-frame analysis

## Quality Settings

- **Low**: Fast rendering, 480p resolution
- **Medium** (default): Balanced rendering, 720p resolution  
- **High**: Slow rendering, 1080p resolution

## Scenario Files

AGLoViz uses HarborNet scenario files in YAML format. Example:

```yaml
name: "Storm-Damaged Harbor"
description: "A challenging pathfinding scenario with obstacles"
width: 10
height: 10
start: [0, 0]
goal: [9, 9]
obstacles:
  - [1, 1]
  - [2, 2]
  - [3, 3]
```

## Architecture

```
viz/
├── agloviz/                    # Main CLI package
│   ├── main.py                 # CLI entry point
│   ├── commands/               # Command implementations
│   │   ├── render.py           # Render commands
│   │   ├── list.py             # List commands
│   │   └── validate.py         # Validate commands
│   ├── utils/                  # Utility modules
│   │   ├── scene_registry.py   # Algorithm scene discovery
│   │   ├── render_helpers.py   # Rendering utilities
│   │   └── config.py           # Configuration management
│   └── tests/                  # Test modules
└── manim/                      # Manim scene implementations
    ├── pathfinding/            # Pathfinding algorithms
    │   ├── bfs_scene.py        # BFS visualization
    │   ├── dfs_scene.py        # DFS visualization
    │   └── astar_scene.py      # A* visualization
    └── reinforcement_learning/ # RL algorithms
        ├── qlearning_scene.py  # Q-Learning visualization
        └── sarsa_scene.py      # SARSA visualization
```

## Adding New Algorithms

1. Create a new scene class in the appropriate `manim/` subdirectory
2. Follow the naming convention: `{algorithm}_scene.py`
3. Implement the required scene methods
4. AGLoViz will automatically discover and register the new scene

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest agloviz/tests/

# Format code
black agloviz/
ruff check agloviz/

# Type checking
mypy agloviz/
```

## Dependencies

- **Typer**: CLI framework
- **Rich**: Beautiful terminal output
- **Manim**: Mathematical animation engine
- **PyYAML**: YAML file processing
- **Pydantic**: Data validation

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: https://github.com/jeffrichley/algokit/issues
- Documentation: https://algokit.readthedocs.io/
