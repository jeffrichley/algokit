# Algorithm Kit

**Algorithm Kit** is a modern Python project built with best practices and comprehensive tooling for implementing control and learning algorithms.

[:material-arrow-right: Get Started](api.md){ .md-button .md-button--primary }
[:material-play: Run Tests](https://github.com/jeffrichley/algokit#development){ .md-button }

[:material-book-open: Documentation](api.md) · [:material-code-braces: Source Code](https://github.com/jeffrichley/algokit) · [:material-github: GitHub](https://github.com/jeffrichley/algokit)

## Algorithm Families

Explore the algorithm landscapes covered by Algorithm Kit.

| Family | Highlighted Algorithms | Status |
| --- | --- | --- |
| Dynamic Programming | Fibonacci, Coin Change, Knapsack | :material-code-tags: Code |
| Reinforcement Learning | Q-Learning, SARSA, DQN | :material-progress-clock: Coming Soon |
| Hierarchical RL | Options, Feudal RL, MAXQ, HIRO | :material-progress-clock: Coming Soon |
| Control | PID, Adaptive, Sliding Mode | :material-progress-clock: Coming Soon |
| Model Predictive Control | Linear, Nonlinear, Robust | :material-progress-clock: Coming Soon |
| Planning | A*, GraphPlan, Partial Order | :material-progress-clock: Coming Soon |
| Gaussian Process | Regression, Classification, Optimization | :material-progress-clock: Coming Soon |
| Dynamic Movement Primitives | DMP Encoding, Imitation Learning | :material-progress-clock: Coming Soon |
| Real-Time Control | PID, Bang-bang, Kalman Filter | :material-progress-clock: Coming Soon |

**Status legend**: :material-code-tags: Code available · :material-progress-clock: Coming Soon

### Family Overviews

Get a feel for the types of problems each family tackles before diving in.

#### Dynamic Programming
Break complex problems into overlapping subproblems with optimal substructure. Ideal for tasks like path finding, resource allocation, and sequence analysis.

**Available Algorithms:**
- Fibonacci Sequence - Complete implementation with multiple approaches

#### Reinforcement Learning
Learn policies through trial and error when environment dynamics are unknown. Useful for game playing, robotics, and other sequential decision-making tasks.

#### Hierarchical RL
Introduce temporal abstraction via options or subgoals to manage long-horizon tasks more efficiently.

#### Control
Design feedback-driven controllers for dynamic systems, including PID, adaptive, and sliding mode approaches.

#### Model Predictive Control
Solve constrained optimization problems over a moving horizon to compute control actions, balancing performance and safety.

#### Planning
Search state spaces for action sequences that achieve goals in robotics and AI planning domains.

#### Gaussian Process Modeling
Model functions with quantified uncertainty using Bayesian regression—great for surrogate modeling, regression, and Bayesian optimization.

#### Dynamic Movement Primitives
Encode smooth, reusable motion trajectories as stable dynamical systems, enabling robots to adapt motions to new goals and timings.

#### Real-Time Control
Design fast, feedback-driven controllers such as PID or Kalman filters for systems with tight latency requirements.

## Quick Start

Get up and running with Algorithm Kit in minutes:

- **Install**: `uv pip install -e .` (editable development install)
- **Test**: `just test` (run the test suite)
- **Lint**: `just lint` (check code quality)

=== "CLI"

    ```bash
    # Install in editable mode
    uv pip install -e .

    # Run tests
    just test

    # Check code quality
    just lint
    ```

=== "Python"

    ```python
    # Import the package
    import algokit

    # Your algorithm implementations here
    print("Algorithm Kit is ready!")
    ```

!!! tip "Need help setting up?"
    See our [Development Guide](https://github.com/jeffrichley/algokit#development) for environment setup and dependencies.

## Feature Highlights

- **Modern Python Tooling**: Built with uv and just for optimal development experience
- **Quality Assurance**: Comprehensive testing, linting, and type checking
- **Documentation**: Automated documentation generation with MkDocs
- **CI/CD Ready**: Pre-configured GitHub Actions workflows
- **Type Safety**: Full type hints and mypy integration
- **Extensible Architecture**: Modular design with plugin support

## Technology Stack

- **Package Manager**: uv for fast dependency management
- **Testing**: pytest with comprehensive coverage reporting
- **Linting**: ruff for fast Python linting
- **Formatting**: black for consistent code formatting
- **Type Checking**: mypy for static type analysis
- **Documentation**: MkDocs with Material theme
- **CI/CD**: GitHub Actions with quality gates

## Architecture at a Glance

```mermaid
graph LR
    A[Source Code] --> B[Testing]
    B --> C[Quality Checks]
    C --> D[Documentation]
    D --> E[Deployment]

    F[uv] --> A
    G[just] --> B
    H[just] --> C
    I[MkDocs] --> D
```

## Why Algorithm Kit?

- **Faster development**: Modern tooling stack reduces setup time
- **Quality by design**: Comprehensive testing and linting from day one
- **Professional standards**: Production-ready CI/CD and documentation
- **Extensible**: Plugin architecture for custom algorithms

## Get Started

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Quickstart](api.md)**

    Get up and running in 10 minutes

-   :material-cog: **[Installation](https://github.com/jeffrichley/algokit#development)**

    Set up your development environment

-   :material-play: **[Testing](https://github.com/jeffrichley/algokit#development)**

    Run the test suite and quality checks

-   :material-cog-outline: **[Configuration](https://github.com/jeffrichley/algokit#development)**

    Configure your development workflow

-   :material-puzzle: **[Contributing](contributing.md)**

    Contribute to the project

-   :material-shield-check: **[Quality](https://github.com/jeffrichley/algokit#development)**

    Maintain high code quality standards

</div>

## Development

For more detailed information about the project architecture, development guidelines, and quality standards, please refer to the project documentation in the main repository.

[:material-code-braces: API Reference](api.md)

Spotted an issue? Edit this page.
