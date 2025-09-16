Here’s a single polished Markdown doc that combines your workflow + directory structure, and adds a reusable AI coding agent prompt with a detailed checklist.

---

# Algorithm Demo Workflow

## Standard Workflow for Each Algorithm

1. **Algorithm Implementation**

   * Write a clean, idiomatic Python implementation of the algorithm.
   * Include type hints and docstrings.
   * Keep code modular (e.g., core function vs. helper functions).
   * Add complexity notes (time/space analysis in comments or docstring).
   * **Use NetworkX for graph operations** - leverage mature, tested graph library.
   * **Implement event streaming** - emit SearchEvent objects for visualization decoupling.

2. **Testing the Algorithm**

   * Write unit tests using `pytest`.
   * Cover both correctness (expected outputs) and edge cases.
   * Add property-based tests (e.g., with Hypothesis) where meaningful.
   * Validate performance on small/medium input sizes.
   * **Group tests by algorithm family** - organize in subdirectories (e.g., `tests/pathfinding/`, `tests/sorting/`).

3. **Visualization**

   * Use **Manim** (or similar) to animate the algorithm step by step.
   * **Consume event streams** from algorithm implementations.
   * Show:

     * Initial state
     * Iterative process (comparisons, swaps, recursion, etc.)
     * Final result
   * Export both a video and (if possible) an interactive notebook version.
   * Keep animation style consistent across all algorithms.

4. **CLI Integration**

   * **Plumb algorithm into CLI run system** - make it executable via command line.
   * Add algorithm-specific CLI parameters (input files, output formats, visualization options).
   * Integrate with existing Algokit CLI infrastructure.
   * Support both interactive and batch execution modes.

5. **Documentation**

   * Write a **Markdown or Jupyter Notebook** covering:

     * Algorithm description
     * Family of algorithms (e.g., sorting, graph traversal)
     * Big-O analysis (time, space, best/average/worst case)
     * Applications and variants
     * Links to visualization video
     * **CLI usage examples and parameters**
   * Include input/output examples inline.

---

## Recommended Directory Layout

```
project-root/
├── src/algokit/        # Algorithm source code (organized by family)
│   ├── pathfinding/    # Graph algorithms (BFS, DFS, Dijkstra, A*)
│   ├── sorting/        # Sorting algorithms
│   ├── viz/            # Reusable visualization framework
│   │   ├── adapters.py # Algorithm-to-viz adapters
│   │   └── scenes.py   # Shared Manim scene classes
│   └── cli/            # CLI integration modules
├── tests/              # Unit tests (organized by family)
│   ├── pathfinding/    # Graph algorithm tests
│   ├── sorting/        # Sorting algorithm tests
│   └── integration/    # Cross-module tests
├── viz/manim/          # Concrete Manim scene files
│   ├── pathfinding/    # Graph algorithm visualizations
│   └── sorting/        # Sorting algorithm visualizations
├── data/examples/      # Test data and example graphs
│   ├── graphs/         # Graph files (JSON, GraphML, etc.)
│   └── scenarios/      # HarborNet scenario files
├── docs/               # Explanatory markdown or notebooks
│   └── algorithms/     # Algorithm documentation
└── README.md           # Project overview
```

This structure cleanly separates code, tests, visuals, and documentation by algorithm family. It makes automation (e.g., build scripts) easier and keeps each algorithm demo consistent while supporting the CLI integration and event streaming architecture.

---

## AI Coding Agent Prompt

You can paste this into your coding agent every time you start a new algorithm.

```
You are an expert AI coding assistant helping me build a complete demo for an algorithm. 
Follow this checklist carefully and produce the deliverables in the correct folders.

### Checklist
1. **Algorithm Implementation**
   - Implement the algorithm in `src/algokit/<family>/<algorithm>.py`.
   - Use **NetworkX** for graph operations and data structures.
   - Implement **event streaming** - emit SearchEvent objects for visualization.
   - Use clean, idiomatic Python with type hints and docstrings.
   - Include inline notes on time/space complexity.

2. **Testing**
   - Write `pytest` tests in `tests/<family>/test_<algorithm>.py`.
   - Cover correctness, edge cases, and (if applicable) randomized/property tests.
   - Test both algorithm logic and event stream generation.

3. **Visualization**
   - Create reusable visualization components in `src/algokit/viz/`.
   - Create concrete Manim scene in `viz/manim/<family>/<algorithm>_scene.py`.
   - Consume event streams from algorithm implementations.
   - Show algorithm progression step by step.
   - Export as MP4 (and optional notebook embed).

4. **CLI Integration**
   - Add CLI command in `src/algokit/cli/` for the algorithm.
   - Support input file loading, output formats, and visualization options.
   - Integrate with existing Algokit CLI infrastructure.
   - Add algorithm to main CLI command registry.

5. **Documentation**
   - Write `docs/algorithms/<algorithm>.md`.
   - Include:
     - Description of the algorithm and its family.
     - Complexity analysis (Big-O).
     - Example inputs and outputs.
     - Applications and variants.
     - Reference to the visualization.
     - **CLI usage examples and parameters**.

6. **Data & Examples**
   - Create example data files in `data/examples/<family>/`.
   - Include test graphs and HarborNet scenario files.
   - Provide both small examples and benchmark datasets.

7. **Consistency & Style**
   - Follow the existing project folder structure by algorithm family.
   - Ensure naming conventions are consistent.
   - Code should pass linting and typing checks.

---

### Algorithm-Specific Instructions
(Paste detailed instructions for this algorithm here, e.g., implementation nuances, visualization ideas, benchmarks, etc.)
```
