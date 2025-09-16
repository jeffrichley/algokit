# BFS Implementation Plan - HarborNet: Search & Rescue

## Overview
Implementation plan for Breadth-First Search algorithm as part of the Algokit pathfinding suite, following the HarborNet storyworld narrative and reusable visualization framework.

## Phase 1: Foundation & Core Infrastructure
**Goal**: Establish the foundational graph primitives and project structure

### 1.1 Project Structure Setup
- [x] Create directory structure:
  - [x] `src/algokit/pathfinding/` (new module for graph algorithms)
  - [x] `src/algokit/viz/` (new module for visualization framework)
  - [x] `src/algokit/cli/` (new module for CLI integration)
  - [x] `tests/pathfinding/` (new test directory for graph algorithms)
  - [x] `viz/manim/pathfinding/` (new visualization directory for graph algorithms)
  - [x] `data/examples/graphs/` (new data directory for graph files)
  - [x] `data/examples/scenarios/` (new data directory for HarborNet scenarios)

### 1.2 NetworkX Integration & Graph Utilities
- [x] Add NetworkX dependency to project requirements
- [x] Implement `src/algokit/pathfinding/graph_utils.py`:
  - [x] HarborNet scenario loading functions
  - [x] Grid graph generation utilities using NetworkX
  - [x] Graph file I/O (JSON, GraphML, etc.)
  - [x] Graph validation and preprocessing
  - [x] Type hints for all functions
  - [x] Docstrings following Google style

### 1.3 Simple Event System for Data Collection
- [x] Implement `src/algokit/viz/adapters.py`:
  - [x] **Simple SearchEvent** with essential fields (type, node, parent, step, data)
  - [x] **EventType enum** for core algorithm events (ENQUEUE, DEQUEUE, DISCOVER, GOAL_FOUND, PATH_RECONSTRUCT)
  - [x] **SimpleTracker** for lightweight event collection
  - [x] **Generic tracked data structures** (TrackedDeque, TrackedSet, TrackedList) for event emission
  - [x] **Universal context manager** for any algorithm
  - [x] Type hints and validation
  - [x] **Post-processing visualization support** - collect events, render later
  - [x] **Comprehensive test suite** with 8 unit tests covering all functionality

## Phase 2: BFS Algorithm Implementation
**Goal**: Implement the core BFS algorithm with event streaming

### 2.1 Core BFS Implementation
- [x] Implement `src/algokit/pathfinding/bfs.py`:
  - [x] `bfs_shortest_path()` function (pure implementation using NetworkX)
  - [x] `bfs_with_data_collection()` function (using SimpleTracker for post-processing)
  - [x] `bfs_path_length()` function (lightweight path length calculation)
  - [x] `bfs_all_reachable()` function (find all nodes within distance)
  - [x] Proper type hints (NetworkX Graph, Optional, list)
  - [x] Comprehensive docstrings
  - [x] Handle edge cases: start==goal, disconnected graphs
  - [x] Integration with NetworkX graph objects
  - [x] **Keep algorithm logic completely pure** - no event code in core algorithm
  - [x] **Collect events for post-processing visualization** - no real-time rendering needed
  - [x] **Comprehensive test suite** - 17 unit and integration tests covering all functionality

### 2.2 Algorithm Properties & Validation
- [x] Ensure O(|V|+|E|) time complexity
- [x] Validate minimum-hop path guarantee
- [x] Implement proper queue-based frontier management
- [x] Parent pointer reconstruction validation
- [x] **Comprehensive property tests** - 11 tests validating all BFS properties
- [x] **Performance validation** - O(|V|+|E|) complexity confirmed
- [x] **Edge case testing** - disconnected graphs, single nodes, empty graphs
- [x] **Large graph testing** - 10,000+ node graphs perform well

## Phase 3: Comprehensive Testing
**Goal**: Achieve 100% test coverage with edge case validation

### 3.1 Unit Tests
- [x] Implement `tests/pathfinding/test_bfs.py`:
  - [x] `test_bfs_trivial_start_is_goal()` - single node case
  - [x] `test_bfs_simple_path()` - basic pathfinding
  - [x] `test_bfs_disconnected()` - unreachable goal
  - [x] `test_bfs_grid_obstacles()` - grid with blocked cells
  - [x] `test_bfs_multiple_shortest_paths()` - tie-breaking behavior
  - [x] `test_bfs_empty_graph()` - edge case handling
  - [x] `test_bfs_large_sparse_graph()` - performance validation
  - [x] `test_bfs_large_dense_graph()` - memory usage validation
  - [x] `test_bfs_networkx_integration()` - NetworkX graph compatibility

### 3.2 Event Stream Testing
- [x] Test event sequence correctness:
  - [x] `test_bfs_events_enqueue_order()`
  - [x] `test_bfs_events_dequeue_order()`
  - [x] `test_bfs_events_discover_sequence()`
  - [x] `test_bfs_events_goal_found()`
  - [x] `test_bfs_events_reconstruct_path()`
  - [x] `test_bfs_pure_vs_tracked_equivalence()` - ensure both versions produce same results
  - [x] `test_proxy_object_behavior()` - verify TrackedDeque/TrackedSet work correctly

### 3.3 Property-Based Testing
- [x] Implement property tests:
  - [x] Path validity: all returned paths are valid
  - [x] Optimality: returned paths have minimum hop count
  - [x] Completeness: finds path if one exists
  - [x] Determinism: same input produces same output

### 3.4 Test Infrastructure
- [x] Update `tests/conftest.py` with graph fixtures
- [x] Create test data files in `data/examples/`
- [x] Ensure all tests follow AAA pattern with manual comments
- [x] Achieve 100% code coverage

## Phase 4: Visualization Framework
**Goal**: Create reusable Manim visualization system

**Current Status**: ⚠️ **PARTIALLY COMPLETE** - Code structure implemented but not tested
- ✅ **4.1**: Shared scene infrastructure complete
- ⚠️ **4.2**: Event animation helper methods created but untested  
- ⚠️ **4.3**: BFS scene class structure created but untested
- ❌ **4.4**: Visualization testing not started

### 4.1 Shared Scene Infrastructure
- [x] Implement `src/algokit/viz/scenes.py`:
  - [x] `HarborGridScene` base class
  - [x] Grid rendering with proper coordinates
  - [x] Start/goal marker placement
  - [x] Obstacle visualization
  - [x] Color scheme implementation (blue=start, orange=goal, etc.)

### 4.2 Event Animation System
- [x] Implement event-driven animation helper methods:
  - [x] Frontier highlighting (queue visualization) - helper methods created
  - [x] Visited node coloring (pale cyan) - helper methods created
  - [x] Current node pulsing (bright yellow) - helper methods created
  - [x] Edge expansion flashing - helper methods created
  - [x] Path reconstruction animation (thick teal spline) - helper methods created
- [ ] **Test and validate** animation methods with real BFS events
- [ ] **Render** sample visualizations to verify functionality

### 4.3 BFS-Specific Visualization
- [x] Implement `viz/manim/pathfinding/bfs_scene.py` structure:
  - [x] BFS scene driver - class structure created
  - [x] Layer-by-layer frontier visualization - method structure created
  - [x] Queue panel showing live enqueue/dequeue - method structure created
  - [x] HarborNet narrative integration - method structure created
  - [x] Performance metrics display (hop count) - method structure created
- [ ] **Test** BFS scene with real HarborNet scenarios
- [ ] **Render** actual Manim videos to validate functionality

### 4.4 Visualization Testing
- [ ] Create test scenarios:
  - [ ] 3x3 grid with center obstacle
  - [ ] 5x5 grid with multiple obstacles
  - [ ] Simple linear path
  - [ ] Complex maze-like structure
- [ ] **Validate visualization functionality**:
  - [ ] Test `HarborGridScene` with sample scenarios
  - [ ] Test `BFSScene` with real BFS events
  - [ ] Render sample Manim videos to verify animations work
  - [ ] Validate event-driven animations with actual BFS execution
  - [ ] Test queue panel updates and performance metrics display

## Phase 5: CLI Integration
**Goal**: Integrate BFS algorithm into the Algokit CLI system

### 5.1 CLI Command Implementation
- [ ] Implement `src/algokit/cli/bfs.py`:
  - [ ] BFS command with Typer integration
  - [ ] Input file parameter (graph files, HarborNet scenarios)
  - [ ] Output format options (JSON, text, visualization)
  - [ ] Visualization rendering options (MP4, GIF, static images)
  - [ ] Algorithm parameters (start node, goal node, etc.)
  - [ ] Help text and parameter validation

### 5.2 CLI Integration
- [ ] Add BFS command to main CLI registry
- [ ] Update `src/algokit/cli/__init__.py` to include BFS command
- [ ] Test CLI integration with existing Algokit infrastructure
- [ ] Support both interactive and batch execution modes

### 5.3 CLI Testing
- [ ] Test CLI command execution:
  - [ ] `test_cli_bfs_basic_execution()`
  - [ ] `test_cli_bfs_file_input()`
  - [ ] `test_cli_bfs_visualization_output()`
  - [ ] `test_cli_bfs_error_handling()`
  - [ ] `test_cli_bfs_help_text()`

## Phase 6: Documentation & Integration
**Goal**: Complete documentation and integrate with existing system

### 6.1 Algorithm Documentation
- [ ] Create `docs/algorithms/bfs.md`:
  - [ ] HarborNet narrative integration
  - [ ] Algorithm concept explanation
  - [ ] Complexity analysis (O(|V|+|E|))
  - [ ] Correctness proof sketch
  - [ ] Comparison with other algorithms
  - [ ] Failure modes and limitations
  - [ ] Annotated execution examples
  - [ ] **CLI usage examples and parameters**

### 6.2 Visual Documentation
- [ ] Generate visualization assets:
  - [ ] Screenshots from Manim scenes
  - [ ] Animated GIFs for documentation
  - [ ] Step-by-step execution diagrams
  - [ ] Performance comparison charts

### 6.3 Integration Testing
- [ ] Test integration with existing system:
  - [ ] CLI integration validation
  - [ ] Import system validation
  - [ ] Documentation site generation
  - [ ] Example usage validation

### 6.4 Quality Assurance
- [ ] Code quality checks:
  - [ ] MyPy type checking (strict mode)
  - [ ] Ruff linting and formatting
  - [ ] Black code formatting
  - [ ] Xenon complexity analysis
  - [ ] Security audit (pip-audit)

## Phase 7: Performance & Optimization
**Goal**: Validate performance characteristics and optimize if needed

### 7.1 Performance Testing
- [ ] Implement performance benchmarks:
  - [ ] Large graph scalability tests
  - [ ] Memory usage profiling
  - [ ] Execution time validation
  - [ ] Comparison with reference implementations

### 7.2 Optimization (if needed)
- [ ] Profile and optimize bottlenecks:
  - [ ] Data structure efficiency
  - [ ] Memory allocation patterns
  - [ ] Algorithm constant factors

## Phase 8: Extension Preparation
**Goal**: Prepare framework for additional algorithms

### 8.1 Reusable Components
- [ ] Document reusable patterns:
  - [ ] **Universal tracking system** - works for any algorithm type
  - [ ] AlgorithmAdapter interface design
  - [ ] Event stream protocol
  - [ ] Visualization integration points
  - [ ] Testing framework patterns
  - [ ] CLI integration patterns
  - [ ] **Algorithm family patterns** - pathfinding, sorting, search, etc.

### 8.2 Template Creation
- [ ] Create templates for future algorithms:
  - [ ] Algorithm implementation template
  - [ ] Test suite template
  - [ ] Visualization adapter template
  - [ ] CLI command template
  - [ ] Documentation template

## Phase 9: Final Validation & Delivery
**Goal**: Complete validation and prepare for production

### 9.1 End-to-End Testing
- [ ] Full system validation:
  - [ ] Complete test suite execution
  - [ ] Visualization rendering validation
  - [ ] Documentation generation
  - [ ] Example execution
  - [ ] CLI command execution

### 9.2 User Experience Testing
- [ ] Validate user-facing components:
  - [ ] API usability
  - [ ] CLI usability and help text
  - [ ] Documentation clarity
  - [ ] Example effectiveness
  - [ ] Error message quality

### 9.3 Delivery Preparation
- [ ] Prepare deliverables:
  - [ ] Code review preparation
  - [ ] Documentation review
  - [ ] Performance benchmarks
  - [ ] Integration guide
  - [ ] CLI usage guide

## Success Criteria

### Technical Requirements
- [x] 100% test coverage for BFS implementation
- [x] All tests pass with strict type checking
- [ ] Visualization renders correctly for all test cases
- [ ] Documentation is complete and accurate
- [x] Performance meets O(|V|+|E|) complexity requirements

### Quality Requirements
- [x] Code follows project style guidelines
- [x] All functions have proper type hints
- [x] All public APIs have docstrings
- [x] No security vulnerabilities
- [x] Complexity within acceptable limits

### User Experience Requirements
- [x] Clear and intuitive API
- [ ] **Functional CLI interface with helpful commands**
- [ ] Comprehensive documentation with examples
- [ ] Effective visualizations
- [x] Easy integration with existing code

## Dependencies & Prerequisites

### External Dependencies
- [x] **NetworkX** (for graph operations and data structures)
- [x] Manim (for visualizations)
- [x] PyTest (for testing)
- [x] MyPy (for type checking)
- [x] Ruff (for linting)
- [x] Black (for formatting)
- [ ] Typer (for CLI integration)

### Internal Dependencies
- [ ] Existing Algokit project structure
- [ ] Documentation system (MkDocs)
- [ ] CI/CD pipeline configuration

## Risk Mitigation

### Technical Risks
- **Manim complexity**: Start with simple visualizations, iterate
- **Performance issues**: Profile early and often
- **Type system complexity**: Use strict typing from the start

### Project Risks
- **Scope creep**: Stick to BFS implementation, defer other algorithms
- **Documentation debt**: Write docs alongside code
- **Testing gaps**: Implement tests before features

## Timeline Estimate
- **Phase 1-2**: 2-3 days (Foundation + Core Implementation)
- **Phase 3**: 2-3 days (Comprehensive Testing)
- **Phase 4**: 3-4 days (Visualization Framework)
- **Phase 5**: 2-3 days (CLI Integration)
- **Phase 6**: 2-3 days (Documentation)
- **Phase 7-9**: 2-3 days (Optimization + Delivery)

**Total Estimated Time**: 13-19 days

## Current Status Summary

### ✅ **COMPLETED PHASES**
- **Phase 1**: Foundation & Core Infrastructure (100% complete)
- **Phase 2**: BFS Algorithm Implementation (100% complete)  
- **Phase 3**: Comprehensive Testing (100% complete)

### ⚠️ **IN PROGRESS**
- **Phase 4**: Visualization Framework (60% complete)
  - ✅ Code structure implemented
  - ❌ Testing and validation needed

### ❌ **NOT STARTED**
- **Phase 5**: CLI Integration
- **Phase 6**: Documentation & Integration
- **Phase 7**: Performance & Optimization
- **Phase 8**: Extension Preparation
- **Phase 9**: Final Validation & Delivery

## Next Steps After Completion
1. Gather feedback on BFS implementation
2. Refine visualization framework based on usage
3. Begin DFS implementation using established patterns
4. Expand HarborNet storyworld with additional scenarios
5. Create algorithm comparison visualizations
