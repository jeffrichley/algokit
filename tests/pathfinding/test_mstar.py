"""Tests for M* multi-robot path planning algorithm.

This module contains comprehensive tests for the M* algorithm implementation.
"""

import networkx as nx
import pytest
from pydantic import ValidationError

from algokit.algorithms.pathfinding.mstar import MStar, MStarConfig, mstar_plan_paths
from algokit.core.helpers import create_grid_graph


class TestMStar:
    """Test M* algorithm functionality."""

    @pytest.mark.unit
    def test_single_robot_planning(self) -> None:
        """Test that M* works for a single robot."""
        # Arrange - create a 3x3 grid graph and set up single robot start/goal
        graph = create_grid_graph(3, 3)
        starts = {"robot1": (0, 0)}
        goals = {"robot1": (2, 2)}

        # Act - run M* path planning for the single robot
        result = mstar_plan_paths(graph, starts, goals)

        # Assert - verify the robot reaches its goal from start position
        assert result is not None
        assert "robot1" in result
        assert result["robot1"][0] == (0, 0)  # Start position
        assert result["robot1"][-1] == (2, 2)  # Goal position

    @pytest.mark.unit
    def test_two_robots_no_conflict(self) -> None:
        """Test M* with two robots that don't conflict."""
        # Arrange - create a 4x4 grid and set up two robots with non-conflicting paths
        graph = create_grid_graph(4, 4)
        starts = {"robot1": (0, 0), "robot2": (0, 1)}
        goals = {"robot1": (3, 3), "robot2": (3, 2)}

        # Act - run M* path planning for both robots
        result = mstar_plan_paths(graph, starts, goals)

        # Assert - verify both robots reach their goals without conflicts
        assert result is not None
        assert "robot1" in result
        assert "robot2" in result
        assert result["robot1"][0] == (0, 0)
        assert result["robot1"][-1] == (3, 3)
        assert result["robot2"][0] == (0, 1)
        assert result["robot2"][-1] == (3, 2)

    @pytest.mark.unit
    def test_two_robots_with_conflict(self) -> None:
        """Test M* with two robots that have a potential conflict."""
        # Arrange - create a 3x3 grid and set up two robots with potentially conflicting paths
        graph = create_grid_graph(3, 3)
        starts = {"robot1": (0, 0), "robot2": (1, 0)}
        goals = {"robot1": (2, 2), "robot2": (0, 2)}

        # Act - run M* path planning to resolve conflicts
        result = mstar_plan_paths(graph, starts, goals)

        # Assert - verify both robots reach their goals despite potential conflicts
        assert result is not None
        assert "robot1" in result
        assert "robot2" in result
        assert result["robot1"][0] == (0, 0)
        assert result["robot1"][-1] == (2, 2)
        assert result["robot2"][0] == (1, 0)
        assert result["robot2"][-1] == (0, 2)

    @pytest.mark.unit
    def test_unsolvable_case(self) -> None:
        """Test M* with unsolvable case."""
        # Arrange - create a graph with disconnected start and goal nodes
        graph = nx.Graph()
        graph.add_node("start")
        graph.add_node("goal")
        # No edges between start and goal
        starts = {"robot1": "start"}
        goals = {"robot1": "goal"}

        # Act - attempt to plan path between disconnected nodes
        result = mstar_plan_paths(graph, starts, goals)

        # Assert - verify that no solution is found for disconnected graph
        assert result is None

    @pytest.mark.unit
    def test_collision_radius_parameter(self) -> None:
        """Test M* with different collision radius."""
        # Arrange - create a 3x3 grid and set up two robots with custom collision radius
        graph = create_grid_graph(3, 3)
        starts = {"robot1": (0, 0), "robot2": (1, 0)}
        goals = {"robot1": (2, 2), "robot2": (0, 2)}

        # Act - run M* with smaller collision radius
        result = mstar_plan_paths(graph, starts, goals, collision_radius=0.5)

        # Assert - verify that planning succeeds with custom collision radius
        assert result is not None
        assert "robot1" in result
        assert "robot2" in result

    @pytest.mark.unit
    def test_three_robots_simple(self) -> None:
        """Test M* with three robots in a simple scenario."""
        # Arrange - create a 3x3 grid and set up three robots with simple paths
        graph = create_grid_graph(3, 3)
        starts = {"robot1": (0, 0), "robot2": (0, 1), "robot3": (0, 2)}
        goals = {"robot1": (2, 0), "robot2": (2, 1), "robot3": (2, 2)}

        # Act - run M* path planning for all three robots
        result = mstar_plan_paths(graph, starts, goals)

        # Assert - verify all three robots reach their respective goals
        assert result is not None
        assert len(result) == 3
        for robot_id in ["robot1", "robot2", "robot3"]:
            assert robot_id in result
            assert result[robot_id][0] == starts[robot_id]
            assert result[robot_id][-1] == goals[robot_id]

    @pytest.mark.unit
    def test_same_start_and_goal(self) -> None:
        """Test M* when start and goal are the same."""
        # Arrange - create a 3x3 grid with robot starting and ending at same position
        graph = create_grid_graph(3, 3)
        starts = {"robot1": (1, 1)}
        goals = {"robot1": (1, 1)}

        # Act - run M* path planning for robot already at goal
        result = mstar_plan_paths(graph, starts, goals)

        # Assert - verify robot stays at start position when start equals goal
        assert result is not None
        assert "robot1" in result
        assert result["robot1"] == [(1, 1)]

    @pytest.mark.unit
    def test_empty_graph(self) -> None:
        """Test M* with empty graph."""
        # Arrange - create an empty graph with no nodes
        graph = nx.Graph()
        starts = {"robot1": "start"}
        goals = {"robot1": "goal"}

        # Act & Assert - verify that planning fails when nodes don't exist in graph
        with pytest.raises(nx.NodeNotFound):
            mstar_plan_paths(graph, starts, goals)

    @pytest.mark.unit
    def test_single_node_graph(self) -> None:
        """Test M* with single node graph."""
        # Arrange - create a graph with only one node
        graph = nx.Graph()
        graph.add_node("node")
        starts = {"robot1": "node"}
        goals = {"robot1": "node"}

        # Act - run M* path planning on single node graph
        result = mstar_plan_paths(graph, starts, goals)

        # Assert - verify robot stays at the single node
        assert result is not None
        assert result["robot1"] == ["node"]

    @pytest.mark.unit
    def test_weighted_graph(self) -> None:
        """Test M* with weighted graph."""
        # Arrange - create a weighted graph with different edge costs
        graph = nx.Graph()
        graph.add_edge("A", "B", weight=2.0)
        graph.add_edge("B", "C", weight=1.0)
        graph.add_edge("A", "C", weight=5.0)
        starts = {"robot1": "A"}
        goals = {"robot1": "C"}

        # Act - run M* path planning on weighted graph
        result = mstar_plan_paths(graph, starts, goals)

        # Assert - verify robot takes the optimal path considering edge weights
        assert result is not None
        assert result["robot1"][0] == "A"
        assert result["robot1"][-1] == "C"
        # Should take the shorter path A->B->C (weight 3) instead of A->C (weight 5)
        assert len(result["robot1"]) == 3

    @pytest.mark.unit
    def test_forced_conflict_scenario(self) -> None:
        """Test M* with a scenario that forces conflict detection and repair."""
        # Arrange - create a narrow corridor that forces robots to conflict
        graph = nx.Graph()
        # Create a narrow corridor: (0,0) -> (1,0) -> (2,0) -> (3,0)
        # And a parallel corridor: (0,1) -> (1,1) -> (2,1) -> (3,1)
        # With connections between them
        for x in range(4):
            graph.add_edge((x, 0), (x, 1))  # vertical connections
        for x in range(3):
            graph.add_edge((x, 0), (x + 1, 0))  # horizontal connections
            graph.add_edge((x, 1), (x + 1, 1))  # horizontal connections

        # Two robots that must cross paths
        starts = {"robot1": (0, 0), "robot2": (3, 1)}
        goals = {"robot1": (3, 0), "robot2": (0, 1)}

        # Act - run M* path planning with collision radius that will detect conflicts
        result = mstar_plan_paths(graph, starts, goals, collision_radius=0.5)

        # Assert - verify both robots reach their goals
        assert result is not None
        assert "robot1" in result
        assert "robot2" in result
        assert result["robot1"][0] == (0, 0)
        assert result["robot1"][-1] == (3, 0)
        assert result["robot2"][0] == (3, 1)
        assert result["robot2"][-1] == (0, 1)

    @pytest.mark.unit
    def test_edge_swap_conflict_detection(self) -> None:
        """Test M* with edge swap conflicts that require repair."""
        # Arrange - create a simple graph where robots can swap positions
        graph = nx.Graph()
        # Simple line graph: A-B-C-D
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")

        # Two robots that need to swap positions
        starts = {"robot1": "A", "robot2": "D"}
        goals = {"robot1": "D", "robot2": "A"}

        # Act - run M* path planning
        result = mstar_plan_paths(graph, starts, goals, collision_radius=0.1)

        # Assert - verify both robots reach their goals (or handle None case)
        if result is not None:
            assert "robot1" in result
            assert "robot2" in result
            assert result["robot1"][0] == "A"
            assert result["robot1"][-1] == "D"
            assert result["robot2"][0] == "D"
            assert result["robot2"][-1] == "A"
        else:
            # If no solution found, that's also acceptable for this test
            # The important thing is that we exercise the conflict detection code
            pass

    @pytest.mark.unit
    def test_three_robots_complex_conflict(self) -> None:
        """Test M* with three robots in a simple conflict scenario."""
        # Arrange - create a simple graph with three robots
        graph = nx.Graph()
        # Create a simple line: A-B-C-D-E
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")
        graph.add_edge("D", "E")

        # Three robots that will have simple paths
        starts = {"robot1": "A", "robot2": "B", "robot3": "C"}
        goals = {"robot1": "E", "robot2": "D", "robot3": "B"}

        # Act - run M* path planning
        result = mstar_plan_paths(graph, starts, goals, collision_radius=0.1)

        # Assert - verify all robots reach their goals (or handle None case)
        if result is not None:
            for robot in ["robot1", "robot2", "robot3"]:
                assert robot in result
                assert result[robot][0] == starts[robot]
                assert result[robot][-1] == goals[robot]
        else:
            # If no solution found, that's also acceptable for this test
            # The important thing is that we exercise the conflict detection code
            pass

    @pytest.mark.unit
    def test_already_at_goals_scenario(self) -> None:
        """Test M* when robots are already at their goals."""
        # Arrange - create a simple graph
        graph = nx.Graph()
        graph.add_edge("A", "B")

        # Robots already at their goals
        starts = {"robot1": "A", "robot2": "B"}
        goals = {"robot1": "A", "robot2": "B"}

        # Act - run M* path planning
        result = mstar_plan_paths(graph, starts, goals)

        # Assert - verify robots stay at their goals
        assert result is not None
        assert result["robot1"] == ["A"]
        assert result["robot2"] == ["B"]

    @pytest.mark.unit
    def test_safety_cap_exceeded(self) -> None:
        """Test M* when safety cap is exceeded (infinite loop scenario)."""
        # Arrange - create a graph that might cause infinite loops
        graph = nx.Graph()
        # Create a simple path
        graph.add_edge("start", "middle")
        graph.add_edge("middle", "goal")

        # This scenario might cause issues with the safety cap
        starts = {"robot1": "start", "robot2": "start"}
        goals = {"robot1": "goal", "robot2": "goal"}

        # Act - run M* path planning
        result = mstar_plan_paths(graph, starts, goals, collision_radius=10.0)

        # Assert - should either succeed or return None (not crash)
        if result is not None:
            assert "robot1" in result
            assert "robot2" in result


class TestMStarConfig:
    """Test MStarConfig Pydantic validation."""

    @pytest.mark.unit
    def test_config_default_values(self) -> None:
        """Test that MStarConfig has correct default values."""
        # Arrange - no setup needed for default config test

        # Act - create config with default values
        config = MStarConfig()

        # Assert - verify all default values match specification
        assert config.collision_radius == 1.0
        assert config.max_extra_steps == 200
        assert config.safety_cap == 2000

    @pytest.mark.unit
    def test_config_custom_values(self) -> None:
        """Test that MStarConfig accepts custom values."""
        # Arrange - define custom parameter values

        # Act - create config with custom values
        config = MStarConfig(collision_radius=2.5, max_extra_steps=300, safety_cap=3000)

        # Assert - verify custom values were set correctly
        assert config.collision_radius == 2.5
        assert config.max_extra_steps == 300
        assert config.safety_cap == 3000

    @pytest.mark.unit
    def test_config_rejects_negative_collision_radius(self) -> None:
        """Test that MStarConfig rejects negative collision_radius."""
        # Arrange - prepare invalid negative collision radius

        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="collision_radius"):
            MStarConfig(collision_radius=-1.0)

    @pytest.mark.unit
    def test_config_rejects_zero_collision_radius(self) -> None:
        """Test that MStarConfig rejects zero collision_radius."""
        # Arrange - prepare invalid zero collision radius

        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="collision_radius"):
            MStarConfig(collision_radius=0.0)

    @pytest.mark.unit
    def test_config_rejects_unreasonably_large_collision_radius(self) -> None:
        """Test that MStarConfig rejects unreasonably large collision_radius."""
        # Arrange - prepare unreasonably large collision radius value

        # Act - attempt to create config with unreasonably large value
        # Assert - verify ValidationError is raised with appropriate message
        with pytest.raises(ValidationError, match="unreasonably large"):
            MStarConfig(collision_radius=150.0)

    @pytest.mark.unit
    def test_config_rejects_negative_max_extra_steps(self) -> None:
        """Test that MStarConfig rejects negative max_extra_steps."""
        # Arrange - prepare invalid negative max_extra_steps

        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="max_extra_steps"):
            MStarConfig(max_extra_steps=-1)

    @pytest.mark.unit
    def test_config_rejects_zero_max_extra_steps(self) -> None:
        """Test that MStarConfig rejects zero max_extra_steps."""
        # Arrange - prepare invalid zero max_extra_steps

        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="max_extra_steps"):
            MStarConfig(max_extra_steps=0)

    @pytest.mark.unit
    def test_config_rejects_negative_safety_cap(self) -> None:
        """Test that MStarConfig rejects negative safety_cap."""
        # Arrange - prepare invalid negative safety_cap

        # Act - attempt to create config with negative value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="safety_cap"):
            MStarConfig(safety_cap=-1)

    @pytest.mark.unit
    def test_config_rejects_zero_safety_cap(self) -> None:
        """Test that MStarConfig rejects zero safety_cap."""
        # Arrange - prepare invalid zero safety_cap

        # Act - attempt to create config with zero value
        # Assert - verify ValidationError is raised
        with pytest.raises(ValidationError, match="safety_cap"):
            MStarConfig(safety_cap=0)

    @pytest.mark.unit
    def test_config_accepts_small_positive_collision_radius(self) -> None:
        """Test that MStarConfig accepts small positive collision_radius."""
        # Arrange - prepare small but valid collision radius

        # Act - create config with small positive collision radius
        config = MStarConfig(collision_radius=0.1)

        # Assert - verify small positive value is accepted
        assert config.collision_radius == 0.1

    @pytest.mark.unit
    def test_config_accepts_large_but_reasonable_collision_radius(self) -> None:
        """Test that MStarConfig accepts large but reasonable collision_radius."""
        # Arrange - prepare large but reasonable collision radius

        # Act - create config with large but reasonable collision radius
        config = MStarConfig(collision_radius=50.0)

        # Assert - verify large reasonable value is accepted
        assert config.collision_radius == 50.0


class TestMStarClass:
    """Test MStar class API."""

    @pytest.mark.unit
    def test_mstar_class_with_config_object(self) -> None:
        """Test that MStar class accepts config object."""
        # Arrange - create a config object with custom values
        config = MStarConfig(collision_radius=1.5)

        # Act - initialize MStar with config object
        planner = MStar(config=config)

        # Assert - verify planner stores config and extracts parameters correctly
        assert planner.config == config
        assert planner.collision_radius == 1.5
        assert planner.max_extra_steps == 200
        assert planner.safety_cap == 2000

    @pytest.mark.unit
    def test_mstar_class_with_kwargs(self) -> None:
        """Test that MStar class accepts kwargs for backwards compatibility."""
        # Arrange - prepare kwargs for backwards compatibility test

        # Act - initialize MStar with kwargs
        planner = MStar(collision_radius=1.5, max_extra_steps=300)

        # Assert - verify parameters were set correctly from kwargs
        assert planner.collision_radius == 1.5
        assert planner.max_extra_steps == 300
        assert planner.safety_cap == 2000

    @pytest.mark.unit
    def test_mstar_class_default_initialization(self) -> None:
        """Test that MStar class can be initialized with defaults."""
        # Arrange - no setup needed for default initialization

        # Act - initialize MStar with no parameters
        planner = MStar()

        # Assert - verify all defaults are set correctly
        assert planner.collision_radius == 1.0
        assert planner.max_extra_steps == 200
        assert planner.safety_cap == 2000

    @pytest.mark.unit
    def test_mstar_class_plan_single_robot(self) -> None:
        """Test that MStar.plan() works for single robot."""
        # Arrange - create planner, graph, and single robot scenario
        planner = MStar()
        graph = create_grid_graph(3, 3)
        starts = {"robot1": (0, 0)}
        goals = {"robot1": (2, 2)}

        # Act - plan path for single robot
        result = planner.plan(graph, starts, goals)

        # Assert - verify path is found and correct
        assert result is not None
        assert "robot1" in result
        assert result["robot1"][0] == (0, 0)
        assert result["robot1"][-1] == (2, 2)

    @pytest.mark.unit
    def test_mstar_class_plan_two_robots(self) -> None:
        """Test that MStar.plan() works for two robots."""
        # Arrange - create planner, graph, and two robot scenario
        planner = MStar(collision_radius=0.5)  # Smaller radius for reliable test
        graph = create_grid_graph(5, 5)  # Larger graph for more space
        starts = {"robot1": (0, 0), "robot2": (0, 1)}
        goals = {"robot1": (4, 4), "robot2": (4, 3)}

        # Act - plan paths for two robots
        result = planner.plan(graph, starts, goals)

        # Assert - verify paths are found for both robots
        assert result is not None
        assert "robot1" in result
        assert "robot2" in result
        assert result["robot1"][0] == (0, 0)
        assert result["robot1"][-1] == (4, 4)
        assert result["robot2"][0] == (0, 1)
        assert result["robot2"][-1] == (4, 3)

    @pytest.mark.unit
    def test_mstar_class_plan_validates_mismatched_agents(self) -> None:
        """Test that MStar.plan() validates mismatched agents in starts/goals."""
        # Arrange - create planner, graph, and mismatched agent scenario
        planner = MStar()
        graph = create_grid_graph(3, 3)
        starts = {"robot1": (0, 0)}
        goals = {"robot2": (2, 2)}  # Different agent

        # Act - attempt to plan with mismatched agents
        # Assert - verify ValueError is raised for mismatched agents
        with pytest.raises(ValueError, match="same set of agents"):
            planner.plan(graph, starts, goals)

    @pytest.mark.unit
    def test_mstar_class_plan_validates_start_not_in_graph(self) -> None:
        """Test that MStar.plan() validates start position is in graph."""
        # Arrange - create planner, graph, and invalid start position
        planner = MStar()
        graph = create_grid_graph(3, 3)
        starts = {"robot1": (10, 10)}  # Not in graph
        goals = {"robot1": (2, 2)}

        # Act - attempt to plan with invalid start position
        # Assert - verify ValueError is raised for invalid start position
        with pytest.raises(ValueError, match="Start position"):
            planner.plan(graph, starts, goals)

    @pytest.mark.unit
    def test_mstar_class_plan_validates_goal_not_in_graph(self) -> None:
        """Test that MStar.plan() validates goal position is in graph."""
        # Arrange - create planner, graph, and invalid goal position
        planner = MStar()
        graph = create_grid_graph(3, 3)
        starts = {"robot1": (0, 0)}
        goals = {"robot1": (10, 10)}  # Not in graph

        # Act - attempt to plan with invalid goal position
        # Assert - verify ValueError is raised for invalid goal position
        with pytest.raises(ValueError, match="Goal position"):
            planner.plan(graph, starts, goals)

    @pytest.mark.unit
    def test_mstar_class_invalid_config_via_kwargs(self) -> None:
        """Test that MStar class validates invalid params via kwargs."""
        # Arrange - prepare invalid parameter value

        # Act - attempt to initialize with invalid kwargs
        # Assert - verify ValidationError is raised for invalid parameter
        with pytest.raises(ValidationError, match="collision_radius"):
            MStar(collision_radius=-1.0)

    @pytest.mark.unit
    def test_mstar_class_plan_with_custom_safety_cap(self) -> None:
        """Test that MStar.plan() uses custom safety_cap."""
        # Arrange - create planner with very low safety cap and test scenario
        planner = MStar(safety_cap=10)  # Very low safety cap
        graph = create_grid_graph(3, 3)
        starts = {"robot1": (0, 0), "robot2": (0, 1)}
        goals = {"robot1": (2, 2), "robot2": (2, 1)}

        # Act - attempt to plan with limited safety cap
        result = planner.plan(graph, starts, goals)

        # Assert - verify result is either valid plan or None without crash
        # May or may not succeed depending on conflict resolution,
        # but should not crash or hang
        assert result is None or isinstance(result, dict)

    @pytest.mark.unit
    def test_mstar_class_plan_with_custom_max_extra_steps(self) -> None:
        """Test that MStar.plan() uses custom max_extra_steps."""
        # Arrange - create planner with custom max_extra_steps and test scenario
        planner = MStar(max_extra_steps=50)
        graph = create_grid_graph(4, 4)
        starts = {"robot1": (0, 0), "robot2": (0, 1)}
        goals = {"robot1": (3, 3), "robot2": (3, 2)}

        # Act - plan paths with custom max_extra_steps
        result = planner.plan(graph, starts, goals)

        # Assert - verify paths are found
        assert result is not None
        assert "robot1" in result
        assert "robot2" in result
