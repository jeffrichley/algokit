"""Tests for M* multi-robot path planning algorithm.

This module contains comprehensive tests for the M* algorithm implementation.
"""

import networkx as nx
import pytest

from algokit.algorithms.pathfinding.mstar import mstar_plan_paths
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
