"""M* multi-robot path planning algorithm with subdimensional expansion.

This module implements the M* algorithm, an efficient multi-robot path planning
algorithm that uses subdimensional expansion to avoid the exponential complexity
of traditional multi-robot planning approaches.

Key Concepts:
- Subdimensional Expansion: Robots plan independently until conflicts are detected
- Conflict Detection: Identifies vertex collisions and edge swaps between robots
- Coupling: When conflicts occur, robots are coupled and planned together
- Decoupling: Robots can be decoupled when conflicts are resolved

The algorithm maintains individual robot paths and only couples robots when
necessary, making it much more efficient than naive approaches that consider
all robots simultaneously.

Time Complexity: O(V + E) per robot in the best case (no conflicts)
Space Complexity: O(V) per robot
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Hashable
from typing import Any

import networkx as nx

Pos = Hashable  # a node in the graph (e.g., tuple[int,int] or any hashable)
Agent = Hashable  # e.g., "robot1"
Path = list[Pos]
Plan = dict[Agent, Path]


# --------------------------- Utilities ---------------------------


def edge_weight(G: nx.Graph, u: Pos, v: Pos) -> float:
    """Get the weight of an edge between two positions.

    Args:
        G: NetworkX graph containing the edge weights
        u: Source position
        v: Target position

    Returns:
        Edge weight (default 1.0 for unweighted edges)

    Raises:
        KeyError: If no edge exists between u and v (except self-loops)
    """
    data = G.get_edge_data(u, v, default=None)
    if data is None:
        # Waiting at a node (self-loop) is not in the graph; treat as cost 1
        if u == v:
            return 1.0
        raise KeyError(f"No edge between {u} and {v}")
    return float(data.get("weight", 1.0))


def euclid(a: Any, b: Any) -> float | None:
    """Calculate Euclidean distance between two positions if they are numeric tuples.

    This function attempts to compute the Euclidean distance between two positions
    if they can be interpreted as numeric coordinates (tuples or lists of numbers).

    Args:
        a: First position (tuple/list of numbers or any type)
        b: Second position (tuple/list of numbers or any type)

    Returns:
        Euclidean distance if both positions are numeric tuples/lists, None otherwise

    Example:
        >>> euclid((0, 0), (3, 4))
        5.0
        >>> euclid("node1", "node2")
        None
    """
    try:
        if isinstance(a, tuple | list) and isinstance(b, tuple | list):
            return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    except Exception:
        pass
    return None


def has_vertex_collision(p1: Pos, p2: Pos, collision_radius: float) -> bool:
    """Check if two positions represent a vertex collision.

    A vertex collision occurs when two robots occupy the same position or
    are within the collision radius of each other.

    Args:
        p1: First robot's position
        p2: Second robot's position
        collision_radius: Minimum safe distance between robots

    Returns:
        True if there is a vertex collision, False otherwise

    Example:
        >>> has_vertex_collision((0, 0), (0, 0), 1.0)
        True
        >>> has_vertex_collision((0, 0), (2, 0), 1.0)
        False
    """
    if p1 == p2:
        return True
    d = euclid(p1, p2)
    if d is None:
        return False
    return d < collision_radius


def has_edge_swap(prev1: Pos, next1: Pos, prev2: Pos, next2: Pos) -> bool:
    """Check if two robots are performing an edge swap collision.

    An edge swap occurs when two robots cross paths by moving in opposite
    directions along the same edge at the same time step.

    Args:
        prev1: First robot's previous position
        next1: First robot's next position
        prev2: Second robot's previous position
        next2: Second robot's next position

    Returns:
        True if an edge swap collision occurs, False otherwise

    Example:
        >>> has_edge_swap((0, 0), (1, 0), (1, 0), (0, 0))
        True  # Robots crossing paths
        >>> has_edge_swap((0, 0), (1, 0), (0, 1), (1, 1))
        False  # No collision
    """
    return prev1 == next2 and prev2 == next1


def pad_to_length(path: Path, L: int) -> Path:
    """Pad a path to a specified length by repeating the last position.

    This is useful for synchronizing paths of different lengths by making
    shorter paths wait at their final position.

    Args:
        path: The path to pad
        L: Target length for the path

    Returns:
        Padded path of length L (or original path if already longer)

    Example:
        >>> pad_to_length([(0, 0), (1, 0)], 4)
        [(0, 0), (1, 0), (1, 0), (1, 0)]
    """
    if not path:
        return path
    if len(path) >= L:
        return path
    return path + [path[-1]] * (L - len(path))


def neighbors_with_wait(G: nx.Graph, u: Pos) -> list[Pos]:
    """Get all possible next positions including waiting at current position.

    This function returns all neighbors of a position plus the position itself,
    allowing robots to wait at their current location.

    Args:
        G: NetworkX graph
        u: Current position

    Returns:
        List of possible next positions (including current position for waiting)

    Example:
        >>> G = nx.grid_2d_graph(3, 3)
        >>> neighbors_with_wait(G, (1, 1))
        [(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)]
    """
    return [u] + list(G.neighbors(u))


def single_source_to_goal_costs(G: nx.Graph, goal: Pos) -> dict[Pos, float]:
    """Compute admissible heuristic costs from all nodes to the goal.

    This function precomputes the shortest path costs from every node to the goal,
    which serves as an admissible heuristic for A* search. Since the graph is
    undirected, we can run Dijkstra from the goal to all other nodes.

    Args:
        G: NetworkX graph (should be undirected)
        goal: Target goal position

    Returns:
        Dictionary mapping each position to its shortest cost to the goal

    Example:
        >>> G = nx.grid_2d_graph(3, 3)
        >>> costs = single_source_to_goal_costs(G, (2, 2))
        >>> costs[(0, 0)]
        4.0  # Manhattan distance from (0,0) to (2,2)
    """
    # Run Dijkstra from goal to all nodes; edge weights are symmetric for undirected graphs
    return nx.single_source_dijkstra_path_length(G, goal, weight="weight")


def reconstruct_path(came_from: dict[Pos, Pos], start: Pos, goal: Pos) -> Path:
    """Reconstruct the path from start to goal using parent pointers.

    This function traces back through the parent pointers to reconstruct
    the complete path from start to goal.

    Args:
        came_from: Dictionary mapping each position to its parent position
        start: Starting position
        goal: Goal position

    Returns:
        Complete path from start to goal as a list of positions

    Example:
        >>> came_from = {(1, 0): (0, 0), (2, 0): (1, 0)}
        >>> reconstruct_path(came_from, (0, 0), (2, 0))
        [(0, 0), (1, 0), (2, 0)]
    """
    cur = goal
    out = [cur]
    while cur != start:
        cur = came_from[cur]
        out.append(cur)
    out.reverse()
    return out


def astar_single(
    G: nx.Graph, start: Pos, goal: Pos, h_costs: dict[Pos, float]
) -> Path | None:
    """Standard A* search for a single agent with precomputed heuristic costs.

    This function implements the standard A* algorithm for finding the shortest
    path from start to goal using a precomputed admissible heuristic.

    Args:
        G: NetworkX graph to search
        start: Starting position
        goal: Target goal position
        h_costs: Precomputed heuristic costs from each position to goal

    Returns:
        Shortest path from start to goal, or None if no path exists

    Example:
        >>> G = nx.grid_2d_graph(3, 3)
        >>> h_costs = single_source_to_goal_costs(G, (2, 2))
        >>> path = astar_single(G, (0, 0), (2, 2), h_costs)
        >>> len(path)
        5  # Path length from (0,0) to (2,2)
    """
    if start == goal:
        return [start]

    # Initialize A* data structures
    open_heap: list[tuple[float, Pos]] = []  # Priority queue: (f_score, position)
    g = {start: 0.0}  # Actual cost from start to each position
    f0 = g[start] + h_costs.get(start, 0.0)  # f_score = g_score + h_score
    heapq.heappush(open_heap, (f0, start))
    came: dict[Pos, Pos] = {}  # Parent pointers for path reconstruction

    # A* main loop
    while open_heap:
        _, u = heapq.heappop(open_heap)  # Get position with lowest f_score

        # Check if we reached the goal
        if u == goal:
            return reconstruct_path(came, start, goal)

        # Explore all neighbors (including waiting at current position)
        for v in neighbors_with_wait(G, u):
            w = edge_weight(G, u, v)  # Get edge weight
            tentative = g[u] + w  # Calculate tentative g_score

            # Update if we found a better path to v
            if tentative < g.get(v, float("inf")):
                g[v] = tentative  # Update g_score
                came[v] = u  # Update parent pointer
                f = tentative + h_costs.get(v, 0.0)  # Calculate f_score
                heapq.heappush(open_heap, (f, v))  # Add to open set

    return None  # No path found


# ----------------------- Collision Checking ----------------------


def get_pos_at(path: Path, t: int) -> Pos:
    if t < len(path):
        return path[t]
    return path[-1]


def any_conflicts_at_time(
    plan: Plan, agents: list[Agent], t: int, collision_radius: float
) -> set[Agent]:
    """Return the set of agents that are in conflict at time t."""
    bad: set[Agent] = set()
    for i in range(len(agents)):
        ai = agents[i]
        pi_t = get_pos_at(plan[ai], t)
        for j in range(i + 1, len(agents)):
            aj = agents[j]
            pj_t = get_pos_at(plan[aj], t)
            # Vertex collision (with radius)
            if has_vertex_collision(pi_t, pj_t, collision_radius):
                bad.update((ai, aj))
    return bad


def any_edge_swaps_at_time(plan: Plan, agents: list[Agent], t: int) -> set[Agent]:
    """Return agents involved in an edge swap between t-1 and t."""
    bad: set[Agent] = set()
    if t == 0:
        return bad
    for i in range(len(agents)):
        ai = agents[i]
        pi_prev = get_pos_at(plan[ai], t - 1)
        pi = get_pos_at(plan[ai], t)
        for j in range(i + 1, len(agents)):
            aj = agents[j]
            pj_prev = get_pos_at(plan[aj], t - 1)
            pj = get_pos_at(plan[aj], t)
            if has_edge_swap(pi_prev, pi, pj_prev, pj):
                bad.update((ai, aj))
    return bad


# --------------------- Coupled A* (repair step) ------------------


class JointNode:
    __slots__ = ("poses", "t")

    def __init__(self, poses: tuple[Pos, ...], t: int):
        self.poses = poses
        self.t = t

    def __hash__(self) -> int:
        return hash((self.poses, self.t))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, JointNode)
            and self.poses == other.poses
            and self.t == other.t
        )


def coupled_astar_repair(
    G: nx.Graph,
    subset: list[Agent],
    plan: Plan,
    goals: dict[Agent, Pos],
    other_agents: list[Agent],
    start_time: int,
    h_costs_per_agent: dict[Agent, dict[Pos, float]],
    collision_radius: float,
    max_extra_steps: int = 200,
) -> dict[Agent, Path] | None:
    """
    Replan for 'subset' agents starting at time 'start_time', holding other_agents
    as dynamic obstacles following their current planned trajectories.
    Returns new tails for subset agents (from start_time to end).
    """
    # Initial joint state (current positions at start_time)
    start_poses = tuple(get_pos_at(plan[a], start_time) for a in subset)
    goals_tuple = tuple(goals[a] for a in subset)

    # If already at goals, nothing to do
    if all(p == g for p, g in zip(start_poses, goals_tuple)):
        return {a: [get_pos_at(plan[a], start_time)] for a in subset}

    # Heuristic: sum of single-agent to-goal distances
    def h(poses: tuple[Pos, ...]) -> float:
        return sum(h_costs_per_agent[a].get(p, 0.0) for a, p in zip(subset, poses))

    # To bound the search horizon, estimate:
    #   baseline = max of single-agent distances; add buffer
    horizon = int(
        max(
            h_costs_per_agent[a].get(get_pos_at(plan[a], start_time), 0.0)
            for a in subset
        )
        + 2 * len(subset)
        + 10
    )
    horizon = min(horizon + max_extra_steps, horizon + 400)

    # Open set
    start_node = JointNode(start_poses, start_time)
    open_heap: list[tuple[float, int, JointNode]] = []
    g_cost = {start_node: 0.0}
    f0 = h(start_poses)
    heapq.heappush(open_heap, (f0, 0, start_node))  # (f, tie, node)
    parent: dict[JointNode, tuple[JointNode, tuple[Pos, ...]]] = {}
    tie = 1

    def other_pos_at(a: Agent, t: int) -> Pos:
        return get_pos_at(plan[a], t)

    def valid_transition(
        prev_poses: tuple[Pos, ...], next_poses: tuple[Pos, ...], t_from: int, t_to: int
    ) -> bool:
        # Vertex collisions inside subset
        for i in range(len(next_poses)):
            for j in range(i + 1, len(next_poses)):
                if has_vertex_collision(next_poses[i], next_poses[j], collision_radius):
                    return False
                # Edge swap inside subset
                if has_edge_swap(
                    prev_poses[i], next_poses[i], prev_poses[j], next_poses[j]
                ):
                    return False

        # Collisions against other agents following their (fixed) plan
        for k, _a in enumerate(subset):
            next_k = next_poses[k]
            prev_k = prev_poses[k]
            # Vertex collision with others at t_to
            for b in other_agents:
                pb_to = other_pos_at(b, t_to)
                if has_vertex_collision(next_k, pb_to, collision_radius):
                    return False
                # Edge swap with others across (t_from -> t_to)
                pb_from = other_pos_at(b, t_from)
                if has_edge_swap(prev_k, next_k, pb_from, pb_to):
                    return False
        return True

    visited: set[JointNode] = set()

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)

        # Goal test: all at goals (allow arriving before horizon; then wait)
        if cur.poses == goals_tuple:
            # Reconstruct joint sequence
            joint_seq: list[tuple[Pos, ...]] = [cur.poses]
            node = cur
            while node in parent:
                node, step_poses = parent[node]
                joint_seq.append(node.poses)
            joint_seq.reverse()  # from start to goal
            # Convert to per-agent tails (include start_time position)
            tails: dict[Agent, Path] = {a: [] for a in subset}
            for poses in joint_seq:
                for idx, a in enumerate(subset):
                    tails[a].append(poses[idx])
            return tails

        if cur.t - start_time > horizon:
            # give up on this branch
            continue

        # Expand: each agent may move to neighbor or wait
        next_time = cur.t + 1
        choices_per_agent: list[list[Pos]] = []
        for idx, _a in enumerate(subset):
            choices_per_agent.append(neighbors_with_wait(G, cur.poses[idx]))

        # Cartesian product without recursion (small subset; typical size 2–3)
        def cartesian(prod_lists: list[list[Pos]]) -> list[tuple[Pos, ...]]:
            out: list[tuple[Pos, ...]] = [()]
            for L in prod_lists:
                out = [p + (x,) for p in out for x in L]
            return out

        for next_poses in cartesian(choices_per_agent):
            if not valid_transition(cur.poses, next_poses, cur.t, next_time):
                continue
            # step cost = sum of agent step costs
            step = sum(
                edge_weight(G, cur.poses[i], next_poses[i])
                for i in range(len(next_poses))
            )
            nxt = JointNode(next_poses, next_time)
            ng = g_cost[cur] + step
            if ng < g_cost.get(nxt, float("inf")):
                g_cost[nxt] = ng
                parent[nxt] = (cur, next_poses)
                f = ng + h(next_poses)
                heapq.heappush(open_heap, (f, tie, nxt))
                tie += 1

    return None  # failed to repair


# ----------------------------- M* -------------------------------


def mstar_plan_paths(
    graph: nx.Graph,
    starts: dict[Agent, Pos],
    goals: dict[Agent, Pos],
    collision_radius: float = 1.0,
) -> Plan | None:
    """
    Subdimensional Expansion (M*) planner.
    - graph: networkx Graph with weighted edges (weight=...), undirected or directed.
    - starts/goals: mapping agent -> node.
    - collision_radius: if nodes are numeric tuples, robots colliding if distance < radius (also treats same-node as collision).
    Returns dict agent -> path (list of nodes), or None if unsolvable.
    """
    agents: list[Agent] = list(starts.keys())

    # Precompute single-agent admissible heuristics to each goal
    h_costs_per_agent: dict[Agent, dict[Pos, float]] = {
        a: single_source_to_goal_costs(graph, goals[a]) for a in agents
    }

    # Initial independent plans
    plan: Plan = {}
    for a in agents:
        p = astar_single(graph, starts[a], goals[a], h_costs_per_agent[a])
        if p is None:
            return None
        plan[a] = p

    # Main loop: extend time and repair when conflicts appear
    # We iteratively look for earliest conflict, repair from that time, and continue.
    safety_cap = 2000  # guard against infinite loops
    iters = 0

    while iters < safety_cap:
        iters += 1
        # Determine planning horizon so far
        horizon = max(len(p) for p in plan.values())
        # Scan times for conflicts
        conflicted_time: int | None = None
        conflict_set: set[Agent] = set()

        for t in range(horizon):
            bad_v = any_conflicts_at_time(plan, agents, t, collision_radius)
            bad_e = any_edge_swaps_at_time(plan, agents, t)
            bad = bad_v | bad_e
            if bad:
                conflicted_time = t
                conflict_set = bad
                break

        if conflicted_time is None:
            # No conflicts; success
            return plan

        # Prepare coupled repair for the conflict set starting at conflicted_time
        subset = sorted(list(conflict_set), key=str)
        others = [a for a in agents if a not in conflict_set]

        tails = coupled_astar_repair(
            graph,
            subset=subset,
            plan=plan,
            goals=goals,
            other_agents=others,
            start_time=conflicted_time,
            h_costs_per_agent=h_costs_per_agent,
            collision_radius=collision_radius,
        )
        if tails is None:
            return None

        # Splice repaired tails back into each agent's plan
        # Keep prefix up to conflicted_time (exclusive), then append new tail.
        # If tails end early, they end at goal and will be padded in later loops.
        for a in subset:
            prefix = plan[a][:conflicted_time]
            new_tail = tails[a]
            # Avoid duplicating the starting position at conflicted_time if prefix already ends with it
            if prefix and new_tail and prefix[-1] == new_tail[0]:
                spliced = prefix + new_tail[1:]
            else:
                spliced = prefix + new_tail
            plan[a] = spliced

        # Equalize lengths to maintain consistent future collision checks
        new_h = max(len(p) for p in plan.values())
        for a in agents:
            plan[a] = pad_to_length(plan[a], new_h)

    # If we hit the safety cap, declare failure
    return None
