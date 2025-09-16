Algokit Pathfinding Storyworld + BFS Implementation Pack

Working draft for a shared narrative, visualization spec, and first algorithm (Breadth‑First Search). This follows our process doc: code → tests → visualization (Manim) → docs and includes a reusable AI‑coding‑agent prompt.

🎭 Shared Storyworld — "HarborNet: Search & Rescue"

A coastal research city floods after a storm. Autonomous dinghy drones (Type‑3 Searchers) patrol canals to locate survivors. Relay buoys (Type‑2 Movers) reposition to maintain radio links back to the Command Pier (Type‑1 Base). Streets and canals form a graph. Obstacles are debris or closed locks.

We’ll reuse this world across all pathfinding algorithms so comparisons are intuitive:

Nodes = intersections/locks/checkpoints (grid cells or arbitrary graph vertices)

Edges = passable water channels/streets (uniform or weighted by current / congestion)

Goal = reach a target (survivor location) and/or maintain comms to base

Costs (for weighted algos) = current strength, debris, traffic, risk

Heuristics (A*) = straight‑line/Manhattan water distance or radio SNR proxy

Algorithm Personas in This World

BFS: "Wavefront Sweep" — spreads evenly in rings; great for fewest‑hops routes when all edges have equal effort.

DFS: "Tunnel Rat" — dives deep along a promising canal, backtracks when blocked; good for exploration, not optimal paths.

Dijkstra: "Quartermaster" — plans the cheapest route accounting for currents/risks; optimal with non‑negative costs.

A*: "Harbor Pilot" — guided by a chart (heuristic); balances speed and optimality when the heuristic is admissible.

🎥 Shared Visualization Spec (Manim)

We’ll implement a single Manim scaffold and feed it a per‑algorithm adapter.

Visual language:

Grid/graph overlay of the harbor; obstacles as dark tiles; start (Command Pier) = blue; goal (SOS) = orange.

Frontier (queue/stack/priority) highlighted near the top‑right as live data structure.

Visited nodes = pale cyan; current node = bright yellow pulse.

Edges relaxed/expanded flash briefly.

Final path animates as a thick teal spline; caption shows metric: hops (BFS), distance (Dijkstra), f=g+h (A*).

Reusability hooks:

AlgorithmAdapter interface: push_frontier, pop_frontier, should_revisit, update_metadata.

Event stream (SearchEvent) that the scene consumes to animate in lock‑step with the algorithm.

📂 Project Structure (per‑algo pattern)
algokit/
  src/algokit/algos/bfs.py
  src/algokit/graphs/primitives.py
  src/algokit/viz/adapters.py
  src/algokit/viz/scenes.py
  tests/algos/test_bfs.py
  viz/manim/bfs_scene.py  # thin wrapper around shared Scene
  docs/algos/bfs.md
  data/examples/harbor_small.json
🧠 BFS — Design Notes

Input: unweighted graph G=(V,E); start s; optional goal t.

Guarantee: finds minimum‑hop path in O(|V|+|E|).

Data structures: queue, parent map, visited bitset.

Events for viz: enqueue, dequeue, discover, goal_found, reconstruct_path.

Edge cases to test:

Start==Goal; disconnected graph; multiple shortest paths; empty graph; large sparse vs dense; obstacles/walls.

✅ Tests First (PyTest)
# tests/algos/test_bfs.py
import pytest
from algokit.algos.bfs import bfs_shortest_path
from algokit.graphs.primitives import Graph




def test_bfs_trivial_start_is_goal():
    g = Graph()
    g.add_node("A")
    assert bfs_shortest_path(g, "A", "A") == ["A"]




def test_bfs_simple_path():
    g = Graph.from_edges([("A","B"),("B","C"),("A","C")])
    # A->C should be length 1 via direct edge, but BFS returns any min path
    path = bfs_shortest_path(g, "A", "C")
    assert len(path) == 2
    assert path[0] == "A" and path[-1] == "C"




def test_bfs_disconnected():
    g = Graph.from_edges([("A","B"),("C","D")])
    assert bfs_shortest_path(g, "A", "D") is None




def test_bfs_grid_obstacles():
    # 3x3 grid with a wall in the middle
    g = Graph.grid(width=3, height=3, blocked={(1,1)})
    path = bfs_shortest_path(g, (0,0), (2,2))
    assert path is not None
    # minimal hops through available corridors
    assert path[0] == (0,0) and path[-1] == (2,2)
🧩 Core Graph Primitives
# src/algokit/graphs/primitives.py
from collections import defaultdict
from typing import Hashable, Iterable


class Graph:
    def __init__(self) -> None:
        self._adj: dict[Hashable, set[Hashable]] = defaultdict(set)


    def add_node(self, u: Hashable) -> None:
        _ = self._adj[u]


    def add_edge(self, u: Hashable, v: Hashable, undirected: bool = True) -> None:
        self._adj[u].add(v)
        if undirected:
            self._adj[v].add(u)


    @classmethod
    def from_edges(cls, edges: Iterable[tuple[Hashable, Hashable]], undirected: bool = True) -> "Graph":
        g = cls()
        for u, v in edges:
            g.add_edge(u, v, undirected=undirected)
        return g


    @classmethod
    def grid(cls, width: int, height: int, blocked: set[tuple[int,int]] | None = None) -> "Graph":
        blocked = blocked or set()
        g = cls()
        for x in range(width):
            for y in range(height):
                if (x,y) in blocked:
                    continue
                g.add_node((x,y))
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < width and 0 <= ny < height and (nx,ny) not in blocked:
                        g.add_edge((x,y),(nx,ny))
        return g


    def neighbors(self, u: Hashable):
        return self._adj.get(u, ())
🚀 BFS Implementation (+ Event Stream)
# src/algokit/algos/bfs.py
from collections import deque
from dataclasses import dataclass
from typing import Hashable, Iterable, Iterator, Optional


@dataclass(frozen=True)
class SearchEvent:
    kind: str  # 'enqueue' | 'dequeue' | 'discover' | 'goal_found' | 'reconstruct'
    node: Hashable
    parent: Optional[Hashable] = None




def bfs_shortest_path(G, start: Hashable, goal: Hashable | None = None) -> Optional[list[Hashable]]:
    """Return a min-hop path from start to goal (or None). If goal is None, explores all reachable nodes."""
    if start == goal:
        return [start]
    q = deque([start])
    visited: set[Hashable] = {start}
    parent: dict[Hashable, Hashable] = {}
    while q:
        u = q.popleft()
        if u == goal:
            return _reconstruct_path(parent, start, goal)
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                q.append(v)
    return None if goal is not None else [*visited]




def bfs_with_events(G, start: Hashable, goal: Hashable | None = None) -> tuple[Optional[list[Hashable]], list[SearchEvent]]:
    if start == goal:
        return [start], [SearchEvent("enqueue", start), SearchEvent("goal_found", start)]
    q = deque([start])
    events: list[SearchEvent] = [SearchEvent("enqueue", start)]
    visited: set[Hashable] = {start}
    parent: dict[Hashable, Hashable] = {}
    while q:
        u = q.popleft()
        events.append(SearchEvent("dequeue", u))
        if u == goal:
            path = _reconstruct_path(parent, start, goal)
            events.append(SearchEvent("goal_found", u))
            events.append(SearchEvent("reconstruct", u))
            return path, events
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                q.append(v)
                events.append(SearchEvent("discover", v, parent=u))
    return (None if goal is not None else [*visited], events)




def _reconstruct_path(parent: dict, s: Hashable, t: Hashable) -> list[Hashable]:
    path = [t]
    while path[-1] != s:
        path.append(parent[path[-1]])
    path.reverse()
    return path
🎬 Manim Scene (Shared) — BFS Adapter Example
# src/algokit/viz/scenes.py
from manim import *
from typing import Hashable


class HarborGridScene(Scene):
    def construct(self, grid, events, start, goal):
        # grid: object with width, height, blocked
        # events: list[SearchEvent]
        # Draw grid
        tiles = VGroup()
        for x in range(grid.width):
            for y in range(grid.height):
                rect = Square(side_length=0.5)
                rect.move_to(np.array([x*0.5, y*0.5, 0]))
                rect.set_fill(color=GREY_E, opacity=0.2)
                if (x,y) in grid.blocked:
                    rect.set_fill(color=GREY_D, opacity=1.0)
                tiles.add(rect)
        tiles.center()
        self.play(FadeIn(tiles))


        # Start/Goal markers
        start_dot = Dot(color=BLUE).move_to(tiles[0].get_center())
        goal_dot = Dot(color=ORANGE).move_to(tiles[-1].get_center())
        self.play(FadeIn(start_dot), FadeIn(goal_dot))


        # Animate events (simplified placeholder)
        for ev in events:
            if ev.kind == "discover":
                self.play(Indicate(start_dot, scale_factor=1.1), run_time=0.05)


        self.wait(0.5)
# viz/manim/bfs_scene.py
from algokit.graphs.primitives import Graph
from algokit.algos.bfs import bfs_with_events
from algokit.viz.scenes import HarborGridScene


# Provide a small harbor grid and run manim CLI to render the scene

Note: We’ll refine coordinates, tile ↔ node mapping, camera framing, and proper highlighting as we iterate. The key is that all algos fire the same SearchEvents, so the scene logic is reusable.

🧾 Docs Outline — docs/algos/bfs.md

Concept: BFS for uniform‑cost hop minimization

Where it shines in HarborNet: first‑response routing when currents are negligible

Complexity: O(|V|+|E|); memory proportional to frontier size

Correctness sketch: invariant of non‑decreasing distance by layers

Comparison callouts: vs DFS (completeness/optimality), vs Dijkstra (weights), vs A* (speed with heuristic)

Failure modes: huge memory on broad graphs; misleading when weights matter

Annotated run: screenshots/GIF from Manim scene + step table

🔁 Reusable AI‑Coding‑Agent Prompt (Checklist Template)

Copy/paste and fill the Algorithm‑Specific Notes section.

Role: You are an elite Python engineer collaborating on Algokit’s pathfinding suite. You write production‑quality code, tests, and a Manim visualization adapter.


Objectives:
1) Implement the algorithm in `src/algokit/algos/<algo>.py` with clean types and docstrings.
2) Write focused PyTest tests in `tests/algos/test_<algo>.py` covering edge cases and property checks.
3) Emit a portable event stream (`SearchEvent`) from the algorithm for the shared Manim scene.
4) Provide a thin Manim driver in `viz/manim/<algo>_scene.py` that feeds the shared scene.
5) Draft `docs/algos/<algo>.md` with complexity, correctness notes, and HarborNet tie‑ins.


Standards Checklist:
- [ ] Pure function core (no I/O) + deterministic behavior
- [ ] Type hints (Python ≥3.12), mypy‑clean, ruff‑clean
- [ ] O(|V|+|E|) behavior validated on synthetic graphs
- [ ] Property: Parent pointers reconstruct a valid path
- [ ] Tests: trivial, simple, disconnected, blocked grid, multiple min paths
- [ ] Event stream includes: enqueue/dequeue/discover/goal_found/reconstruct
- [ ] Manim driver renders at least one small example (3×3 / 5×5)
- [ ] Docs include HarborNet narrative comparison section


Deliverables:
- PR modifying only allowed paths (src/tests/viz/docs)
- `README` snippet: how to run tests & render scene


Algorithm‑Specific Notes (to paste here when invoking the agent):
- Algorithm: Breadth‑First Search (BFS)
- Graph: unweighted; use `Graph.grid(width,height,blocked)` for examples
- Metrics: hop count
- Visualization details: color frontier ring per layer; pulse current node
- Edge cases to emphasize: start==goal, disconnected, multiple optimal paths
🧪 Quick Local Run Recipe

Tests: pytest -q

Visualization (example): manim -pqh viz/manim/bfs_scene.py HarborGridScene

Docs preview: mkdocs serve (or your doc system)

📌 Next Steps

Wire up the minimal HarborGridScene to actually map grid coords → squares and show real frontier coloring.

Add a QueuePanel VGroup that shows live enqueue/dequeue.

Record a short GIF for the docs.

Clone this pack for DFS → Dijkstra → A* using the same story + viz.

Love it. I spun up a shared “HarborNet: Search & Rescue” storyworld, a reusable Manim viz spec, and a full BFS starter pack (tests-first, core code, viz adapter, docs outline, plus a reusable AI-agent checklist prompt). It’s all organized and ready to iterate in the canvas.

Want me to also drop in a tiny manim driver that renders a 5×5 harbor grid GIF for your docs next?