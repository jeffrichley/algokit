Jeff, Iris here—director’s chair on, Manim brain engaged. Below is a cinematic, step-by-step storyboard you can translate straight into scenes and beats. I’ll reference specific Manim moves so Lily can wire it up fast.

# BFS “Small Flooded Harbor” — Director’s Plan

## Visual language & assets (set once)

* **Palette**

  * Background: BLACK
  * Grid lines: `GRAY_B`
  * Start: `GREEN_C` (filled)
  * Goal: `GOLD` (star)
  * Obstacles (“flooded” water): `BLUE_E` fill with faint wave overlay (Dashed arcs)
  * Frontier: `BLUE_C` fill, 60% opacity
  * Visited: `YELLOW_E` (pale), 40% opacity
  * Final path: `MAGENTA` (thick outline)
* **Objects**

  * `cell_squares[(x,y)] -> Square` (you already have these; keep a dict)
  * `start_dot` or filled square token; `goal_star = Star()`
  * `queue_panel = RoundedRectangle()` + `Text("Queue (FIFO)")`
  * `token(x,y) = SmallSquare(fill=<status-color>)` to travel into/out of queue
  * `parent_arrows[(x,y)] = Arrow(parent_cell, this_cell, buff=0.05, stroke_width=2)`
  * HUD: `Visited`, `Frontier`, `Depth`, `Queue` counters via `ValueTracker` + `DecimalNumber`
  * Ring hint: thin `Square` outlines for L1/L2 shells, or glow on frontier layer

---

## Scene 0 — Establishing (5–7s)

1. **Title + grid on**

   * You already do:

     ```python
     title = Text(f"BFS Grid: {self.scenario.name}", font_size=36, color=WHITE).to_edge(UP)
     self.play(Write(title), Create(self.grid_group), run_time=1.8)
     ```
   * **Polish**: after `Create(grid)`, `ApplyWave(self.grid_group, amplitude=0.05, run_time=0.8)` for a subtle shimmer.

2. **Subtitle + legend** (1.5s)

   * Subtitle under title: `"Shortest paths on unweighted graphs"`.
   * Bottom-right legend (small VGroup with dot, star, water square, frontier/visited swatches).
   * `LaggedStart(Write(subtitle), FadeIn(legend, shift=DOWN*0.25), lag_ratio=0.2)`

3. **Frame the stage** (0.8s)

   * `SurroundingRectangle(self.grid_group, color=GRAY_D)` then fade its opacity to 30%: `self.play(FadeIn(frame_rect, run_time=0.4)); self.play(frame_rect.animate.set_stroke(opacity=0.3), run_time=0.4)`

---

## Scene 1 — Cast the actors (3–4s)

4. **Start + Goal placement**

   * Drop **start** token on its cell: `GrowFromCenter(start_token)` + `Indicate(start_cell)`
   * Drop **goal**: `GrowFromCenter(goal_star)` + a tiny `Flash(goal_star.get_center())`
   * Optional labels: `Start`, `Goal` as small `Text` near cells.

5. **Flooded harbor obstacles**

   * For each obstacle cell: `cell.set_fill(BLUE_E, opacity=0.65)` with `GrowFromCenter(wave_overlay)` (thin `Arc` or `CurvedArrow` pieces arranged as waves), drive them in using `LaggedStartMap(FadeIn, lag_ratio=0.03)` so they “ripple” on.

---

## Scene 2 — Meet the Queue (2s)

6. **Queue panel enters**

   * Right side: `queue_panel` slides in: `self.play(FadeIn(queue_panel, shift=RIGHT*0.5), Write(queue_title))`
   * Enqueue the start: create a mini `token_start` cloned from start color, `MoveToTarget` into the panel’s left side.

7. **HUD counters**

   * Top-left: `Visited: 0  Frontier: 1  Depth: 0  Queue: 1` using `ValueTracker`s.
   * `FadeIn(hud_group, shift=UP*0.2)`

---

## Scene 3 — BFS wavefront (core loop, \~15–20s)

> Play this as a repeating pattern. Drive it from your BFS event stream: `dequeue -> explore -> discover/enqueue -> mark visited`.

### Loop template per node

8. **Dequeue**

   * Animate the leftmost queue token sliding out to the associated grid cell:
     `self.play(token.animate.move_to(cell.center), run_time=0.35)`
   * `queue_count.decrement(1)`, `Write` a small “pop” indicator on the panel; update HUD number smoothly (`DecimalNumber.animate.set_value`).

9. **Visit**

   * Turn dequeued cell from frontier color to visited color:
     `self.play(cell.animate.set_fill(YELLOW_E, opacity=0.4), run_time=0.2)`
   * Pulse: `Indicate(cell, scale_factor=1.02, color=YELLOW_E)`

10. **Explore neighbors (4-way)**

* For each neighbor in order (Up, Right, Down, Left):

  * **Out of bounds**: briefly shake a ghost square at the edge (`ApplyMethod(neighbor_proxy.shift, RIGHT*0.05).reverse()`) then `FadeOut`.
  * **Obstacle**: outline neighbor cell in red for 0.2s (`ShowPassingFlash(SurroundingRectangle(ob_cell, color=RED))`).
  * **Already seen**: quick yellow tick over that cell (`✓` as `Text` tiny), `FadeOut`.
  * **New discovery**:

    * Color neighbor as **frontier**: `set_fill(BLUE_C, opacity=0.6)`
    * Draw **parent arrow**: `GrowArrow(Arrow(curr_cell, neighbor_cell, buff=0.05))`
    * Add a small “depth d+1” badge (`RoundedRectangle` + `Text(str(depth+1))` near corner).
    * Create a mini token at neighbor cell, then **enqueue** to panel tail:
      `self.play(token_neighbor.animate.move_to(queue_tail_pos), run_time=0.35)`
    * Update HUD: `frontier += 1`, `queue += 1`.

11. **Layer pulse (end of current breadth)**

* When the queue has just finished processing all nodes at the current depth, fire a **ring pulse** on that layer:

  * Option A: briefly outline all cells at `depth=k` with `ShowCreationThenFadeOut(SurroundingRectangle(cell))` via `LaggedStartMap`.
  * Option B: a thin `Square` ring centered at start with side length matching manhattan layer; expand opacity 0.4 → 0.0 in 0.6s.
* Update HUD `Depth: k+1` with a soft `TransformMatchingShapes` on the “Depth” label.

> Keep rhythm tight: `LaggedStart` for neighbor discoveries with `lag_ratio ~ 0.12–0.18`, faster for empty space, slower near obstacles for suspense.

---

## Scene 4 — Edge-case montage (quick 3s, optional)

12. In one iteration show all branches:

* Obstacle skip (red flash), out-of-bounds shake, already-visited tick, and new discovery (arrow+enqueue).
* Overlay tiny captions: “obstacle”, “out of bounds”, “seen”, “discovered”.

---

## Scene 5 — Goal found (4–5s)

13. **Discovery ping**

* The moment the goal is discovered (added to queue): subtle glow on goal (`OuterGlow` style via `ShowPassingFlashAround(goal_star)`), tiny “found in frontier” label, but **don’t** stop yet.

14. **Dequeued goal = success beat**

* When the goal is dequeued:

  * Freeze motion for 0.3s.
  * Fire a tasteful **confetti**: `VGroup(*[Dot().set_color(random_color) for _ in range(n)]).animate.scale(0.6).shift(random_vector)` with `LaggedStart`.
  * Big text card center: `Text(f"Goal reached at depth {D}")` with `GrowFromCenter`.
  * Dim non-path elements to 30% opacity: `self.play(self.grid_group.animate.set_opacity(0.3))` (except path cells we’ll reveal next).

---

## Scene 6 — Path reconstruction (6–7s)

15. **Backtrack along parents**

* Start from goal, follow `parent_arrows` backwards to start.
* For each step, thicken border or draw a **PathLine** (polyline) segment with `Create`.
* Animate a bright dot moving along the path at constant speed: `MoveAlongPath(dot, path_mobject)`
* As each path cell is confirmed, raise its z-order and set stroke to `MAGENTA`, stroke\_width 6–8.

16. **Lock it in**

* Leave the full polyline path up; brighten all path cells back to 100% while the rest stay dim.
* Caption: `Text("Shortest path (unweighted BFS)")` fades in below the grid.

17. **Optional proof hint** (1.5s)

* Fade in depth rings (0..D) around start; overlay: `Text("BFS explores by increasing depth")`.

---

## Scene 7 — Complexity & outro (4–5s)

18. **Complexity card**

* Right side slide-in card:

  ```
  Time: O(V + E)  (grid ~ O(W·H))
  Space: O(V)     (visited + queue)
  ```
* Fine arrows tying the words “visited” → yellow cells, “queue” → panel.

19. **Outro UI**

* Keep: title, legend, final path, goal+start, small “Replay / Speed x1 / Show parents” labels (non-interactive text hints).
* Fade to black or hold.

---

# Manim mapping: concrete calls & timings

* **Beats & tempo**

  * Setup beats: 0.3–0.8s per move.
  * BFS inner loop: keep step animations ≤0.35s; string them in `Succession` or `LaggedStart` for flow.
  * Layer pulses: 0.5–0.7s.
  * Goal confetti + card: \~1.8s.
  * Backtrack path: \~2.5–3.5s depending on length.

* **Frequently used primitives**

  * `Create`, `Write`, `FadeIn/FadeOut`, `Indicate`, `GrowFromCenter`, `GrowArrow`, `ShowPassingFlash`, `ShowCreationThenFadeOut`, `LaggedStart`, `LaggedStartMap`, `Succession`, `TransformMatchingShapes`, `MoveAlongPath`.

---

# Script skeleton (drop-in structure)

```python
from manim import *

class BFSHarbor(Scene):
    def construct(self):
        # 0) Title + Grid
        title = Text(f"BFS Grid: {self.scenario.name}", font_size=36).to_edge(UP)
        self.create_grid()                      # your method; sets self.grid_group
        self.play(Write(title), Create(self.grid_group), run_time=1.8)
        self.play(ApplyWave(self.grid_group, amplitude=0.05, run_time=0.8))

        # 1) Subtitle + Legend
        subtitle = Text("Shortest paths on unweighted graphs", font_size=22).next_to(title, DOWN, buff=0.2)
        legend = self.build_legend()            # returns VGroup
        self.play(LaggedStart(Write(subtitle), FadeIn(legend, shift=DOWN*0.25), lag_ratio=0.2))

        # 2) Start/Goal/Obstacles
        start_cell, goal_cell = self.place_start_goal()
        self.animate_obstacles()                # fill water + waves via LaggedStartMap

        # 3) Queue + HUD
        queue_panel, queue_slots = self.build_queue_panel()
        hud = self.build_hud()                  # uses ValueTrackers
        self.play(FadeIn(queue_panel, shift=RIGHT*0.5), Write(queue_panel.title))
        self.enqueue_start_token(start_cell)    # animates token into panel, updates HUD

        # 4) BFS Core Loop (drive from events)
        for event in self.bfs_events():         # yield events: dequeue, visit, neighbor_outcome, enqueue, layer_pulse, found_goal
            self.play(*self.render_event(event))# returns animations + HUD updates

        # 5) Goal beat + Backtrack
        self.goal_celebration()
        path = self.build_path_polyline()       # from parents
        self.play(Create(path), run_time=2.2)
        tracer = Dot(color=WHITE).move_to(path.get_start())
        self.add(tracer)
        self.play(MoveAlongPath(tracer, path), run_time=2.2)

        # 6) Complexity card + Outro
        card = self.complexity_card()
        self.play(FadeIn(card, shift=LEFT*0.2))
        self.hold_final_frame()
```

> Key helpers you’ll want:

* `build_legend()`: swatches + text.
* `place_start_goal()`: returns start/goal cell squares; draws markers (`Dot`/`Star`).
* `animate_obstacles()`: fills water squares + optional wave overlays with `LaggedStartMap`.
* `build_queue_panel()`: rounded rect + queue slot coordinates.
* `build_hud()`: sets `ValueTracker`s, `DecimalNumber`s, and a function `update_hud(dv, df, dq, dd)` to animate changes.
* `bfs_events()`: generator from your BFS run (pure Python), emitting a canonical event dict.
* `render_event(e)`: maps events to Manim animations.
* `goal_celebration()`: confetti VGroup w/ `LaggedStart`.
* `build_path_polyline()`: collects parent chain → `VMobject().set_points_as_corners([...])`.

---

# Event protocol (so the animation stays clean)

Emit events while running BFS in Python; the scene just *renders* them:

```python
{"type": "dequeue", "node": (x,y), "depth": k}
{"type": "visit", "node": (x,y)}
{"type": "neighbor", "from": (x,y), "to": (nx,ny), "status": "obstacle|oob|seen|new", "depth": k+1, "parent": (x,y)}
{"type": "enqueue", "node": (nx,ny)}
{"type": "layer_pulse", "depth": k}
{"type": "found_goal", "node": (gx,gy), "depth": D}
{"type": "backtrack", "path": [(gx,gy), ..., (sx,sy)]}
```

That clean separation keeps your visuals deterministic and makes it trivial to tweak timing.

---

# Micro-touches for “world-class” feel

* **Consistent easing**: use `rate_func=rate_functions.ease_in_out_sine` for most moves.
* **Breathing room**: insert 0.1–0.2s gaps (`self.wait(0.1)`) at act transitions.
* **Focus**: when expanding a cluster, slightly dim non-frontier cells to 70% then restore.
* **Accessibility**: pair color with pattern—frontier cells get dotted borders (`DashedVMobject(cell, num_dashes=24)`), visited keep solid pale fill.

---

If you want, I can convert this plan into a concrete Manim class with the event dispatcher + helper builders so you only plug in your `grid_to_screen`, start/goal/obstacles, and BFS generator.
