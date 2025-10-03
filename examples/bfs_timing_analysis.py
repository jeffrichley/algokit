#!/usr/bin/env python3
"""Example showing how to use the timing tracker with BFS scene for analysis."""

import sys
from pathlib import Path

# Add the main project to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from agloviz.scenes.planning.bfs.breadth_first_search_scene import BreadthFirstSearchScene
    from agloviz.core.timing_tracker import get_timing_tracker, reset_timing_tracker
    from algokit.core.helpers import load_harbor_scenario
    print("✅ Successfully imported BFS scene and timing modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def analyze_bfs_timing():
    """Analyze timing usage in BFS scene without running Manim."""
    print("🎬 BFS TIMING ANALYSIS")
    print("=" * 50)

    # Reset tracker to start fresh
    reset_timing_tracker()

    # Load a scenario
    try:
        scenario_file = project_root / "data" / "examples" / "scenarios" / "harbor_flood_small.yaml"
        scenario = load_harbor_scenario(str(scenario_file))
        print(f"✅ Loaded scenario: {scenario.name}")
    except Exception as e:
        print(f"⚠️ Could not load scenario: {e}")
        print("   Creating default scenario...")
        from algokit.core.helpers import HarborNetScenario
        scenario = HarborNetScenario(
            name="Default Test Scenario",
            description="Default scenario for timing analysis",
            width=5, height=4,
            start=(0, 0), goal=(4, 3),
            obstacles=[(1, 1), (2, 1), (2, 2)],
            text_below_grid_offset=3.0
        )

    # Create BFS scene instance (but don't run construct - just initialize)
    print("🎯 Creating BFS scene instance...")
    scene = BreadthFirstSearchScene(scenario=scenario)

    # Simulate some timing requests that would happen during scene construction
    print("📊 Simulating timing requests...")

    # Title and setup timings
    scene.get_animation_time("title_write")
    scene.get_animation_time("subtitle_write")
    scene.get_animation_time("grid_wave")
    scene.get_animation_time("legend_fade_in")
    scene.get_wait_time("initial_setup")

    # Token placement timings
    scene.get_animation_time("start_indicate")
    scene.get_animation_time("start_label_write")
    scene.get_animation_time("goal_flash")
    scene.get_animation_time("goal_label_write")

    # Water token animations (multiple obstacles)
    for _ in range(len(scenario.obstacles)):
        scene.get_animation_time("token_entrance")

    # HUD and queue setup
    scene.get_animation_time("hud_fade_in")
    scene.get_animation_time("queue_fade_in")

    # Simulate BFS event processing
    num_events = 25  # Simulated number of BFS events
    for _ in range(num_events):
        scene.get_animation_time("path_dot_grow")
        scene.get_animation_time("path_line_create")
        scene.get_wait_time("dequeue_delay")

        # Some events trigger layer completion
        if _ % 5 == 0:
            scene.get_animation_time("layer_complete_pulse")
            scene.get_wait_time("layer_complete_wait")

    # Goal celebration
    scene.get_animation_time("confetti_effects")
    scene.get_animation_time("goal_text_reveal")
    scene.get_wait_time("celebration_pause")

    # Path reconstruction
    scene.get_animation_time("tracer_movement")
    scene.get_animation_time("path_highlight_pulse")
    scene.get_animation_time("conclusion_text")
    scene.get_wait_time("final_study_pause")

    # Get timing statistics
    print("📈 Generating timing analysis...")

    # Print quick summary
    scene.print_timing_summary()

    # Generate full report
    report = scene.generate_timing_report(save_to_file=True, print_to_console=False)

    # Show specific insights
    print("\n🔍 TIMING INSIGHTS:")

    # Get the tracker for detailed analysis
    tracker = get_timing_tracker()

    # Find most frequently requested animations
    animation_counts = [(name, len(requests)) for name, requests in tracker.animation_requests.items()]
    animation_counts.sort(key=lambda x: x[1], reverse=True)

    print("   Most Frequently Requested Animations:")
    for name, count in animation_counts[:5]:
        stats = tracker.get_animation_stats(name)
        print(f"     {name}: {count} requests, {stats['total_time']:.3f}s total")

    # Find longest total time animations
    animation_times = [(name, tracker.get_animation_stats(name)['total_time'])
                      for name in tracker.animation_requests.keys()]
    animation_times.sort(key=lambda x: x[1], reverse=True)

    print("\n   Animations with Most Total Time:")
    for name, total_time in animation_times[:5]:
        stats = tracker.get_animation_stats(name)
        print(f"     {name}: {total_time:.3f}s total ({stats['count']} requests)")

    # Find most time-consuming waits
    wait_times = [(name, tracker.get_wait_stats(name)['total_time'])
                  for name in tracker.wait_requests.keys()]
    wait_times.sort(key=lambda x: x[1], reverse=True)

    print("\n   Most Time-Consuming Waits:")
    for name, total_time in wait_times:
        stats = tracker.get_wait_stats(name)
        print(f"     {name}: {total_time:.3f}s total ({stats['count']} requests)")

    # Calculate total estimated scene time
    total_time = (
        sum(tracker.get_animation_stats(name)['total_time'] for name in tracker.animation_requests.keys()) +
        sum(tracker.get_wait_stats(name)['total_time'] for name in tracker.wait_requests.keys())
    )

    print(f"\n⏱️  ESTIMATED TOTAL SCENE TIME: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Current timing mode: {scene.current_mode}")

    # Show time breakdown by category
    total_animation_time = sum(tracker.get_animation_stats(name)['total_time']
                              for name in tracker.animation_requests.keys())
    total_wait_time = sum(tracker.get_wait_stats(name)['total_time']
                         for name in tracker.wait_requests.keys())

    print(f"\n📊 TIME BREAKDOWN:")
    print(f"   Animation Time: {total_animation_time:.1f}s ({total_animation_time/total_time*100:.1f}%)")
    print(f"   Wait Time: {total_wait_time:.1f}s ({total_wait_time/total_time*100:.1f}%)")

    return scene


def compare_timing_modes():
    """Compare timing usage across different speed modes."""
    print("\n" + "=" * 50)
    print("🏃 TIMING MODE COMPARISON")
    print("=" * 50)

    modes = ["cinematic", "development", "quick_demo"]
    mode_results = {}

    for mode in modes:
        print(f"\n📊 Analyzing {mode.upper()} mode...")

        # Reset tracker
        reset_timing_tracker()

        # Create scene with specific mode
        scenario_file = project_root / "data" / "examples" / "scenarios" / "harbor_flood_small.yaml"
        try:
            scenario = load_harbor_scenario(str(scenario_file))
        except:
            from algokit.core.helpers import HarborNetScenario
            scenario = HarborNetScenario(
                name="Default Test Scenario",
                width=5, height=4,
                start=(0, 0), goal=(4, 3),
                obstacles=[(1, 1), (2, 1), (2, 2)]
            )

        scene = BreadthFirstSearchScene(scenario=scenario)
        scene.set_timing_mode(mode)

        # Simulate typical timing requests
        typical_animations = [
            "title_write", "subtitle_write", "grid_wave", "legend_fade_in",
            "start_indicate", "goal_flash", "hud_fade_in", "queue_fade_in",
            "token_entrance", "path_dot_grow", "layer_complete_pulse",
            "confetti_effects", "tracer_movement"
        ]

        typical_waits = [
            "initial_setup", "dequeue_delay", "layer_complete_wait",
            "celebration_pause", "final_study_pause"
        ]

        # Request each timing
        for animation in typical_animations:
            scene.get_animation_time(animation)

        for wait in typical_waits:
            scene.get_wait_time(wait)

        # Get total time
        tracker = get_timing_tracker()
        total_animation_time = sum(tracker.get_animation_stats(name)['total_time']
                                  for name in tracker.animation_requests.keys())
        total_wait_time = sum(tracker.get_wait_stats(name)['total_time']
                             for name in tracker.wait_requests.keys())
        total_time = total_animation_time + total_wait_time

        mode_results[mode] = {
            "total_time": total_time,
            "animation_time": total_animation_time,
            "wait_time": total_wait_time,
            "total_requests": len(typical_animations) + len(typical_waits)
        }

        print(f"   Total Time: {total_time:.1f}s")
        print(f"   Animation: {total_animation_time:.1f}s, Wait: {total_wait_time:.1f}s")

    # Show comparison
    print(f"\n📈 MODE COMPARISON SUMMARY:")
    print(f"{'Mode':<12} {'Total Time':<12} {'Animation':<12} {'Waits':<12} {'Speedup':<10}")
    print("-" * 60)

    cinematic_time = mode_results["cinematic"]["total_time"]
    for mode in modes:
        result = mode_results[mode]
        speedup = cinematic_time / result["total_time"] if result["total_time"] > 0 else 0
        print(f"{mode:<12} {result['total_time']:<12.1f} {result['animation_time']:<12.1f} "
              f"{result['wait_time']:<12.1f} {speedup:<10.1f}x")


if __name__ == "__main__":
    # Run timing analysis
    scene = analyze_bfs_timing()

    # Compare modes
    compare_timing_modes()

    print(f"\n📁 Detailed reports saved in: {project_root / 'output'}")
    print("🎬 Timing analysis complete!")
