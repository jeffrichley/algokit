#!/usr/bin/env python3
"""Example showing how to customize BFS timing for different use cases."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from agloviz.scenes.planning.bfs.breadth_first_search_scene import BreadthFirstSearchScene


def create_fast_demo_scene():
    """Create a BFS scene optimized for fast demonstration."""
    print("🚀 Creating fast demo scene...")
    
    scene = BreadthFirstSearchScene()
    
    # Switch to quick demo mode for maximum speed
    scene.set_timing_mode("quick_demo")
    
    # Further customize by editing the config file
    print("For even faster speeds, edit:")
    print("  data/examples/scenarios/bfs_timing_config.yaml")
    print("And reduce these values:")
    print("  - max_events_displayed: 15")
    print("  - base_timings.confetti_effects: 0.1")
    print("  - wait_times.final_study_pause: 0.5")
    
    return scene


def create_presentation_scene():
    """Create a BFS scene optimized for presentations."""
    print("🎬 Creating presentation scene...")
    
    scene = BreadthFirstSearchScene()
    
    # Use cinematic mode for smooth viewing
    scene.set_timing_mode("cinematic")
    
    # For presentations, you might want to:
    print("For presentations, consider:")
    print("  - Using cinematic mode (default)")
    print("  - Reducing max_events_displayed to 20-30")
    print("  - Increasing wait times for audience comprehension")
    
    return scene


def create_development_scene():
    """Create a BFS scene optimized for development/testing."""
    print("⚡ Creating development scene...")
    
    scene = BreadthFirstSearchScene()
    
    # Use development mode for faster iteration
    scene.set_timing_mode("development")
    
    print("Development mode provides:")
    print("  - 4-10x faster animations")
    print("  - Quick iteration for testing")
    print("  - Good balance of speed vs. visibility")
    
    return scene


def main():
    """Demonstrate different timing configurations."""
    print("🎬 BFS Timing Configuration Examples")
    print("=" * 50)
    
    # Create different scene types
    fast_scene = create_fast_demo_scene()
    print()
    
    presentation_scene = create_presentation_scene()
    print()
    
    dev_scene = create_development_scene()
    print()
    
    # Show timing differences
    print("Timing Comparison:")
    print("-" * 30)
    
    animations = ["title_write", "confetti_effects", "tracer_movement"]
    wait_times = ["initial_setup", "final_study_pause"]
    
    print("Animation Times (seconds):")
    for anim in animations:
        fast_time = fast_scene.get_animation_time(anim)
        pres_time = presentation_scene.get_animation_time(anim)
        dev_time = dev_scene.get_animation_time(anim)
        print(f"  {anim}:")
        print(f"    Quick Demo: {fast_time:.3f}s")
        print(f"    Cinematic:  {pres_time:.3f}s")
        print(f"    Development: {dev_time:.3f}s")
    
    print("\nWait Times (seconds):")
    for wait in wait_times:
        fast_time = fast_scene.get_wait_time(wait)
        pres_time = presentation_scene.get_wait_time(wait)
        dev_time = dev_scene.get_wait_time(wait)
        print(f"  {wait}:")
        print(f"    Quick Demo: {fast_time:.3f}s")
        print(f"    Cinematic:  {pres_time:.3f}s")
        print(f"    Development: {dev_time:.3f}s")


if __name__ == "__main__":
    main()
