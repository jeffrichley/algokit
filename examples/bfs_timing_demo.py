#!/usr/bin/env python3
"""Demo script showing how to use BFS timing configuration."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from agloviz.core.timing_config import get_timing_config
from agloviz.scenes.planning.bfs.breadth_first_search_scene import BreadthFirstSearchScene


def main():
    """Demonstrate timing configuration usage."""
    print("🎬 BFS Timing Configuration Demo")
    print("=" * 50)
    
    # Load timing configuration
    config = get_timing_config()
    config.print_current_settings()
    print()
    
    # Show available modes
    print("Available timing modes:")
    for mode in config.get_available_modes():
        print(f"  - {mode}")
    print()
    
    # Create BFS scene
    scene = BreadthFirstSearchScene()
    print(f"Scene created with mode: {scene.current_mode}")
    print(f"Max events: {scene.max_events_displayed}")
    print()
    
    # Demonstrate different timing modes
    print("Timing examples for different modes:")
    animation = "title_write"
    wait = "initial_setup"
    
    for mode in config.get_available_modes():
        scene.set_timing_mode(mode)
        anim_time = scene.get_animation_time(animation)
        wait_time = scene.get_wait_time(wait)
        print(f"  {mode}: {animation} = {anim_time:.2f}s, {wait} = {wait_time:.2f}s")
    print()
    
    # Show how to modify configuration
    print("To modify timings, edit:")
    print(f"  {config.config_file}")
    print()
    print("Example modifications:")
    print("  - Change 'max_events_displayed' to limit BFS events")
    print("  - Adjust 'base_timings' values for specific animations")
    print("  - Modify 'speed_multipliers' for different modes")
    print("  - Set 'path_reconstruction.enabled: false' to skip path animation")


if __name__ == "__main__":
    main()
