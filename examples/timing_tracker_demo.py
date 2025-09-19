#!/usr/bin/env python3
"""Demo script to test the timing tracker functionality."""

import sys
from pathlib import Path

# Add the main project to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from agloviz.core.timing_tracker import TimingTracker, get_timing_tracker, reset_timing_tracker
    from agloviz.core.timing_config import BfsTimingConfig, get_timing_config
    print("✅ Successfully imported timing modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def demo_timing_tracker():
    """Demonstrate the timing tracker functionality."""
    print("🎬 TIMING TRACKER DEMO")
    print("=" * 50)
    
    # Reset tracker to start fresh
    reset_timing_tracker()
    
    # Get timing tracker and config
    tracker = get_timing_tracker()
    
    try:
        config = get_timing_config()
        tracker.set_timing_config(config)
        print(f"✅ Loaded timing config from: {config.config_file}")
        print(f"   Available modes: {', '.join(config.get_available_modes())}")
        print(f"   Current mode: {config.current_mode}")
    except Exception as e:
        print(f"⚠️ Could not load timing config: {e}")
        print("   Continuing with manual timing values...")
    
    print("\n📊 Simulating timing requests...")
    
    # Simulate animation timing requests
    animation_timings = [
        ("title_write", 1.8),
        ("subtitle_write", 1.0),
        ("grid_wave", 0.8),
        ("legend_fade_in", 1.0),
        ("start_indicate", 0.3),
        ("goal_flash", 0.3),
        ("hud_fade_in", 0.8),
        ("queue_fade_in", 1.0),
        ("token_entrance", 0.8),
        ("path_dot_grow", 0.4),
        ("path_line_create", 0.3),
        ("layer_complete_pulse", 0.8),
        ("confetti_effects", 2.0),
        ("tracer_movement", 3.0)
    ]
    
    # Simulate multiple requests for each animation (as would happen in real scene)
    for animation_name, base_time in animation_timings:
        # Simulate different timing modes
        cinematic_time = base_time
        development_time = base_time / 4.0  # 4x faster
        quick_demo_time = base_time / 8.0   # 8x faster
        
        # Track requests as if they came from different parts of the scene
        tracker.track_animation_time(animation_name, cinematic_time)
        tracker.track_animation_time(animation_name, development_time)
        tracker.track_animation_time(animation_name, quick_demo_time)
        
        # Some animations get called multiple times
        if animation_name in ["token_entrance", "path_dot_grow", "path_line_create"]:
            for _ in range(5):  # Simulate multiple tokens/path segments
                tracker.track_animation_time(animation_name, cinematic_time)
    
    # Simulate wait timing requests
    wait_timings = [
        ("initial_setup", 3.0),
        ("educational_pause", 2.0),
        ("celebration_pause", 0.5),
        ("dequeue_delay", 0.3),
        ("layer_complete_wait", 1.0),
        ("path_reconstruction_pause", 1.5),
        ("final_study_pause", 4.0)
    ]
    
    for wait_name, base_time in wait_timings:
        # Simulate different timing modes
        cinematic_time = base_time
        development_time = base_time / 10.0  # 10x faster for waits
        quick_demo_time = base_time / 20.0   # 20x faster for waits
        
        tracker.track_wait_time(wait_name, cinematic_time)
        tracker.track_wait_time(wait_name, development_time)
        
        # Some waits happen multiple times
        if wait_name in ["dequeue_delay", "layer_complete_wait"]:
            for _ in range(10):  # Simulate multiple BFS events
                tracker.track_wait_time(wait_name, cinematic_time)
    
    # Simulate legacy timing requests
    legacy_stages = [
        ("setup", 1.0),
        ("bfs_events", 0.5),
        ("path_drawing", 0.8),
        ("celebrations", 1.2),
        ("educational", 2.0),
        ("waits", 1.5)
    ]
    
    for stage, base_time in legacy_stages:
        # Simulate different speed multipliers
        for multiplier in [1.0, 4.0, 8.0]:  # cinematic, development, quick_demo
            adjusted_time = base_time / multiplier
            tracker.track_legacy_timing(stage, base_time, adjusted_time)
            
            # Some stages get called many times
            if stage in ["bfs_events", "waits"]:
                for _ in range(15):
                    tracker.track_legacy_timing(stage, base_time, adjusted_time)
    
    print(f"✅ Simulated {tracker.total_animation_requests} animation requests")
    print(f"✅ Simulated {tracker.total_wait_requests} wait requests")
    print(f"✅ Simulated {tracker.total_legacy_requests} legacy requests")
    
    # Generate reports
    print("\n🎯 GENERATING REPORTS...")
    
    # Comprehensive report
    report = tracker.generate_comprehensive_report()
    print(report)
    
    # Save to file
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "timing_tracker_demo_report.txt"
    tracker.save_report_to_file(str(report_file), include_detailed=True)
    
    print(f"\n💾 Reports saved to: {report_file}")
    
    # Show some individual stats
    print("\n🔍 INDIVIDUAL ANIMATION STATS:")
    for animation_name in ["title_write", "token_entrance", "tracer_movement"]:
        stats = tracker.get_animation_stats(animation_name)
        print(f"  {animation_name}: {stats['count']} requests, {stats['total_time']:.3f}s total")
    
    print("\n🔍 INDIVIDUAL WAIT STATS:")
    for wait_name in ["initial_setup", "dequeue_delay", "final_study_pause"]:
        stats = tracker.get_wait_stats(wait_name)
        print(f"  {wait_name}: {stats['count']} requests, {stats['total_time']:.3f}s total")
        
    print("\n🔍 LEGACY STAGE STATS:")
    for stage in ["bfs_events", "waits"]:
        stats = tracker.get_legacy_stats(stage)
        print(f"  {stage}: {stats['count']} requests, {stats['total_time']:.3f}s total")


if __name__ == "__main__":
    demo_timing_tracker()
