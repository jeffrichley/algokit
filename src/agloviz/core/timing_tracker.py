"""Timing tracker for monitoring and reporting timing usage in BFS scene animations."""

from collections import defaultdict
from typing import Any


class TimingTracker:
    """Tracks timing requests and generates comprehensive usage reports."""

    def __init__(self):
        """Initialize timing tracker."""
        # Track animation timing requests: {animation_name: [list of requested times]}
        self.animation_requests: dict[str, list[float]] = defaultdict(list)

        # Track wait timing requests: {wait_name: [list of requested times]}
        self.wait_requests: dict[str, list[float]] = defaultdict(list)

        # Track legacy timing requests: {stage_category: {timing_value: count}}
        self.legacy_requests: dict[str, dict[float, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Store timing configuration reference for base values
        self.timing_config = None

        # Track total requests
        self.total_animation_requests = 0
        self.total_wait_requests = 0
        self.total_legacy_requests = 0

    def set_timing_config(self, timing_config: Any) -> None:
        """Set reference to timing configuration for base value lookups.

        Args:
            timing_config: BfsTimingConfig instance
        """
        self.timing_config = timing_config

    def track_animation_time(self, animation_name: str, requested_time: float) -> float:
        """Track an animation timing request.

        Args:
            animation_name: Name of the animation
            requested_time: The time value that was requested/returned

        Returns:
            The same requested_time (passthrough)
        """
        self.animation_requests[animation_name].append(requested_time)
        self.total_animation_requests += 1
        return requested_time

    def track_wait_time(self, wait_name: str, requested_time: float) -> float:
        """Track a wait timing request.

        Args:
            wait_name: Name of the wait
            requested_time: The time value that was requested/returned

        Returns:
            The same requested_time (passthrough)
        """
        self.wait_requests[wait_name].append(requested_time)
        self.total_wait_requests += 1
        return requested_time

    def track_legacy_timing(
        self, stage: str, base_time: float, returned_time: float
    ) -> float:
        """Track a legacy timing request (from _get_timing method).

        Args:
            stage: Stage category (setup, bfs_events, etc.)
            base_time: Base timing value before adjustment
            returned_time: Adjusted timing value that was returned

        Returns:
            The same returned_time (passthrough)
        """
        self.legacy_requests[stage][returned_time] += 1
        self.total_legacy_requests += 1
        return returned_time

    def get_animation_stats(self, animation_name: str) -> dict[str, Any]:
        """Get statistics for a specific animation.

        Args:
            animation_name: Name of the animation

        Returns:
            Dictionary with statistics
        """
        requests = self.animation_requests.get(animation_name, [])
        if not requests:
            return {"count": 0, "total_time": 0.0, "avg_time": 0.0, "base_time": 0.0}

        count = len(requests)
        total_time = sum(requests)
        avg_time = total_time / count

        # Get base time from config if available
        base_time = 0.0
        if self.timing_config:
            try:
                base_time = self.timing_config.get_animation_time(
                    animation_name, "cinematic"
                )
            except Exception:
                base_time = 0.0

        return {
            "count": count,
            "total_time": total_time,
            "avg_time": avg_time,
            "base_time": base_time,
            "all_times": requests.copy(),
        }

    def get_wait_stats(self, wait_name: str) -> dict[str, Any]:
        """Get statistics for a specific wait.

        Args:
            wait_name: Name of the wait

        Returns:
            Dictionary with statistics
        """
        requests = self.wait_requests.get(wait_name, [])
        if not requests:
            return {"count": 0, "total_time": 0.0, "avg_time": 0.0, "base_time": 0.0}

        count = len(requests)
        total_time = sum(requests)
        avg_time = total_time / count

        # Get base time from config if available
        base_time = 0.0
        if self.timing_config:
            try:
                base_time = self.timing_config.get_wait_time(wait_name, "cinematic")
            except Exception:
                base_time = 0.0

        return {
            "count": count,
            "total_time": total_time,
            "avg_time": avg_time,
            "base_time": base_time,
            "all_times": requests.copy(),
        }

    def get_legacy_stats(self, stage: str) -> dict[str, Any]:
        """Get statistics for legacy timing requests by stage.

        Args:
            stage: Stage category

        Returns:
            Dictionary with statistics
        """
        stage_requests = self.legacy_requests.get(stage, {})
        if not stage_requests:
            return {
                "count": 0,
                "total_time": 0.0,
                "unique_times": 0,
                "time_distribution": {},
            }

        total_count = sum(stage_requests.values())
        total_time = sum(time * count for time, count in stage_requests.items())
        unique_times = len(stage_requests)

        return {
            "count": total_count,
            "total_time": total_time,
            "unique_times": unique_times,
            "time_distribution": dict(stage_requests),
        }

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive timing usage report.

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("🎬 COMPREHENSIVE TIMING USAGE REPORT")
        lines.append("=" * 80)

        # Overall summary
        total_requests = (
            self.total_animation_requests
            + self.total_wait_requests
            + self.total_legacy_requests
        )
        lines.append(f"Total Timing Requests: {total_requests}")
        lines.append(f"  - Animation Requests: {self.total_animation_requests}")
        lines.append(f"  - Wait Requests: {self.total_wait_requests}")
        lines.append(f"  - Legacy Requests: {self.total_legacy_requests}")
        lines.append("")

        # Animation timing report
        if self.animation_requests:
            lines.append("🎭 ANIMATION TIMINGS")
            lines.append("-" * 50)
            lines.append(
                f"{'Animation Name':<25} {'Count':<8} {'Base':<8} {'Avg':<8} {'Total':<10}"
            )
            lines.append("-" * 50)

            total_animation_time = 0.0
            for animation_name in sorted(self.animation_requests.keys()):
                stats = self.get_animation_stats(animation_name)
                total_animation_time += stats["total_time"]
                lines.append(
                    f"{animation_name:<25} "
                    f"{stats['count']:<8} "
                    f"{stats['base_time']:<8.3f} "
                    f"{stats['avg_time']:<8.3f} "
                    f"{stats['total_time']:<10.3f}"
                )

            lines.append("-" * 50)
            lines.append(
                f"{'TOTAL ANIMATION TIME':<25} {'':<8} {'':<8} {'':<8} {total_animation_time:<10.3f}"
            )
            lines.append("")

        # Wait timing report
        if self.wait_requests:
            lines.append("⏳ WAIT TIMINGS")
            lines.append("-" * 50)
            lines.append(
                f"{'Wait Name':<25} {'Count':<8} {'Base':<8} {'Avg':<8} {'Total':<10}"
            )
            lines.append("-" * 50)

            total_wait_time = 0.0
            for wait_name in sorted(self.wait_requests.keys()):
                stats = self.get_wait_stats(wait_name)
                total_wait_time += stats["total_time"]
                lines.append(
                    f"{wait_name:<25} "
                    f"{stats['count']:<8} "
                    f"{stats['base_time']:<8.3f} "
                    f"{stats['avg_time']:<8.3f} "
                    f"{stats['total_time']:<10.3f}"
                )

            lines.append("-" * 50)
            lines.append(
                f"{'TOTAL WAIT TIME':<25} {'':<8} {'':<8} {'':<8} {total_wait_time:<10.3f}"
            )
            lines.append("")

        # Legacy timing report
        if self.legacy_requests:
            lines.append("🕰️  LEGACY TIMINGS (by Stage)")
            lines.append("-" * 50)
            lines.append(f"{'Stage':<20} {'Count':<8} {'Unique':<8} {'Total':<10}")
            lines.append("-" * 50)

            total_legacy_time = 0.0
            for stage in sorted(self.legacy_requests.keys()):
                stats = self.get_legacy_stats(stage)
                total_legacy_time += stats["total_time"]
                lines.append(
                    f"{stage:<20} "
                    f"{stats['count']:<8} "
                    f"{stats['unique_times']:<8} "
                    f"{stats['total_time']:<10.3f}"
                )

            lines.append("-" * 50)
            lines.append(
                f"{'TOTAL LEGACY TIME':<20} {'':<8} {'':<8} {total_legacy_time:<10.3f}"
            )
            lines.append("")

        # Grand totals
        grand_total_time = (
            (total_animation_time if self.animation_requests else 0.0)
            + (total_wait_time if self.wait_requests else 0.0)
            + (total_legacy_time if self.legacy_requests else 0.0)
        )

        lines.append("🏁 GRAND TOTALS")
        lines.append("-" * 30)
        lines.append(f"Total Requests: {total_requests}")
        lines.append(f"Total Time: {grand_total_time:.3f} seconds")
        lines.append(
            f"Average Time per Request: {grand_total_time / total_requests if total_requests > 0 else 0:.3f} seconds"
        )
        lines.append("")

        # Timing mode information
        if self.timing_config:
            lines.append("⚙️  TIMING CONFIGURATION")
            lines.append("-" * 30)
            lines.append(f"Current Mode: {self.timing_config.current_mode}")
            lines.append(
                f"Available Modes: {', '.join(self.timing_config.get_available_modes())}"
            )
            lines.append(f"Config File: {self.timing_config.config_file}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_detailed_breakdown(self) -> str:
        """Generate a detailed breakdown showing all individual requests.

        Returns:
            Detailed breakdown string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("🔍 DETAILED TIMING BREAKDOWN")
        lines.append("=" * 80)

        # Animation details
        if self.animation_requests:
            lines.append("🎭 ANIMATION TIMING DETAILS")
            lines.append("-" * 50)
            for animation_name in sorted(self.animation_requests.keys()):
                stats = self.get_animation_stats(animation_name)
                lines.append(f"\n{animation_name}:")
                lines.append(f"  Count: {stats['count']}")
                lines.append(f"  Base Time: {stats['base_time']:.3f}s")
                lines.append(f"  Total Time: {stats['total_time']:.3f}s")
                lines.append(f"  Average Time: {stats['avg_time']:.3f}s")
                lines.append(
                    f"  All Requests: {[f'{t:.3f}' for t in stats['all_times']]}"
                )

        # Wait details
        if self.wait_requests:
            lines.append("\n⏳ WAIT TIMING DETAILS")
            lines.append("-" * 50)
            for wait_name in sorted(self.wait_requests.keys()):
                stats = self.get_wait_stats(wait_name)
                lines.append(f"\n{wait_name}:")
                lines.append(f"  Count: {stats['count']}")
                lines.append(f"  Base Time: {stats['base_time']:.3f}s")
                lines.append(f"  Total Time: {stats['total_time']:.3f}s")
                lines.append(f"  Average Time: {stats['avg_time']:.3f}s")
                lines.append(
                    f"  All Requests: {[f'{t:.3f}' for t in stats['all_times']]}"
                )

        # Legacy details
        if self.legacy_requests:
            lines.append("\n🕰️  LEGACY TIMING DETAILS")
            lines.append("-" * 50)
            for stage in sorted(self.legacy_requests.keys()):
                stats = self.get_legacy_stats(stage)
                lines.append(f"\n{stage}:")
                lines.append(f"  Total Count: {stats['count']}")
                lines.append(f"  Unique Times: {stats['unique_times']}")
                lines.append(f"  Total Time: {stats['total_time']:.3f}s")
                lines.append("  Time Distribution:")
                for time_value, count in sorted(stats["time_distribution"].items()):
                    lines.append(
                        f"    {time_value:.3f}s × {count} = {time_value * count:.3f}s"
                    )

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all tracking data."""
        self.animation_requests.clear()
        self.wait_requests.clear()
        self.legacy_requests.clear()
        self.total_animation_requests = 0
        self.total_wait_requests = 0
        self.total_legacy_requests = 0

    def save_report_to_file(self, filename: str, include_detailed: bool = True) -> None:
        """Save timing report to a file.

        Args:
            filename: Output filename
            include_detailed: Whether to include detailed breakdown
        """
        with open(filename, "w") as f:
            f.write(self.generate_comprehensive_report())
            if include_detailed:
                f.write("\n\n")
                f.write(self.generate_detailed_breakdown())

        print(f"📊 Timing report saved to: {filename}")


# Global timing tracker instance
_timing_tracker: TimingTracker | None = None


def get_timing_tracker() -> TimingTracker:
    """Get global timing tracker instance.

    Returns:
        TimingTracker instance
    """
    global _timing_tracker
    if _timing_tracker is None:
        _timing_tracker = TimingTracker()
    return _timing_tracker


def reset_timing_tracker() -> None:
    """Reset the global timing tracker."""
    global _timing_tracker
    if _timing_tracker is not None:
        _timing_tracker.reset()
