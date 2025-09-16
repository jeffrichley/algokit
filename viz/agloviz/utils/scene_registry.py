"""Scene registry for auto-discovering algorithm scenes.

This module provides a registry system for automatically discovering
and managing algorithm scene classes for visualization.
"""

import importlib
import inspect
import sys
from pathlib import Path
from typing import Type, Dict, List, Any

from rich.console import Console

console = Console()


class SceneRegistry:
    """Registry for auto-discovering algorithm scenes.
    
    This registry automatically scans the manim directory structure
    to find scene classes for different algorithms.
    """
    
    def __init__(self) -> None:
        """Initialize the scene registry."""
        self._scenes: Dict[str, Type[Any]] = {}
        self._aliases: Dict[str, str] = {
            # Common algorithm aliases
            'bfs': 'breadth_first_search',
            'dfs': 'depth_first_search', 
            'astar': 'a_star',
            'dijkstra': 'dijkstra',
            'bellman_ford': 'bellman_ford',
        }
        self._discover_scenes()
    
    def _discover_scenes(self) -> None:
        """Auto-discover scene classes in the scenes directory structure."""
        try:
            # Get the scenes directory path
            scenes_dir = Path(__file__).parent.parent.parent / "scenes"
            
            if not scenes_dir.exists():
                console.print(f"[yellow]Warning: Scenes directory not found: {scenes_dir}[/yellow]")
                return
            
            # Add scenes directory to Python path
            scenes_path = str(scenes_dir.parent)
            if scenes_path not in sys.path:
                sys.path.insert(0, scenes_path)
            
            # Scan for scene files in subdirectories
            for subdir in scenes_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('__'):
                    self._scan_directory(subdir)
                    
        except Exception as e:
            console.print(f"[yellow]Warning: Error discovering scenes: {e}[/yellow]")
    
    def _scan_directory(self, directory: Path) -> None:
        """Scan a directory for scene classes.
        
        Args:
            directory: Directory to scan for scene files
        """
        try:
            for py_file in directory.glob("*.py"):
                if py_file.name.startswith('__'):
                    continue
                    
                self._scan_file(directory, py_file)
                
        except Exception as e:
            console.print(f"[yellow]Warning: Error scanning directory {directory}: {e}[/yellow]")
    
    def _scan_file(self, directory: Path, py_file: Path) -> None:
        """Scan a Python file for scene classes.
        
        Args:
            directory: Parent directory of the file
            py_file: Python file to scan
        """
        try:
            # Construct module name
            relative_path = py_file.relative_to(Path(__file__).parent.parent.parent)
            module_name = str(relative_path.with_suffix('')).replace('/', '.')
            
            # Import the module
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                console.print(f"[dim]Could not import {module_name}: {e}[/dim]")
                return
            
            # Find scene classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_scene_class(obj):
                    algorithm_name = self._extract_algorithm_name(name, py_file.name)
                    if algorithm_name:
                        self._scenes[algorithm_name] = obj
                        console.print(f"[dim]Registered scene: {algorithm_name} -> {name}[/dim]")
                        
        except Exception as e:
            console.print(f"[yellow]Warning: Error scanning file {py_file}: {e}[/yellow]")
    
    def _is_scene_class(self, obj: Type[Any]) -> bool:
        """Check if a class is a scene class.
        
        Args:
            obj: Class to check
            
        Returns:
            True if the class is a scene class
        """
        try:
            # Check if it has the expected scene methods
            required_methods = ['construct', '__init__']
            has_required_methods = all(hasattr(obj, method) for method in required_methods)
            
            # Also check that it's not a base class (like HarborGridScene)
            # Algorithm scenes should have 'Scene' in their name and not be base classes
            class_name = obj.__name__
            is_algorithm_scene = (
                'Scene' in class_name and 
                class_name != 'HarborGridScene' and  # Exclude base class
                not class_name.startswith('Base') and  # Exclude base classes
                has_required_methods
            )
            
            return is_algorithm_scene
        except Exception:
            return False
    
    def _extract_algorithm_name(self, class_name: str, file_name: str) -> str | None:
        """Extract algorithm name from class name or file name.
        
        Args:
            class_name: Name of the scene class
            file_name: Name of the file containing the class
            
        Returns:
            Algorithm name or None if extraction fails
        """
        # Try to extract from file name first (more reliable)
        if '_scene.py' in file_name:
            algorithm_name = file_name.replace('_scene.py', '')
            # Convert kebab-case to snake_case if needed
            algorithm_name = algorithm_name.replace('-', '_').lower()
            return algorithm_name
        
        # Try to extract from class name
        if 'Scene' in class_name:
            algorithm_name = class_name.replace('Scene', '').lower()
            # Convert CamelCase to snake_case
            import re
            algorithm_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', algorithm_name).lower()
            return algorithm_name
        
        # Try other patterns
        if file_name.endswith('.py'):
            algorithm_name = file_name[:-3].replace('-', '_').lower()
            return algorithm_name
        
        return None
    
    def get_scene_class(self, algorithm: str) -> Type[Any] | None:
        """Get scene class for a specific algorithm.
        
        Args:
            algorithm: Algorithm name (e.g., 'bfs', 'dfs')
            
        Returns:
            Scene class or None if not found
        """
        algorithm = algorithm.lower()
        
        # Check if it's an alias first
        if algorithm in self._aliases:
            algorithm = self._aliases[algorithm]
            
        return self._scenes.get(algorithm)
    
    def list_algorithms(self) -> List[str]:
        """List all available algorithms.
        
        Returns:
            List of algorithm names (including aliases)
        """
        # Include both registered scenes and aliases
        algorithms = list(self._scenes.keys())
        algorithms.extend(self._aliases.keys())
        return sorted(set(algorithms))  # Remove duplicates and sort
    
    def list_scenes(self) -> Dict[str, Type[Any]]:
        """List all registered scenes.
        
        Returns:
            Dictionary mapping algorithm names to scene classes
        """
        return self._scenes.copy()
    
    def register_scene(self, algorithm: str, scene_class: Type[Any]) -> None:
        """Manually register a scene class.
        
        Args:
            algorithm: Algorithm name
            scene_class: Scene class to register
        """
        self._scenes[algorithm.lower()] = scene_class
        console.print(f"[green]Manually registered scene: {algorithm} -> {scene_class.__name__}[/green]")
    
    def unregister_scene(self, algorithm: str) -> bool:
        """Unregister a scene class.
        
        Args:
            algorithm: Algorithm name to unregister
            
        Returns:
            True if the algorithm was registered and removed
        """
        if algorithm.lower() in self._scenes:
            del self._scenes[algorithm.lower()]
            console.print(f"[yellow]Unregistered scene: {algorithm}[/yellow]")
            return True
        return False
    
    def refresh(self) -> None:
        """Refresh the scene registry by re-scanning directories."""
        self._scenes.clear()
        self._discover_scenes()
        console.print(f"[blue]Refreshed scene registry. Found {len(self._scenes)} scenes.[/blue]")


# Global registry instance
_global_registry: SceneRegistry | None = None


def get_scene_registry() -> SceneRegistry:
    """Get the global scene registry instance.
    
    Returns:
        Global scene registry instance
    """
    global _global_registry  # noqa: PLW0603
    if _global_registry is None:
        _global_registry = SceneRegistry()
    return _global_registry
