"""AlgoKit MkDocs Macros Plugin

This module provides the main plugin interface for MkDocs Macros,
including dynamic algorithm page generation hooks.
"""

from .hooks import on_files, on_page_markdown, on_post_build
from .macro_loader import on_startup

# Export the hook functions that MkDocs will call
__all__ = ["on_startup", "on_files", "on_page_markdown", "on_post_build"]
