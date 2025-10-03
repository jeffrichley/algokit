import shutil

from manim import Text

# You can tweak this to include other fallback fonts as needed
DEFAULT_FONT_STACK = ["Fira Mono", "Courier New", "Monospace"]


def get_font_from_stack(stack: list[str] = DEFAULT_FONT_STACK) -> str:
    for font in stack:
        if shutil.which("fc-match"):  # Unix systems with fontconfig
            # `fc-match` returns font file path if installed
            from subprocess import PIPE, run

            result = run(
                ["fc-match", "-f", "%{family}", font],
                check=False,
                stdout=PIPE,
                text=True,
            )
            if result.stdout.strip().lower().startswith(font.lower()):
                return font
        else:
            # Basic fallback (works for Mac with installed fonts)
            try:
                Text("test", font=font)
                return font
            except Exception:
                continue
    return "sans-serif"  # ultimate fallback


def get_font(role: str = "default") -> str:
    """Get a font for the given role."""
    font_roles = {
        "hud": get_font_from_stack(["Fira Mono", "Courier New", "Monaco"]),
        "title": get_font_from_stack(["Arial", "Helvetica", "sans-serif"]),
        "default": get_font_from_stack(),
    }
    return font_roles.get(role, font_roles["default"])
