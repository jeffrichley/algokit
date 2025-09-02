"""Algorithm Kit - A python implementation of control and learning algorithms."""

__version__ = "0.3.0"
__author__ = "Jeff Richley"
__email__ = "jeffrichley@gmail.com"


from algokit.classic_dp import fibonacci


def main_function(input_data: str | None) -> str:
    """Main function for Algorithm Kit.

    Args:
        input_data: Input data to process

    Returns:
        Processed result

    Raises:
        ValueError: If input_data cannot be empty
    """
    if not input_data:
        raise ValueError("input_data cannot be empty")
    return f"Processed: {input_data}"


__all__ = ["main_function", "fibonacci"]
