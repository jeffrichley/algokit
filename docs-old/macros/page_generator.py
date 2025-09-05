"""
Dynamic page generator for MkDocs.

This module generates algorithm pages dynamically during the MkDocs build process,
eliminating the need for static .md files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_algorithms_data() -> Dict[str, Any]:
    """Load algorithms data from YAML file."""
    yaml_path = Path("algorithms.yaml")
    if not yaml_path.exists():
        return {"families": {}, "algorithms": {}}

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def generate_algorithm_page_content(
    algorithm_key: str, algorithm_data: Dict[str, Any], family_data: Dict[str, Any]
) -> str:
    """Generate complete algorithm page content from YAML data."""

    # Extract data with defaults
    name = algorithm_data.get("name", algorithm_key.replace("-", " ").title())
    description = algorithm_data.get("description", "Algorithm description")
    overview = algorithm_data.get("overview", description)
    family_name = family_data.get("name", family_data.get("family", "Unknown Family"))
    status = algorithm_data.get("status", "planned")

    # Complexity information
    complexity = algorithm_data.get("complexity", {})
    time_complexity = complexity.get("time", "O(n)")
    space_complexity = complexity.get("space", "O(n)")

    # Tags
    tags = algorithm_data.get("tags", [family_data.get("family", "algorithm")])
    if isinstance(tags, list):
        tags_str = ", ".join(f'"{tag}"' for tag in tags)
    else:
        tags_str = f'"{tags}"'

    # Mathematical formulation
    math_formulation = algorithm_data.get("mathematical_formulation", {})
    recurrence_relation = math_formulation.get("recurrence_relation", "TBD")
    base_cases = math_formulation.get("base_cases", "TBD")

    # Applications
    applications = algorithm_data.get(
        "applications", ["General algorithm applications"]
    )
    if isinstance(applications, list):
        applications_list = "\n        - ".join(applications)
    else:
        applications_list = str(applications)

    # Implementation info
    implementation = algorithm_data.get("implementation", {})
    source_file = implementation.get("source_file", "TBD")
    approaches = implementation.get("approaches", ["Basic implementation"])
    if isinstance(approaches, list):
        approaches_list = "\n        - ".join(approaches)
    else:
        approaches_list = str(approaches)

    # Status emoji mapping
    status_emoji = {
        "complete": "✅",
        "in-progress": "🔄",
        "planned": "⏳",
        "deprecated": "❌",
    }.get(status, "❓")

    # Generate the complete page content
    page_content = f'''---
tags: [{tags_str}]
title: "{name}"
family: "{algorithm_data.get('family', 'unknown')}"
complexity: "{time_complexity}"
---

# {name}

!!! info "Algorithm Family"
    **Family:** [{family_name}](../../families/{algorithm_data.get('family', 'unknown')}.md)

!!! abstract "Overview"
    {overview}

## Mathematical Formulation

!!! math "Core Algorithm"
    {description}

    **Recurrence Relation:** {recurrence_relation}
    
    **Base Cases:** {base_cases}

!!! success "Key Properties"
    - **Time Complexity**: {time_complexity}
    - **Space Complexity**: {space_complexity}
    - **Status**: {status_emoji} {status.title()}

## Implementation Approaches

=== "Basic Implementation"
    ```python
    def {algorithm_key.replace('-', '_')}(params):
        """Basic implementation of {name}."""
        # TODO: Implement algorithm
        pass
    ```

=== "Optimized Implementation"
    ```python
    def {algorithm_key.replace('-', '_')}_optimized(params):
        """Optimized implementation of {name}."""
        # TODO: Implement optimized version
        pass
    ```

!!! tip "Complete Implementation"
    Find the complete implementation in: `{source_file}`
    
    **Implementation Approaches:**
    - {approaches_list}

## Complexity Analysis

!!! example "**Time & Space Complexity**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    | **Basic** | {time_complexity} | {space_complexity} | Standard implementation |
    | **Optimized** | {time_complexity} | {space_complexity} | Space/time optimized |

!!! warning "Performance Considerations"
    - **Implementation Status**: {status_emoji} {status.title()}
    - **Source File**: `{source_file}`
    - **Testing**: See test files for validation

## Use Cases & Applications

!!! grid "Application Categories"
    !!! grid-item "Primary Applications"
        - {applications_list}

    !!! grid-item "Learning Value"
        - Understanding {family_name.lower()} concepts
        - Algorithm design patterns
        - Optimization techniques
        - Problem-solving approaches

!!! success "Educational Value"
    - **Algorithm Family**: {family_name}
    - **Complexity**: {time_complexity} time, {space_complexity} space
    - **Status**: {status_emoji} {status.title()}
    - **Implementation**: Available in source code

## References & Further Reading

!!! grid "Reference Categories"
    !!! grid-item "Implementation"
        1. **Source Code**: `{source_file}`
        2. **Tests**: See test directory
        3. **Documentation**: This page

    !!! grid-item "Learning Resources"
        4. Algorithm family overview
        5. Related algorithms in {family_name}
        6. Complexity analysis guides

!!! tip "Interactive Learning"
    Try implementing this algorithm yourself! Start with the basic approach, then optimize for better performance. 
    Check the source code for complete implementations with error handling and comprehensive testing.

## Navigation

!!! grid "Related Content"
    !!! grid-item "Family Overview"
        - **[{family_name}](../../families/{algorithm_data.get('family', 'unknown')}.md)** - Family overview and progress

    !!! grid-item "Related Algorithms"
        - See other algorithms in the {family_name} family
        - Check implementation status and progress

    !!! grid-item "Documentation"
        - **[API Reference](../../api.md)** - Complete implementations
        - **[Contributing Guide](../../contributing.md)** - How to add algorithms
        - **[Home](../../index.md)** - Main documentation

---
*Generated dynamically from algorithms.yaml*
'''

    return page_content


def get_algorithm_page_content(algorithm_key: str) -> Optional[str]:
    """Get algorithm page content dynamically from YAML data.

    This function is called by MkDocs Macros to generate page content
    without requiring static .md files.
    """
    data = load_algorithms_data()
    algorithms = data.get("algorithms", {})
    families = data.get("families", {})

    if algorithm_key not in algorithms:
        return None

    algorithm_data = algorithms[algorithm_key]
    family_key = algorithm_data.get("family", "unknown")
    family_data = families.get(family_key, {})

    return generate_algorithm_page_content(algorithm_key, algorithm_data, family_data)


def get_all_algorithm_keys() -> List[str]:
    """Get all algorithm keys from YAML data."""
    data = load_algorithms_data()
    return list(data.get("algorithms", {}).keys())


def get_algorithms_by_family(family_key: str) -> List[str]:
    """Get all algorithm keys for a specific family."""
    data = load_algorithms_data()
    algorithms = data.get("algorithms", {})

    return [
        key
        for key, algo_data in algorithms.items()
        if algo_data.get("family") == family_key
    ]
