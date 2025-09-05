# Algorithm Documentation Data Schema

This directory contains the structured data for algorithm families and individual algorithms used by the MkDocs documentation system.

## Directory Structure

```
mkdocs_plugins/data/
├── README.md                    # This file - schema documentation
├── shared/                      # Shared resources
│   ├── tags.yaml               # Tag definitions for categorization
│   └── refs.bib                # Bibliography references
└── families/                    # Algorithm family data
    └── {family_id}/            # Individual family directories
        ├── family.yaml         # Family metadata and configuration
        └── algorithms/         # Individual algorithm definitions
            └── {algorithm_id}.yaml
```

## Family Schema (`family.yaml`)

The family schema supports rich metadata and structured content for algorithm families:

### Core Fields

- **`id`**: Unique identifier (e.g., "dp", "rl", "control")
- **`name`**: Human-readable name (e.g., "Dynamic Programming")
- **`slug`**: URL-friendly identifier (e.g., "dynamic-programming")
- **`summary`**: Brief one-sentence description for cards and navigation
- **`description`**: Detailed description (markdown supported) for the full family page

### Characteristics and Applications

- **`key_characteristics`**: List of defining features with importance levels
- **`common_applications`**: Categorized list of use cases and examples
- **`concepts`**: Key concepts and terminology with types

### Algorithm Management

- **`algorithms`**: Configuration for algorithm discovery and ordering
  - `order_mode`: How to sort algorithms (by_algo_order, by_name, by_slug, by_complexity)
  - `include`/`exclude`: Filter algorithms
  - `comparison`: Enable comparison tables with metrics

### Relationships

- **`related_families`**: Cross-references to other algorithm families
- **`tags`**: Categorization tags for search and filtering

### Status and Progress

- **`status`**: Inferred from algorithm statuses (no manual field needed)
  - "complete" if all algorithms are complete
  - "in-progress" if any algorithms are in-progress
  - "planned" if all algorithms are planned
- **`complexity`**: Typical time/space complexity information

### Custom Content

- **`domain_sections`**: Family-specific content sections (markdown)
- **`references`**: Pointers to entries in `shared/refs.bib` (not embedded data)
- **`template_options`**: Control what sections to display

### Metadata

- **`meta`**: Creation, version, and author information (last_updated is automatic)

## Tag System (`shared/tags.yaml`)

Tags provide flexible categorization across families and algorithms:

### Tag Categories

1. **Algorithm Families**: Core algorithm paradigms (dp, rl, control, etc.)
2. **Algorithm Characteristics**: Implementation approaches (recursive, iterative, greedy, etc.)
3. **Application Domains**: Use case areas (robotics, bioinformatics, finance, etc.)

### Tag Structure

```yaml
- id: "unique-identifier"
  name: "Human Readable Name"
  description: "Detailed description of what this tag represents"
```

### Reference System (`shared/refs.bib`)

References use standard BibTeX format and are referenced by key:

```yaml
# In family.yaml or algorithm.yaml
references:
  - bib_key: "cormen_dp"  # Points to @book{cormen_dp, ...} in refs.bib
  - bib_key: "wikipedia_dp"  # Points to @misc{wikipedia_dp, ...} in refs.bib
```

## Algorithm Schema (`algorithms/{algorithm_id}.yaml`)

The algorithm schema supports comprehensive metadata and structured content for individual algorithms:

### Core Fields

- **`slug`**: Unique identifier (e.g., "fibonacci", "coin-change")
- **`name`**: Human-readable name (e.g., "Fibonacci Sequence")
- **`family_id`**: Parent family identifier (e.g., "dp", "rl")
- **`aliases`**: Alternative names and abbreviations
- **`order`**: Display order within family
- **`summary`**: Brief one-sentence description for cards and navigation
- **`description`**: Detailed description (markdown supported) for the full algorithm page

### Problem Definition

- **`formulation`**: Mathematical formulation including recurrence relations
- **`properties`**: Key properties and characteristics with importance levels
- **`mathematical_properties`**: Mathematical formulas and relationships

### Implementation Details

- **`implementations`**: Multiple implementation approaches with:
  - Code examples with syntax highlighting
  - Complexity analysis (time/space)
  - Advantages and disadvantages
  - Use case recommendations

### Analysis and Applications

- **`complexity`**: Comprehensive complexity analysis for all approaches
- **`applications`**: Categorized use cases and real-world examples
- **`educational_value`**: Learning objectives and educational benefits

### Status and Development

- **`status`**: Implementation status and quality metrics
- **`source_files`**: Links to actual source code and tests
- **`references`**: Pointers to entries in `shared/refs.bib`
- **`related_algorithms`**: Cross-references to other algorithms

### Template Control

- **`template_options`**: Control what sections are rendered
- **`tags`**: Categorization tags for search and filtering

## Usage in Templates

The family and algorithm data is consumed by Jinja2 templates to generate:

- **Family overview pages**: Rich descriptions, comparison tables, algorithm lists
- **Algorithm detail pages**: Comprehensive algorithm documentation with code examples
- **Navigation**: Automatic family and algorithm discovery
- **Cross-references**: Related families, tags, and algorithms
- **Search**: Tag-based filtering and categorization

## Extensibility

The schema is designed to be extensible:

- **Custom sections**: Add domain-specific content via `domain_sections`
- **New tags**: Easily add new categorization tags
- **Template options**: Control rendering without changing data
- **Versioning**: Track changes via `meta` fields

## Examples

See `families/dp/family.yaml` for a complete example of the Dynamic Programming family schema.

## Future Enhancements

- **Validation**: YAML schema validation for data integrity
- **Migration tools**: Convert from old documentation format
- **Import/Export**: Tools for bulk data management
- **Visualization**: Generate diagrams from family relationships
