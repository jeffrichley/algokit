# YAML Documentation System Guide

## 🎯 Overview

AlgoKit uses a YAML-based documentation generation system where content is defined in YAML files and automatically transformed into beautiful MkDocs pages. This approach provides a **single source of truth** and ensures consistency across all documentation.

## 📂 System Structure

```
mkdocs_plugins/
├── data/
│   ├── dp/                          # Dynamic Programming family
│   │   ├── family.yaml              # Family metadata and overview
│   │   └── algorithms/
│   │       ├── fibonacci.yaml
│   │       ├── coin-change.yaml
│   │       └── ...
│   ├── rl/                          # Reinforcement Learning family
│   │   ├── family.yaml
│   │   └── algorithms/
│   │       ├── q-learning.yaml
│   │       ├── ppo.yaml
│   │       └── ...
│   ├── hierarchical-rl/             # Hierarchical RL family
│   │   ├── family.yaml
│   │   └── algorithms/
│   │       ├── options.yaml
│   │       ├── feudal.yaml
│   │       └── hiro.yaml
│   ├── planning/                    # Pathfinding/Planning family
│   │   ├── family.yaml
│   │   └── algorithms/
│   │       ├── astar.yaml
│   │       ├── bfs.yaml
│   │       └── ...
│   └── shared/
│       ├── tags.yaml                # Global tags
│       └── refs.bib                 # Bibliography
├── gen_pages.py                     # Main generation script
├── gen_families.py                  # Family page generator
└── gen_algorithms.py                # Algorithm page generator
```

## 🔧 How the System Works

### 1. **YAML → Python → Markdown → HTML**

```
family.yaml
    ↓ (read by gen_families.py)
Python dict
    ↓ (rendered with templates)
index.md (generated)
    ↓ (processed by mkdocs)
HTML page
```

### 2. **Build-Time Generation**

When you run `mkdocs build` or `mkdocs serve`:
1. `gen_pages.py` runs first (mkdocs-gen-files plugin)
2. Reads all `family.yaml` and `algorithm/*.yaml` files
3. Generates markdown pages in memory
4. MkDocs processes the generated pages
5. Beautiful HTML docs are created

## 📝 Adding Content to Family Pages

### Edit `family.yaml`

To add content to a family overview page (e.g., RL, HRL, DP):

```yaml
# mkdocs_plugins/data/rl/family.yaml

description: |
  Your detailed family description here.

  ## Additional Sections

  You can use markdown here including:
  - Lists
  - **Bold** and *italic*
  - Code blocks
  - Links

domain_sections:
  - name: "Our Implementations"
    content: |
      ## Algorithms in This Family

      ### Q-Learning (97% coverage)
      Description here...

  - name: "Choosing the Right Algorithm"
    content: |
      !!! tip "Selection Guide"

          Use Q-Learning when:
          - Small discrete spaces
          - Need convergence guarantees
```

### Key Fields

- **description**: Main content for the family page (markdown supported)
- **domain_sections**: Custom sections that appear on the page
- **key_characteristics**: Bullet points about the family
- **common_applications**: Use cases organized by category
- **related_families**: Cross-references to other families

## 📝 Adding Algorithm Pages

### Create YAML File

```yaml
# mkdocs_plugins/data/rl/algorithms/my-algorithm.yaml

slug: my-algorithm
name: My Algorithm
family_id: rl
hidden: false  # Set to true to hide from navigation

summary: "One-sentence summary for cards"

description: |
  Full description with markdown support.

  ## How It Works

  Explain the algorithm here...

  ## When to Use

  - Use case 1
  - Use case 2

implementations:
  - type: "basic"
    name: "Basic Implementation"
    description: "Description of this implementation"
    complexity:
      time: "O(n)"
      space: "O(n)"
    code: |
      # Python code example
      def my_algorithm(input):
          return result

# ... more fields (see existing files for examples)
```

## 🎨 Customization Options

### Admonitions (Callouts)

```yaml
content: |
  !!! info "Information Box"
      Content here

  !!! tip "Helpful Tip"
      Tip content

  !!! warning "Warning"
      Warning content

  !!! danger "Critical"
      Important warning
```

### Code Blocks

```yaml
content: |
  ```python
  def example():
      return "Hello"
  ```
```

### Tables

```yaml
content: |
  | Algorithm | Coverage | Best For |
  |-----------|----------|----------|
  | Q-Learning | 97% | Discrete spaces |
  | PPO | 91% | Continuous control |
```

## ✅ Current Enhancements

### Reinforcement Learning (`rl/family.yaml`)
Added:
- ✅ "Our Implementations" section with all 6 algorithms
- ✅ "Choosing the Right Algorithm" decision guide
- ✅ Coverage statistics for each algorithm
- ✅ RL framework diagram

### Hierarchical RL (`hierarchical-rl/family.yaml`)
Added:
- ✅ "Our Implementations" section with Options, Feudal, HIRO
- ✅ Coverage statistics (95%, 98%, 99%)
- ✅ Algorithm characteristics and best uses

## 🚀 Next Steps

### To Add API Reference Documentation

The proper way is to create algorithm YAML files with comprehensive content:

```yaml
# mkdocs_plugins/data/rl/algorithms/q-learning.yaml

# ... existing content ...

# Add API documentation sections
api_reference:
  quick_start: |
    ```python
    from algokit.algorithms.reinforcement_learning import QLearningAgent

    agent = QLearningAgent(
        state_size=16,
        action_size=4,
        learning_rate=0.1
    )

    # Training loop
    for episode in range(1000):
        # ... training code
    ```

  advanced_usage: |
    ## Hyperparameter Tuning

    Content here...
```

Then update `gen_algorithms.py` to render these API sections!

## 💡 Best Practices

1. **Keep YAML organized**: Use consistent indentation (2 spaces)
2. **Write markdown in content**: Use the `|` literal block scalar
3. **Test locally**: Run `mkdocs serve` to preview
4. **One source of truth**: Don't create standalone markdown files
5. **Use domain_sections**: Add custom content sections as needed
6. **Cross-reference**: Link between related algorithms

## 🔍 Testing Your Changes

```bash
# Preview docs locally
mkdocs serve

# Build docs
mkdocs build

# Check for errors
mkdocs build --strict
```

---

**Remember**: All documentation content should live in YAML files under `mkdocs_plugins/data/`, not in standalone markdown files! The generation system will create the pages automatically. 🎯
