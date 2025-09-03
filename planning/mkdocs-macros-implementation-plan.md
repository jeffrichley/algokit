# MkDocs Macros Automation Implementation Plan

## 🎯 **Project Overview**

**Goal**: Automate the generation of algorithm documentation using MkDocs Macros to eliminate manual navigation updates, ensure consistency, and scale to 40+ algorithms with minimal maintenance overhead.

**Current State**: Manual documentation with inconsistent navigation, requiring updates to multiple files when adding new algorithms.

**Target State**: Fully automated documentation system where adding a new algorithm to `algorithms.yaml` automatically generates all navigation, cross-references, and consistent structure.

---

## 🏗️ **System Architecture**

### **1. Data Layer (`algorithms.yaml`)**
```yaml
# Central source of truth for all algorithm metadata
algorithm_families:
  dynamic_programming:
    name: "Dynamic Programming"
    description: "Optimization through recursive problem decomposition"
    slug: "dynamic-programming"
    algorithms:
      fibonacci:
        title: "Fibonacci Sequence"
        implemented: true
        status: "complete"
        complexity: "O(n)"
        source_file: "src/algokit/dynamic_programming/fibonacci.py"
        test_file: "tests/dynamic_programming/test_fibonacci.py"
        tags: ["recursion", "memoization", "sequence"]
        use_cases: ["computer science", "finance", "biology"]
        last_updated: "2024-01-15"
        priority: "high"
        dependencies: []
        notes: "Classic DP introduction problem"
```

### **2. Macro Layer (Custom Python Functions)**
```python
# Custom macros that read algorithms.yaml and generate content
def generate_algorithm_navigation(family_slug: str) -> str:
    """Generate navigation grid for algorithm family."""

def generate_related_algorithms(algorithm_name: str, family_slug: str) -> str:
    """Generate related algorithms section."""

def generate_family_overview(family_slug: str) -> str:
    """Generate family overview link."""

def generate_algorithm_families() -> str:
    """Generate links to all algorithm families."""

def generate_documentation_links() -> str:
    """Generate documentation navigation links."""
```

### **3. Template Layer (Markdown with Macro Calls)**
```markdown
# Algorithm pages use macros instead of hardcoded content
## Navigation

{{ generate_algorithm_navigation("dynamic-programming") }}

{{ generate_related_algorithms("fibonacci", "dynamic-programming") }}

{{ generate_family_overview("dynamic-programming") }}

{{ generate_algorithm_families() }}

{{ generate_documentation_links() }}
```

### **4. Output Layer (Generated Documentation)**
- **Consistent navigation** across all algorithm pages
- **Automatic cross-references** between related algorithms
- **Dynamic family overviews** that stay current
- **Scalable structure** that grows with your algorithm library

---

## 🔄 **Workflow & Data Flow**

### **Phase 1: Setup & Configuration**
1. **Install MkDocs Macros** and configure the plugin
2. **Create algorithms.yaml** with current implemented algorithms
3. **Build custom macro functions** for navigation generation
4. **Test with existing algorithms** to ensure compatibility

### **Phase 2: Template Migration**
1. **Update algorithm template** to use macro calls
2. **Migrate existing algorithm pages** to use macros
3. **Test navigation generation** across all pages
4. **Verify consistency** and fix any issues

### **Phase 3: Automation & Scaling**
1. **Add new algorithms** to algorithms.yaml
2. **Generate documentation** automatically
3. **Add planned algorithms** with status tracking
4. **Build progress tracking** and reporting

---

## 🛠️ **Implementation Steps**

### **Step 1: Environment Setup**
```bash
# Install MkDocs Macros
uv add mkdocs-macros-plugin

# Configure mkdocs.yml
plugins:
  - macros:
      include_dir: macros/
      include_yaml: algorithms.yaml
```

### **Step 2: Create Data Structure**
```yaml
# algorithms.yaml - Start with current implementations
algorithm_families:
  dynamic_programming:
    name: "Dynamic Programming"
    slug: "dynamic-programming"
    algorithms:
      fibonacci:
        title: "Fibonacci Sequence"
        implemented: true
        status: "complete"
        complexity: "O(n)"
        source_file: "src/algokit/dynamic_programming/fibonacci.py"
        test_file: "tests/dynamic_programming/test_fibonacci.py"
        tags: ["recursion", "memoization", "sequence"]
        use_cases: ["computer science", "finance", "biology"]
        last_updated: "2024-01-15"
        priority: "high"
        dependencies: []
        notes: "Classic DP introduction problem"
```

### **Step 3: Build Custom Macros**
```python
# macros/navigation.py
from typing import Dict, List, Any
import yaml

def load_algorithms_data() -> Dict[str, Any]:
    """Load algorithms.yaml data."""
    with open("algorithms.yaml", "r") as f:
        return yaml.safe_load(f)

def generate_algorithm_navigation(family_slug: str) -> str:
    """Generate navigation grid for algorithm family."""
    data = load_algorithms_data()
    family = data["algorithm_families"][family_slug]

    # Generate navigation HTML/Markdown
    navigation_html = f"""
    !!! grid "Related Content"
        !!! grid-item "Family Overview"
            - **[{family['name']} Family](../../families/{family_slug}.md)** - Complete overview of all {family['name']} algorithms

        !!! grid-item "Related Algorithms"
    """

    # Add related algorithms
    for alg_name, alg_data in family["algorithms"].items():
        if alg_name != current_algorithm:
            navigation_html += f"""
            - **[{alg_data['title']}](./{alg_name}.md)** - {alg_data.get('notes', '')}
            """

    return navigation_html
```

### **Step 4: Update Algorithm Template**
```markdown
# docs/templates/algorithm-template.md
---
title: "REPLACE: Algorithm Title"
tags: [REPLACE: tags]
---

# REPLACE: Algorithm Title

!!! info "Algorithm Family"
    **Family:** [REPLACE: Family Name](../../families/REPLACE: family-slug.md)

## Overview

REPLACE: Algorithm description

## Navigation

{{ generate_algorithm_navigation("REPLACE: family-slug") }}
```

### **Step 5: Migrate Existing Pages**
```bash
# Update all algorithm pages to use macros
find docs/algorithms -name "*.md" -exec sed -i 's|## Navigation.*|{{ generate_algorithm_navigation("family-slug") }}|g' {} \;
```

---

## 📊 **Data Schema & Relationships**

### **Algorithm Family Structure**
```yaml
algorithm_families:
  [family_slug]:
    name: "Human Readable Name"
    description: "Family description"
    slug: "url-friendly-slug"
    algorithms:
      [algorithm_slug]:
        title: "Human Readable Title"
        implemented: boolean
        status: "complete|planned|in_progress|deprecated"
        complexity: "O(n), O(n²), etc."
        source_file: "path/to/source.py"
        test_file: "path/to/test.py"
        tags: [list, of, tags]
        use_cases: [list, of, use, cases]
        last_updated: "YYYY-MM-DD"
        priority: "high|medium|low"
        dependencies: [list, of, dependencies]
        notes: "Additional notes"
```

### **Cross-Reference Relationships**
- **Family → Algorithms**: One-to-many relationship
- **Algorithm → Dependencies**: Many-to-many relationship
- **Algorithm → Tags**: Many-to-many relationship
- **Algorithm → Use Cases**: Many-to-many relationship

---

## 🎨 **Generated Content Examples**

### **Navigation Grid Output**
```markdown
!!! grid "Related Content"
    !!! grid-item "Family Overview"
        - **[Dynamic Programming Family](../../families/dynamic-programming.md)** - Complete overview of all Dynamic Programming algorithms

    !!! grid-item "Related Algorithms"
        - **[Coin Change Problem](./coin-change.md)** - Minimum coins optimization
        - **[0/1 Knapsack Problem](./knapsack.md)** - Resource allocation optimization
        - **[Longest Common Subsequence](./longest-common-subsequence.md)** - Sequence comparison

    !!! grid-item "Algorithm Families"
        - **[Reinforcement Learning](../../families/reinforcement-learning.md)** - Value-based and policy-based learning
        - **[Control Algorithms](../../families/control.md)** - PID, adaptive, and robust control
        - **[Planning Algorithms](../../families/planning.md)** - Pathfinding and decision making

    !!! grid-item "Documentation"
        - **[API Reference](../../api.md)** - Complete API documentation
        - **[Contributing Guide](../../contributing.md)** - How to contribute
        - **[Home](../../index.md)** - Return to main page
```

### **Family Overview Output**
```markdown
## Dynamic Programming Algorithms

| Algorithm | Status | Complexity | Priority | Last Updated |
|-----------|--------|------------|----------|--------------|
| [Fibonacci Sequence](./fibonacci.md) | ✅ Complete | O(n) | High | 2024-01-15 |
| [Coin Change Problem](./coin-change.md) | 🔄 In Progress | O(n×amount) | High | 2024-01-10 |
| [0/1 Knapsack Problem](./knapsack.md) | 📋 Planned | O(n×W) | Medium | - |
```

---

## 🚀 **Advanced Features & Future Enhancements**

### **Progress Tracking**
```yaml
# Automatically generated progress metrics
progress:
  total_algorithms: 40
  implemented: 15
  in_progress: 8
  planned: 17
  completion_rate: "37.5%"

  by_family:
    dynamic_programming:
      total: 6
      implemented: 1
      completion_rate: "16.7%"
    reinforcement_learning:
      total: 6
      implemented: 0
      completion_rate: "0%"
```

### **Dependency Management**
```yaml
# Track algorithm dependencies
dependencies:
  matrix_chain_multiplication:
    requires: ["dynamic_programming_framework", "matrix_operations"]
    provides: ["optimal_parenthesization"]
    conflicts: []
```

### **Automated Testing**
```yaml
# Link documentation to test coverage
testing:
  fibonacci:
    test_coverage: "95%"
    test_file: "tests/dynamic_programming/test_fibonacci.py"
    last_test_run: "2024-01-15"
    test_status: "passing"
```

---

## 🔧 **Technical Implementation Details**

### **Macro Function Structure**
```python
# Each macro function follows this pattern:
def macro_function_name(parameters: str) -> str:
    """
    Generate content based on algorithms.yaml data.

    Args:
        parameters: Macro parameters from markdown

    Returns:
        Generated markdown/HTML content
    """
    try:
        # Load data
        data = load_algorithms_data()

        # Process data
        result = process_data(data, parameters)

        # Generate output
        return generate_output(result)

    except Exception as e:
        # Graceful fallback
        return f"<!-- Error generating content: {e} -->"
```

### **Error Handling & Fallbacks**
- **Missing data**: Graceful fallback to static content
- **Invalid parameters**: Default to safe values
- **File errors**: Log errors and continue
- **Performance**: Cache data loading for speed

### **Performance Considerations**
- **Lazy loading**: Only load data when macros are called
- **Caching**: Cache parsed YAML data during build
- **Incremental builds**: Only regenerate changed content
- **Memory usage**: Stream large data files if needed

---

## 📈 **Migration Strategy**

### **Phase 1: Foundation (Week 1)**
- [ ] Install and configure MkDocs Macros
- [ ] Create algorithms.yaml with current algorithms
- [ ] Build basic navigation macros
- [ ] Test with one algorithm family

### **Phase 2: Core Implementation (Week 2)**
- [ ] Complete all navigation macros
- [ ] Update algorithm template
- [ ] Migrate Dynamic Programming pages
- [ ] Test full navigation system

### **Phase 3: Expansion (Week 3)**
- [ ] Migrate remaining algorithm families
- [ ] Add progress tracking macros
- [ ] Implement dependency management
- [ ] Add automated testing integration

### **Phase 4: Optimization (Week 4)**
- [ ] Performance optimization
- [ ] Error handling improvements
- [ ] Documentation and training
- [ ] Future roadmap planning

---

## 🎯 **Success Metrics**

### **Immediate Goals**
- [ ] **100% automation** of navigation generation
- [ ] **Zero manual updates** required for new algorithms
- [ ] **Consistent structure** across all algorithm pages
- [ ] **Faster documentation** updates (5x improvement)

### **Long-term Goals**
- [ ] **Scalable to 100+ algorithms** without maintenance overhead
- [ ] **Progress tracking** and reporting automation
- [ ] **Dependency management** and impact analysis
- [ ] **Integration with CI/CD** for automated documentation updates

---

## 🚨 **Risk Mitigation**

### **Technical Risks**
- **Macro complexity**: Start simple, add complexity gradually
- **Performance issues**: Monitor build times, optimize as needed
- **Data consistency**: Validate YAML structure, add error checking

### **Process Risks**
- **Learning curve**: Provide clear examples and documentation
- **Migration effort**: Phase the migration to minimize disruption
- **Maintenance overhead**: Ensure the system reduces, not increases, work

### **Mitigation Strategies**
- **Incremental rollout**: Test with one family before expanding
- **Fallback content**: Always provide static alternatives
- **Comprehensive testing**: Test all scenarios before production
- **Documentation**: Clear guides for future maintenance

---

## 🔮 **Future Vision**

### **Year 1: Foundation**
- Complete automation of current algorithm documentation
- Integration with development workflow
- Progress tracking and reporting

### **Year 2: Enhancement**
- Advanced dependency management
- Automated testing integration
- Performance optimization

### **Year 3: Expansion**
- Integration with external tools
- Advanced analytics and insights
- Community contribution tools

---

## 📚 **Resources & References**

### **Documentation**
- [MkDocs Macros Documentation](https://mkdocs-macros-plugin.readthedocs.io/)
- [YAML Specification](https://yaml.org/spec/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)

### **Examples & Templates**
- [Navigation Macro Examples](./examples/navigation-macros.md)
- [Algorithm Template](./templates/algorithm-template.md)
- [Family Overview Template](./templates/family-overview.md)

### **Tools & Utilities**
- [YAML Validator](https://www.yamllint.com/)
- [Markdown Preview](https://stackedit.io/)
- [MkDocs Build Tools](https://www.mkdocs.org/user-guide/configuration/)

---

**This plan provides a comprehensive roadmap for implementing a fully automated documentation system that will scale with your algorithm library while maintaining quality and consistency.**
