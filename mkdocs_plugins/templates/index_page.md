# Algorithm Kit

**Algorithm Kit** is a comprehensive educational resource providing clean, well-documented implementations of fundamental algorithms in control theory, machine learning, and optimization. Designed for academic instruction and research, this collection offers production-ready Python implementations with detailed explanations, complexity analysis, and practical examples.

Each algorithm is carefully implemented with modern software engineering practices, comprehensive testing, and extensive documentation to serve as both a learning tool and a reliable reference for students and researchers.

[:material-book-open: Documentation](api.md) · [:material-code-braces: Source Code](https://github.com/jeffrichley/algokit) · [:material-github: GitHub](https://github.com/jeffrichley/algokit)

## Algorithm Families

Explore the comprehensive collection of algorithm implementations organized by domain and application area.

| Family | Algorithms | Completion | Status |
| --- | --- | --- | --- |
{% for family in families -%}
| [{{ family.name }}]({{ family.slug }}/index.md) | {{ family.all_algorithms_list }} | {{ family.completion_percentage }}% | {{ family.status_badge }} |
{% endfor %}

**Status legend**: :material-code-tags: Code available · :material-progress-clock: Coming Soon

### Family Overviews

Each algorithm family represents a distinct computational paradigm with specific theoretical foundations and practical applications. Click through the families below to explore the algorithms, their mathematical foundations, and implementation details.

{% if families|length > 1 -%}
<div class="family-overviews-container" style="border: 2px solid var(--md-default-fg-color--light); border-radius: 8px; padding: 16px; margin: 16px 0; background-color: var(--md-default-bg-color--lightest);">
<div class="tabbed-set" data-tabs="1:{{ families|length }}">
{% for family in families -%}
<input id="__tabbed_1_{{ loop.index }}" name="__tabbed_1" type="radio" {% if loop.first %}checked="checked"{% endif %} />
{% endfor %}
<div class="tabbed-labels">
{% for family in families -%}
<label for="__tabbed_1_{{ loop.index }}">{{ family.name }}</label>
{% endfor %}
</div>
<div class="tabbed-content">
{% for family in families -%}
<div class="tabbed-block">
<h4>{{ family.name }}</h4>
<p>{{ family.summary }}</p>

<p><strong>Completion:</strong> {{ family.completion_percentage }}% ({{ family.available_algorithms|length }} of {{ family.all_algorithms|length }} algorithms complete)</p>

{% if family.all_algorithms -%}
<p><strong>Algorithms:</strong></p>
<ul>
{% for algo in family.all_algorithms -%}
{% set algo_status = algo.get('status', {}).get('current', 'planned') -%}
{% if algo_status == 'complete' -%}
<li><a href="{{ family.slug }}/{{ algo.slug }}.md"><strong>{{ algo.name }}</strong></a> ✓ - {{ algo.summary }}</li>
{% elif algo_status == 'in-progress' -%}
<li><a href="{{ family.slug }}/{{ algo.slug }}.md"><em>{{ algo.name }}</em></a> 🚧 - {{ algo.summary }}</li>
{% else -%}
<li>{{ algo.name }} - {{ algo.summary }}</li>
{% endif -%}
{% endfor %}
</ul>
{% else -%}
<p><em>Coming soon - algorithms in development</em></p>
{% endif %}
</div>
{% endfor %}
</div>
</div>
</div>
{% else -%}
{% for family in families -%}
<h4>{{ family.name }}</h4>
<p>{{ family.summary }}</p>

<p><strong>Completion:</strong> {{ family.completion_percentage }}% ({{ family.available_algorithms|length }} of {{ family.all_algorithms|length }} algorithms complete)</p>

{% if family.all_algorithms -%}
<p><strong>Algorithms:</strong></p>
<ul>
{% for algo in family.all_algorithms -%}
{% set algo_status = algo.get('status', {}).get('current', 'planned') -%}
{% if algo_status == 'complete' -%}
<li><a href="{{ family.slug }}/{{ algo.slug }}.md"><strong>{{ algo.name }}</strong></a> ✓ - {{ algo.summary }}</li>
{% elif algo_status == 'in-progress' -%}
<li><a href="{{ family.slug }}/{{ algo.slug }}.md"><em>{{ algo.name }}</em></a> 🚧 - {{ algo.summary }}</li>
{% else -%}
<li>{{ algo.name }} - {{ algo.summary }}</li>
{% endif -%}
{% endfor %}
</ul>
{% else -%}
<p><em>Coming soon - algorithms in development</em></p>
{% endif %}
{% endfor %}
{% endif %}


## Getting Started

### Installation and Setup

To begin using Algorithm Kit in your research or coursework:

- **Install**: `uv pip install -e .` (editable development install)
- **Verify**: `just test` (run the comprehensive test suite)
- **Quality Check**: `just lint` (ensure code quality standards)

=== "Command Line"

    ```bash
    # Install Algorithm Kit in development mode
    uv pip install -e .

    # Verify all implementations with comprehensive tests
    just test

    # Ensure code quality and style compliance
    just lint
    ```

=== "Python"

    ```python
    # Import the package
    import algokit

    # Your algorithm implementations here
    print("Algorithm Kit is ready!")
    ```

!!! tip "Academic Use"
    This resource is designed for educational and research purposes. Each algorithm includes theoretical background, complexity analysis, and practical implementation details suitable for coursework and research projects.

## Key Features

- **Academic Quality**: Rigorous implementations with theoretical foundations and complexity analysis
- **Educational Focus**: Comprehensive documentation designed for learning and teaching
- **Research Ready**: Production-quality code suitable for academic research and publication
- **Comprehensive Testing**: Extensive test suites ensuring correctness and reliability
- **Type Safety**: Full type annotations for better code understanding and IDE support
- **Modular Design**: Clean architecture enabling easy extension and customization

## Implementation Standards

- **Modern Python**: Built with Python 3.12+ and contemporary best practices
- **Rigorous Testing**: pytest with comprehensive coverage ensuring algorithm correctness
- **Code Quality**: Automated linting and formatting for consistent, readable implementations
- **Type Safety**: Complete type annotations for better code understanding and IDE support
- **Documentation**: Professional documentation with mathematical formulations and examples
- **Version Control**: Git-based workflow with automated quality assurance



<!-- ## Get Started

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Quickstart](api.md)**

    Get up and running in 10 minutes

-   :material-cog: **[Installation](https://github.com/jeffrichley/algokit#development)**

    Set up your development environment

-   :material-play: **[Testing](https://github.com/jeffrichley/algokit#development)**

    Run the test suite and quality checks

-   :material-cog-outline: **[Configuration](https://github.com/jeffrichley/algokit#development)**

    Configure your development workflow

-   :material-puzzle: **[Contributing](contributing.md)**

    Contribute to the project

-   :material-shield-check: **[Quality](https://github.com/jeffrichley/algokit#development)**

    Maintain high code quality standards

</div> -->

## Development

For more detailed information about the project architecture, development guidelines, and quality standards, please refer to the project documentation in the main repository.

[:material-code-braces: API Reference](api.md)

Spotted an issue? Edit this page.
