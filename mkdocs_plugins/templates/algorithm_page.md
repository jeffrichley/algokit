---
algorithm_key: "{{ algo.slug }}"
tags: {{ algo.tags }}
title: "{{ algo.name }}"
family: "{{ family.slug }}"
---

!!! info "{{ algo.name }}"
    {{ algo.summary }}
    
    **Family:** {{ family.name }}  
    **Status:** {% if algo.status and algo.status.current == 'complete' %}✅ Complete{% elif algo.status and algo.status.current == 'in-progress' %}🚧 In Progress{% else %}📋 Planned{% endif %}

!!! abstract "Overview"
    {{ algo.description }}

{% if algo.formulation %}
## Mathematical Formulation

{% if algo.formulation.recurrence_relation %}
!!! math "Recurrence Relation"
    $$
    F(n) = \begin{cases}
    0 & \text{if } n = 0 \\
    1 & \text{if } n = 1 \\
    F(n-1) + F(n-2) & \text{if } n > 1
    \end{cases}
    $$
{% endif %}

{% if algo.formulation.mathematical_properties %}
!!! success "Mathematical Properties"
    {% for prop in algo.formulation.mathematical_properties %}
    **{{ prop.name }}**
    
    `{{ prop.formula }}`
    
    {{ prop.description }}
    
    ---
    {% endfor %}
{% endif %}
{% endif %}

{% if algo.properties %}
## Key Properties

<div class="grid cards" markdown>

{% for prop in algo.properties %}
-   :material-{{ "check-circle" if prop.importance == "fundamental" else "information" }}: **{{ prop.name }}**

    ---

    {{ prop.description }}
{% endfor %}

</div>
{% endif %}

{% if algo.implementations %}
## Implementation Approaches

{% for impl in algo.implementations %}
=== "{{ impl.name }}"
    {% if impl.description %}
    {{ impl.description }}
    {% endif %}
    
    {% if impl.complexity %}
    **Complexity:**
    - **Time**: {{ impl.complexity.time }}
    - **Space**: {{ impl.complexity.space }}
    {% endif %}
    
    {% if impl.code %}
    ```python
    {{ impl.code }}
    ```
    {% endif %}
    
    {% if impl.advantages %}
    !!! success "Advantages"
        {% for advantage in impl.advantages %}
        - {{ advantage }}
        {% endfor %}
    {% endif %}
    
    {% if impl.disadvantages %}
    !!! warning "Disadvantages"
        {% for disadvantage in impl.disadvantages %}
        - {{ disadvantage }}
        {% endfor %}
    {% endif %}
{% endfor %}
{% endif %}

{% if algo.status and algo.status.source_files %}
!!! tip "Complete Implementation"
    The full implementation with error handling, comprehensive testing, and additional variants is available in the source code:

    {% for source_file in algo.status.source_files %}
    - **{{ source_file.description }}**: [`{{ source_file.path }}`](https://github.com/jeffrichley/algokit/blob/main/{{ source_file.path }})
    {% endfor %}
{% endif %}

{% if algo.complexity and algo.complexity.analysis %}
## Complexity Analysis

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    {% for analysis in algo.complexity.analysis %}
    | **{{ analysis.approach }}** | {{ analysis.time }} | {{ analysis.space }} | {{ analysis.notes }} |
    {% endfor %}

{% if algo.complexity.performance_notes %}
!!! warning "Performance Considerations"
    {% for note in algo.complexity.performance_notes %}
    - {{ note }}
    {% endfor %}
{% endif %}
{% endif %}

{% if algo.applications %}
## Use Cases & Applications

!!! grid "Application Categories"
    {% for app in algo.applications %}
    !!! grid-item "{{ app.category }}"
        {% for example in app.examples %}
        - **{{ example.split(':')[0] }}**: {{ example.split(':')[1] if ':' in example else example }}
        {% endfor %}
    {% endfor %}
{% endif %}

{% if algo.educational_value %}
!!! success "Educational Value"
    {% for value in algo.educational_value %}
    - **{{ value.split(':')[0] }}**: {{ value.split(':')[1] if ':' in value else value }}
    {% endfor %}
{% endif %}

{% if algo.references %}
## References & Further Reading

!!! tip "Interactive Learning"
    Try implementing the different approaches yourself! This progression will give you deep insight into the algorithm's principles and applications.
{% endif %}

## Navigation

<div class="nav-grid" style="border: 1px solid var(--md-default-fg-color--light); border-radius: 8px; padding: 16px; margin: 16px 0; background-color: var(--md-default-bg-color--lightest);">

**Related Algorithms in {{ family.name }}:**

{% for related_algo in algorithms if related_algo.slug != algo.slug %}
- [{{ related_algo.name }}]({{ related_algo.slug }}.md) - {{ related_algo.summary }}
{% endfor %}

</div>
