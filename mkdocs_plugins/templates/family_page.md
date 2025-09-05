# {{ family.name }} Algorithms

!!! info "{{ family.summary }}"
    {{ family.description }}

## Overview

**Key Characteristics:**

<div class="grid cards" markdown>

{% for char in family.key_characteristics %}
-   :material-{{ "check-circle" if char.importance == "fundamental" else "information" }}: **{{ char.name }}**

    ---

    {{ char.description }}
{% endfor %}

</div>

**Common Applications:**

{% if family.common_applications %}
{% for app in family.common_applications %}
=== "{{ app.category }}"
    {% for example in app.examples %}
    - {{ example }}
    {% endfor %}
{% endfor %}
{% else %}
!!! info "Applications"
    No applications specified.
{% endif %}

## Key Concepts

<div class="grid cards" markdown>

{% for concept in family.concepts %}
-   :material-{{ "lightbulb" if concept.type == "concept" else "cog" if concept.type == "technique" else "function" }}: **{{ concept.name }}**

    ---

    {{ concept.description }}
{% endfor %}

</div>

{% if family.template_options.show_complexity_analysis and family.complexity %}
## Complexity Analysis

!!! abstract "Complexity Overview"
    **Time**: {{ family.complexity.typical_time }}  
    **Space**: {{ family.complexity.typical_space }}
    
    {% if family.complexity.notes %}
    *{{ family.complexity.notes }}*
    {% endif %}
{% endif %}

{% if family.template_options.custom_sections and family.domain_sections %}
{% for section in family.domain_sections %}
=== "{{ section.name }}"

    {{ section.content }}

{% endfor %}
{% endif %}

{% if family.algorithms.comparison.enabled %}
## Comparison Table

{% set metrics = family.algorithms.comparison.metrics %}
| Algorithm{% if 'status' in metrics %} | Status{% endif %}{% if 'time_complexity' in metrics %} | Time Complexity{% endif %}{% if 'space_complexity' in metrics %} | Space Complexity{% endif %}{% if 'difficulty' in metrics %} | Difficulty{% endif %}{% if 'applications' in metrics %} | Applications{% endif %} |
|-----------{% if 'status' in metrics %}|--------{% endif %}{% if 'time_complexity' in metrics %}|---------------{% endif %}{% if 'space_complexity' in metrics %}|----------------{% endif %}{% if 'difficulty' in metrics %}|----------{% endif %}{% if 'applications' in metrics %}|-------------{% endif %}|
{% for algo in algorithms %}| **{{ algo.name }}**{% if 'status' in metrics %} | {% if algo.status.current == 'complete' %}✅ Complete{% elif algo.status.current == 'in_progress' %}🚧 In Progress{% elif algo.status.current == 'planned' %}📋 Planned{% else %}❓ Unknown{% endif %}{% endif %}{% if 'time_complexity' in metrics %} | {% if algo.complexity and algo.complexity.analysis %}{{ algo.complexity.analysis[0].time }}{% else %}Varies{% endif %}{% endif %}{% if 'space_complexity' in metrics %} | {% if algo.complexity and algo.complexity.analysis %}{{ algo.complexity.analysis[0].space }}{% else %}Varies{% endif %}{% endif %}{% if 'difficulty' in metrics %} | {% if algo.difficulty %}{{ algo.difficulty }}{% else %}Medium{% endif %}{% endif %}{% if 'applications' in metrics %} | {% if algo.applications %}{{ algo.applications[0].category }}{% if algo.applications|length > 1 %}, {{ algo.applications[1].category }}{% endif %}{% else %}General applications{% endif %}{% endif %} |
{% endfor %}
{% endif %}

## Algorithms in This Family

{% for algo in algorithms %}
- [**{{ algo.name }}**]({{ algo.slug }}.md) - {{ algo.summary }}
{% endfor %}

{% if family.template_options.show_implementation_status %}
## Implementation Status

{% set complete_count = algorithms | selectattr('status.current', 'equalto', 'complete') | list | length %}
{% set in_progress_count = algorithms | selectattr('status.current', 'equalto', 'in_progress') | list | length %}
{% set planned_count = algorithms | selectattr('status.current', 'equalto', 'planned') | list | length %}
{% set total_count = algorithms | length %}

<div class="grid cards" markdown>

-   :material-check-circle: **Complete**

    ---

    {{ complete_count }}/{{ total_count }} algorithms ({{ (complete_count / total_count * 100) | round(0) | int }}%)

{% if in_progress_count > 0 %}
-   :material-clock: **In Progress**

    ---

    {{ in_progress_count }}/{{ total_count }} algorithms ({{ (in_progress_count / total_count * 100) | round(0) | int }}%)
{% endif %}

-   :material-calendar: **Planned**

    ---

    {{ planned_count }}/{{ total_count }} algorithms ({{ (planned_count / total_count * 100) | round(0) | int }}%)

</div>
{% endif %}

{% if family.template_options.show_related_families %}
## Related Algorithm Families

{% for related in family.related_families %}
- **{{ related.id | title }}**: {{ related.description }}
{% endfor %}
{% endif %}

{% if family.template_options.show_references %}
## References

{% if references %}
{% for ref in references %}
1. {% if ref.author %}{{ ref.author }}{% if ref.year %} ({{ ref.year }}){% endif %}{% if ref.title %}. {% if ref.url %}[{{ ref.title }}]({{ ref.url }}){:target="_blank"}{% else %}{{ ref.title }}{% endif %}{% endif %}{% if ref.publisher %}. {{ ref.publisher }}{% endif %}{% if ref.note %}. {{ ref.note }}{% endif %}{% else %}{% if ref.title %}{% if ref.url %}[{{ ref.title }}]({{ ref.url }}){:target="_blank"}{% else %}{{ ref.title }}{% endif %}{% endif %}{% if ref.publisher %}. {{ ref.publisher }}{% endif %}{% if ref.note %}. {{ ref.note }}{% endif %}{% endif %}
{% endfor %}
{% else %}
!!! info "No References"
    No references available for this algorithm family.
{% endif %}
{% endif %}

## Tags

{% if tags %}
{% for tag in tags %}
[:material-tag: {{ tag.name }}]({{ tag.id }}){: .md-tag .md-tag--primary } {{ tag.description }}
{% endfor %}
{% else %}
!!! info "No Tags"
    No tags available for this algorithm family.
{% endif %}
