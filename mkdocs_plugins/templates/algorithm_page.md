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

!!! tip "Need Help Understanding This Algorithm?"
    {{ chatgpt_widget(algo.name, context='Explain the ' + algo.name + ' algorithm, including its definition, intuition, and why it is important in ' + family.name + '.', button_text='🤖 Ask ChatGPT about ' + algo.name) | safe }}

!!! abstract "Overview"
    {{ algo.description }}

{% if algo.formulation %}
## Mathematical Formulation

{{ chatgpt_widget(algo.name, context='Walk me through the mathematical formulation of ' + algo.name + ', including its recurrence relation, closed form (if any), and growth rate.', button_text='🧮 Ask ChatGPT about Mathematical Formulation') | safe }}

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

{% if algo.formulation.problem_definition %}
!!! abstract "Problem Definition"
    {{ algo.formulation.problem_definition }}
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

{% if algo.formulation.key_properties %}
!!! info "Key Properties"
    {% for prop in algo.formulation.key_properties %}
    **{{ prop.name }}**

    `{{ prop.formula }}`

    {{ prop.description }}

    ---
    {% endfor %}
{% endif %}
{% endif %}

{% if algo.properties %}
## Key Properties

{{ chatgpt_widget(algo.name, context='What are the key properties of ' + algo.name + ', such as optimal substructure, overlapping subproblems, and any unique mathematical insights?', button_text='🔑 Ask ChatGPT about Key Properties') | safe }}

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

{{ chatgpt_widget(algo.name, context='Compare different implementation approaches for ' + algo.name + ' (e.g. naive recursion, memoization, iterative dynamic programming). Discuss pros, cons, and use cases.', button_text='💻 Ask ChatGPT about Implementation') | safe }}

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
    - **{{ source_file.description }}**: [`{{ source_file.path }}`](https://github.com/jeffrichley/algokit/blob/main/{{ source_file.path }}){:target="_blank" rel="noopener noreferrer"}
    {% endfor %}
{% endif %}

{% if algo.complexity and algo.complexity.analysis %}
## Complexity Analysis

{{ chatgpt_widget(algo.name, context='Explain the time and space complexity of ' + algo.name + ' across its common implementations, and describe when each approach is most efficient.', button_text='📊 Ask ChatGPT about Complexity') | safe }}

!!! example "**Time & Space Complexity Comparison**"
    | Approach | Time Complexity | Space Complexity | Notes |
    |----------|-----------------|------------------|-------|
    {% for analysis in algo.complexity.analysis %}| **{{ analysis.approach }}** | {{ analysis.time }} | {{ analysis.space }} | {{ analysis.notes }} |{% endfor %}

{% if algo.complexity.performance_notes %}
!!! warning "Performance Considerations"
    {% for note in algo.complexity.performance_notes %}
    - {{ note }}
    {% endfor %}
{% endif %}
{% endif %}

{% if algo.applications %}
## Use Cases & Applications

{{ chatgpt_widget(algo.name, context='Give real-world applications of ' + algo.name + ' in fields like computer science, finance, biology, and design. Why does it matter outside of theory?', button_text='🌍 Ask ChatGPT about Applications') | safe }}

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

<div class="references-container" style="margin: 24px 0;">

{% for ref_category in algo.references %}

<h3><span class="twemoji">{% if "textbook" in ref_category.category.lower() or "book" in ref_category.category.lower() %}{% include "icons/book.svg" %}{% elif "historical" in ref_category.category.lower() or "cultural" in ref_category.category.lower() %}{% include "icons/history.svg" %}{% elif "online" in ref_category.category.lower() or "resource" in ref_category.category.lower() %}{% include "icons/web.svg" %}{% elif "implementation" in ref_category.category.lower() or "practice" in ref_category.category.lower() %}{% include "icons/code-tags.svg" %}{% else %}{% include "icons/library.svg" %}{% endif %}</span> {{ ref_category.category }}</h3>

<div class="reference-category" style="margin-bottom: 32px; padding: 20px; border-radius: 12px; background: linear-gradient(135deg, var(--md-default-bg-color--lightest) 0%, var(--md-default-bg-color--light) 100%); border-left: 4px solid var(--md-primary-fg-color); box-shadow: 0 2px 8px rgba(0,0,0,0.1);">

<div class="reference-items" style="margin-top: 16px;">

{% for item in ref_category['items'] %}
<div class="reference-item" style="display: flex; align-items: flex-start; margin-bottom: 16px; padding: 12px; border-radius: 8px; background: var(--md-default-bg-color); border: 1px solid var(--md-default-fg-color--lightest); transition: all 0.2s ease;">

<div class="reference-icon" style="margin-right: 12px; margin-top: 2px; flex-shrink: 0;">
<span class="twemoji">{% if item.publisher %}{% include "icons/book.svg" %}{% elif item.url %}{% include "icons/link.svg" %}{% else %}{% include "icons/file-document.svg" %}{% endif %}</span>
</div>

<div class="reference-content" style="flex: 1;">
{% if item.url %}
<div class="reference-title" style="margin-bottom: 4px;">
<a href="{{ item.url }}" target="_blank" rel="noopener noreferrer" style="color: var(--md-primary-fg-color); text-decoration: none; font-weight: 600; font-size: 1.05em;">{{ item.title or item.author }}</a>
</div>
{% else %}
<div class="reference-title" style="margin-bottom: 4px; color: var(--md-default-fg-color); font-weight: 600; font-size: 1.05em;">
<strong>{{ item.title or item.author }}</strong>
</div>
{% endif %}

<div class="reference-meta" style="color: var(--md-default-fg-color--light); font-size: 0.9em; line-height: 1.4;">
{% if item.year %}<span style="color: var(--md-accent-fg-color); font-weight: 500;">{{ item.year }}</span>{% endif %}{% if item.publisher %}{% if item.year %} • {% endif %}<span style="font-style: italic;">{{ item.publisher }}</span>{% endif %}{% if item.isbn %}{% if item.year or item.publisher %} • {% endif %}<span>ISBN {{ item.isbn }}</span>{% endif %}{% if item.note %}{% if item.year or item.publisher or item.isbn %} • {% endif %}<span>{{ item.note }}</span>{% endif %}
</div>

{% if item.note and "ISBN" in item.note %}
{% set isbn = item.note.split("ISBN ")[1] if "ISBN " in item.note else "" %}
{% if isbn %}
{% set clean_isbn = isbn.replace('-', '').replace(' ', '') %}
<div class="reference-links" style="margin-top: 8px;">
<a href="https://www.amazon.com/dp/{{ clean_isbn }}/?tag=mathybits-20" target="_blank" rel="noopener noreferrer" style="display: inline-flex; align-items: center; padding: 4px 8px; background: linear-gradient(135deg, #ff9500 0%, #ff6b00 100%); color: white; text-decoration: none; border-radius: 4px; font-size: 0.8em; font-weight: 500; transition: all 0.2s ease;">
<span class="twemoji">{% include "icons/amazon.svg" %}</span>
<span style="margin-left: 4px;">Buy on Amazon</span>
</a>
</div>
{% endif %}
{% endif %}

</div>

</div>
{% endfor %}

</div>

</div>
{% endfor %}

</div>

!!! success "Interactive Learning"
    Try implementing the different approaches yourself! This progression will give you deep insight into the algorithm's principles and applications.

    <span class="twemoji">{% include "icons/lightbulb.svg" %}</span> **Pro Tip**: Start with the simplest implementation and gradually work your way up to more complex variants.

!!! success "Need More Help? Ask ChatGPT!"
    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;">
        {{ chatgpt_widget(algo.name, context='Explain ' + algo.name + ' like I am 5 years old, using a fun and simple analogy.', button_text='🧒 Explain Simply', style='secondary') | safe }}
        {{ chatgpt_widget(algo.name, context='Give me practice problems and coding exercises to help me master ' + algo.name + '.', button_text='📝 Practice Problems', style='secondary') | safe }}
        {{ chatgpt_widget(algo.name, context='Compare ' + algo.name + ' to other algorithms in the ' + family.name + ' family. Highlight similarities, differences, and when to use each.', button_text='🔀 Compare Algorithms', style='secondary') | safe }}
        {{ chatgpt_widget(algo.name, context='Help me debug a ' + algo.name + ' implementation. What are the most common mistakes students make, and how can I fix them?', button_text='🐛 Debug Help', style='secondary') | safe }}
    </div>
{% endif %}

## Navigation

**Related Algorithms in {{ family.name }}:**

{% for related_algo in algorithms if related_algo.slug != algo.slug %}
- [{{ related_algo.name }}]({{ related_algo.slug }}.md) - {{ related_algo.summary }}

{% endfor %}
