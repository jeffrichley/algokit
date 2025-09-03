# Algorithm Documentation Templates

This directory contains templates for creating consistent, professional algorithm documentation pages.

## 📋 Available Templates

### `algorithm-template.md`
**Complete template for algorithm pages** with extensive commenting and instructions.

## 🚀 Quick Start

1. **Copy the template:**
   ```bash
   cp docs/templates/algorithm-template.md docs/algorithms/<family>/<algorithm-name>.md
   ```

2. **Replace all placeholders** marked with `[REPLACE: ...]`

3. **Customize content** for your specific algorithm

4. **Test the page** builds correctly

## 🎯 Template Features

- ✅ **YAML frontmatter** for metadata and search
- ✅ **Family navigation** links
- ✅ **Mathematical formulation** with LaTeX support
- ✅ **Multiple implementation** approaches
- ✅ **Complexity analysis** tables
- ✅ **Use case categorization** in grids
- ✅ **Comprehensive references**
- ✅ **Interactive learning** elements

## 🔧 Required Sections

Every algorithm page must include:

1. **Overview** - What the algorithm does
2. **Mathematical Formulation** - Core formulas and properties
3. **Implementation Approaches** - Multiple coding examples
4. **Complexity Analysis** - Time/space complexity comparison
5. **Use Cases & Applications** - Real-world examples
6. **References & Further Reading** - Academic and practical resources
7. **Interactive Learning** - Hands-on implementation guidance

## 📚 Example Usage

See `docs/algorithms/dynamic-programming/fibonacci.md` for a complete, polished example of how this template should look when filled out.

## ⚠️ Important Notes

- **Maintain exact structure** - Don't change section order or formatting
- **Test math rendering** - Ensure LaTeX displays correctly
- **Verify all links** - Check family page links work
- **Follow style guide** - Use consistent admonitions and formatting
- **Add to navigation** - Update family overview pages

## 🎨 Styling

The template uses Material theme components:
- `!!! abstract` for overviews
- `!!! math` for mathematical content
- `!!! example` for tables
- `!!! warning` for important notes
- `!!! tip` for helpful information
- `!!! grid` for organized layouts
- `===` for tabbed content

## 🔍 Quality Checklist

Before publishing any algorithm page:

- [ ] All placeholders replaced
- [ ] Math renders correctly
- [ ] Links work properly
- [ ] Content is accurate and complete
- [ ] References are properly formatted
- [ ] Page builds without errors
- [ ] Added to family navigation
- [ ] Follows established style guide
