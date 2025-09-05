# Macros Directory

This directory contains the Python modules that power the MkDocs Macros system for the AlgoKit documentation.

## 📁 File Structure

```
macros/
├── __init__.py          # Main macro definitions and registration
├── data_loader.py       # YAML data loading and validation
├── navigation.py        # Navigation and content generation functions
├── utils.py            # Utility functions for formatting and processing
└── README.md           # This file
```

## 🔧 Module Overview

### `__init__.py`
- **Purpose**: Main entry point for MkDocs Macros
- **Key Function**: `define_env(env)` - Registers all macro functions
- **Macros Defined**:
  - `algorithm_card(key)` - Generate algorithm information cards
  - `nav_grid(current_algorithm, current_family, max_related)` - Create navigation grids
  - `family_overview(family)` - Generate family overview content
  - `progress_summary()` - Show project progress
  - `complexity_badge(complexity)` - Format complexity notation
  - `format_complexity_macro(complexity)` - Format complexity for display

### `data_loader.py`
- **Purpose**: Load and validate algorithm data from `algorithms.yaml`
- **Key Class**: `AlgorithmsDataLoader`
- **Features**:
  - YAML parsing and validation
  - Data caching for performance
  - Error handling and fallbacks
  - Algorithm and family lookup methods

### `navigation.py`
- **Purpose**: Generate dynamic navigation and content
- **Key Functions**:
  - `generate_algorithm_card(algorithm_key)` - Create algorithm cards
  - `generate_navigation_grid(current_algorithm, current_family, max_related)` - Build navigation grids
  - `generate_family_overview(family_key)` - Create family overviews
  - `generate_progress_summary()` - Generate progress reports
  - `generate_learning_paths()` - Create learning path suggestions

### `utils.py`
- **Purpose**: Utility functions for formatting and processing
- **Key Functions**:
  - `format_complexity(complexity)` - Format complexity notation
  - `get_family_display_name(family_key)` - Get human-readable family names
  - `get_algorithm_display_name(algorithm_key)` - Get human-readable algorithm names

## 🚀 How It Works

1. **MkDocs loads** the `macros` module via `module_name: macros` in `mkdocs.yml`
2. **`define_env(env)`** registers all macro functions with the MkDocs environment
3. **Markdown pages** call macros using `{{ macro_name(parameters) }}` syntax
4. **Macros execute** Python code to generate dynamic content
5. **Content is rendered** as HTML in the final documentation

## 🔄 Data Flow

```
algorithms.yaml → data_loader.py → navigation.py → markdown → HTML
```

1. **`algorithms.yaml`** contains all algorithm metadata
2. **`data_loader.py`** loads and validates the data
3. **`navigation.py`** uses the data to generate content
4. **Markdown pages** call macros to get dynamic content
5. **MkDocs** renders everything to HTML

## 🛠️ Development

### Adding New Macros

1. **Define the function** in the appropriate module
2. **Register it** in `__init__.py` using `@env.macro`
3. **Test it** in a markdown page
4. **Document it** in the macro system guide

### Example New Macro

```python
# In navigation.py
def generate_custom_content(data):
    """Generate custom content."""
    return f"Custom content: {data}"

# In __init__.py
@env.macro
def custom_content(data):
    """Wrapper for custom content generation."""
    return generate_custom_content(data)
```

### Testing Macros

Use the test page `docs/test-macros-demo.md` to test new macros:

```markdown
{{ custom_content("test data") }}
```

## 📊 Status System

The system uses emoji status indicators:

- ✅ **Complete** - `status: "complete"`
- ⏳ **Planned** - `status: "planned"`
- 🔄 **In-Progress** - `status: "in-progress"`
- ❌ **Deprecated** - `status: "deprecated"`

## 🔍 Debugging

### Common Issues

1. **Import errors**: Check that all modules are properly imported
2. **Data not found**: Verify algorithm keys exist in `algorithms.yaml`
3. **Macro not working**: Check that it's registered in `__init__.py`
4. **Build errors**: Check MkDocs build output for specific errors

### Debug Mode

Enable debug output by checking the MkDocs build logs:

```bash
uv run mkdocs build --verbose
```

## 📚 Dependencies

- **PyYAML**: For parsing `algorithms.yaml`
- **MkDocs Macros Plugin**: For macro functionality
- **Material for MkDocs**: For theme and features

## 🎯 Best Practices

1. **Keep macros simple** - Complex logic should be in separate functions
2. **Use descriptive names** - Make macro purposes clear
3. **Handle errors gracefully** - Provide fallbacks for missing data
4. **Document everything** - Update guides when adding new features
5. **Test thoroughly** - Verify macros work in different contexts

## 🔮 Future Enhancements

Potential improvements:

- **Caching**: Add intelligent caching for better performance
- **Validation**: Enhanced data validation and error reporting
- **Templates**: More flexible template system
- **Integration**: Better integration with CI/CD systems
- **Analytics**: Usage tracking and optimization

---

*This README is part of the AlgoKit macro system. For questions or contributions, please refer to the main project documentation.*
