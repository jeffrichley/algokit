# Algorithm Kit - Cursor Rules

This directory contains the Cursor rules that define development standards, workflows, and quality requirements for the Algorithm Kit project.

## 📋 Rules Overview

### 🔄 **Core Workflow Rules**
- **`agent-os.mdc`** - Agent OS documentation and workflow instructions
- **`archon.mdc`** - Archon integration and task-driven development workflow ⭐ **NEW**
- **`stuffit.mdc`** - Git + quality workflow with just/nox/uv commands

### 🧪 **Testing & Quality Rules**
- **`testing-rules.mdc`** - Comprehensive testing standards and best practices ⭐ **ENHANCED**
- **`model-cleanup.mdc`** - Pydantic validator cleanup rules

### 📊 **Product Management Rules** ⭐ **NEW**
- **`create-spec.mdc`** - Specification creation workflow
- **`execute-tasks.mdc`** - Task execution workflow
- **`plan-product.mdc`** - Product planning workflow
- **`analyze-product.mdc`** - Product analysis workflow

### 🎯 **AI Development Standards** ⭐ **NEW**
- **`ai-coding-standards.mdc`** - Comprehensive Python coding standards (766 lines)

## 🚀 **Recent Updates from Redwing Core**

The following rules were imported from the redwing_core project to bring Algorithm Kit up to the same development standard:

### **New Rules Added:**
1. **`archon.mdc`** - Critical workflow management with Archon integration
2. **`ai-coding-standards.mdc`** - Comprehensive coding standards and type safety
3. **`create-spec.mdc`** - Specification creation workflow
4. **`execute-tasks.mdc`** - Task execution workflow
5. **`plan-product.mdc`** - Product planning workflow
6. **`analyze-product.mdc`** - Product analysis workflow

### **Enhanced Rules:**
1. **`testing-rules.mdc`** - Added exception message testing guidelines and advanced fixture best practices

## 🔧 **Rule Application**

- **`alwaysApply: true`** - Rules that are automatically applied to all interactions
- **`alwaysApply: false`** - Rules that are applied when specifically requested

## 📚 **Usage**

These rules are automatically applied by Cursor when working in this project. They ensure:

- Consistent code quality and style
- Proper testing practices
- Workflow compliance
- Type safety enforcement
- Documentation standards

## 🔗 **Related Documentation**

- **Project Documentation**: `docs/`
- **Contributing Guide**: `CONTRIBUTING.md`
- **Quality Commands**: `justfile` and `noxfile.py`
