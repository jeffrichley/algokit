Perfect — let’s turn this into a **100x more specific Markdown plan** that your AI coder can execute without guesswork. I’ll cover folder layout, page templates, navigation config, and even conventions for cross-linking and metadata.

Here’s a draft:

---

# 📚 Documentation System Plan

This document describes the **canonical structure, conventions, and layout** for the project documentation site.
The goal is to create a **clear, consistent, and extensible** system for presenting algorithm families and their implementations.

---

## 🏗️ Taxonomy Rulebook

### Core Principles
1. **One canonical home per algorithm.** Each algorithm belongs to exactly one `docs/algorithms/<family>/` folder.
2. **MPC is a family.** MPC itself is not an algorithm — it is a framework that includes variants such as Linear MPC, Nonlinear MPC, Robust MPC, and Economic MPC.
3. **Real-Time Control is a family.** Contains algorithms like PID, Adaptive Control, Sliding Mode Control, etc.
4. **Overlaps use cross-links.** Families that intersect (e.g., MPC within Real-Time Control) are connected via cross-references, not duplication.
5. **Family overviews own definitions and curated tables.** They do not contain algorithm implementations — only links to them.
6. **Cross-links are mandatory** wherever families intersect.

---

## 1. Directory Layout

```bash
docs/
├── index.md                # Landing page
├── contributing.md         # Contribution guide
├── SECURITY.md             # Security policy
├── api.md                  # API reference entry point
├── families/               # Algorithm family overviews
│   ├── dynamic-programming.md
│   ├── reinforcement-learning.md
│   ├── hierarchical-rl.md
│   ├── planning.md
│   ├── control.md
│   ├── real-time-control.md        # Family
│   ├── mpc.md                      # Family
│   ├── gaussian-process.md
│   └── dmps.md
└── algorithms/             # Individual algorithms grouped by family
    ├── dynamic-programming/
    │   ├── coin-change.md
    │   ├── fibonacci.md
    │   └── longest-common-subsequence.md
    ├── control/
    │   ├── pid-control.md
    │   ├── adaptive-control.md
    │   ├── sliding-mode-control.md
    │   └── h-infinity-control.md
    ├── mpc/
    │   ├── linear-mpc.md
    │   ├── nonlinear-mpc.md
    │   ├── robust-mpc.md
    │   └── economic-mpc.md
    ├── planning/
    │   ├── a-star-search.md
    │   └── graphplan.md
    └── gaussian-process/
        ├── gp-classification.md
        └── multi-output-gps.md
```

---

## 2. Page Templates

### 2.1 Family Page (`docs/families/<family>.md`)

Each family doc should follow a **consistent template**:

```markdown
# <Family Name>

## Overview
Definition, motivation, history, applications.

## Key Concepts
- Concept A
- Concept B

## Comparison Table
| Algorithm | Complexity | Strengths | Weaknesses | Use Cases |
|-----------|------------|-----------|------------|-----------|
| Example   | O(n^2)     | ...       | ...        | ...       |

## Algorithms in This Family
- [Algorithm A](../algorithms/<family>/<algo-a>.md)
- [Algorithm B](../algorithms/<family>/<algo-b>.md)

## Relationship to Other Families
- Cross-links to overlapping families.
```

---

### 2.2 Algorithm Page (`docs/algorithms/<family>/<algo>.md`)

Each algorithm doc should follow a **canonical structure**:

```markdown
---
tags: [<family>, <context tags>]
---

# <Algorithm Name>

**Family:** [<Family Name>](../../../families/<family>.md)

## Overview
Intuition and use.

## Mathematical Formulation
Equations.

## Pseudocode
```pseudo
# clear, concise
```

## Python Implementation

```python
# reference implementation
```

## Complexity

* Time: O(...)
* Space: O(...)

## Use Cases

* Example 1
* Example 2

## References

* Papers / textbooks
```

---

## 3. Navigation Config (MkDocs Example)

```yaml
nav:
  - Home: index.md
  - Families:
      - Real-Time Control: families/real-time-control.md
      - MPC: families/mpc.md
      - Control: families/control.md
      - Dynamic Programming: families/dynamic-programming.md
      - Reinforcement Learning: families/reinforcement-learning.md
      - Hierarchical RL: families/hierarchical-rl.md
      - Planning: families/planning.md
      - Gaussian Processes: families/gaussian-process.md
      - Dynamic Movement Primitives: families/dmps.md
  - Algorithms:
      - Control:
          - PID Control: algorithms/control/pid-control.md
          - Adaptive Control: algorithms/control/adaptive-control.md
          - Sliding Mode Control: algorithms/control/sliding-mode-control.md
          - H∞ Control: algorithms/control/h-infinity-control.md
      - MPC:
          - Linear MPC: algorithms/mpc/linear-mpc.md
          - Nonlinear MPC: algorithms/mpc/nonlinear-mpc.md
          - Robust MPC: algorithms/mpc/robust-mpc.md
          - Economic MPC: algorithms/mpc/economic-mpc.md
      - Dynamic Programming:
          - Coin Change: algorithms/dynamic-programming/coin-change.md
          - Fibonacci: algorithms/dynamic-programming/fibonacci.md
          - Longest Common Subsequence: algorithms/dynamic-programming/longest-common-subsequence.md
      - RL:
          - Option-Critic: algorithms/reinforcement-learning/option-critic.md
          - Feudal Networks: algorithms/hierarchical-rl/feudal-networks.md
          - Hierarchical Actor-Critic: algorithms/hierarchical-rl/hierarchical-actor-critic.md
      - Planning:
          - A*: algorithms/planning/a-star-search.md
          - GraphPlan: algorithms/planning/graphplan.md
      - Gaussian Processes:
          - GP Classification: algorithms/gaussian-process/gp-classification.md
          - Multi-Output GPs: algorithms/gaussian-process/multi-output-gps.md
```

---

## 4. Cross-Linking Rules

* Every **algorithm page** links back to its **family page**.
* Every **family page** links forward to **all algorithm pages** in its group.
* **Real-Time Control** → links to PID, Adaptive, Sliding Mode, **and MPC family overview**.
* **MPC** → links back to Real-Time Control.
* Related algorithms (e.g., A\* and GraphPlan) should cross-link.
* Use relative links (`../`) so docs work offline and on GitHub.

---

## 5. Enhancements

* **Tagging**: add YAML frontmatter to each doc:

  ```yaml
  ---
  tags: [dp, optimization, search]
  ---
  ```
* **Visual Index**: create a `docs/index.md` that includes a diagram showing families → algorithms.
* **Glossary**: add `docs/glossary.md` with key terms.
* **Comparisons**: add per-family tables (strengths/weaknesses).

---

⚡ **Next Action for AI Coder**:

1. Create fresh algorithm files in the organized family subdirectories.
2. Apply canonical template to every algorithm page.
3. Update navigation config (`mkdocs.yml`).
4. Add cross-links and metadata.

---

## 📋 **Current Status & Cleanup Summary**

### **What Has Been Completed:**
✅ **Phase 1.1**: Directory structure created
- `docs/families/` directory with 9 family overview pages
- `docs/algorithms/` directory with 9 empty family subdirectories

✅ **Phase 2**: Family overview pages created
- All 9 family pages follow consistent template structure
- Proper cross-links between related families (e.g., MPC ↔ Real-Time Control)

✅ **Cleanup Work Completed:**
- Removed 8 conflicting files from root `docs/` directory
- Eliminated duplicate content between old and new family pages
- Achieved clean, organized structure ready for algorithm files

### **Current Clean State:**
```
docs/
├── families/           # ✅ 9 family overview pages (complete)
├── algorithms/         # ✅ 9 empty family subdirectories (ready)
├── styles/            # ✅ Custom CSS preserved
├── index.md           # ✅ Main landing page
├── contributing.md    # ✅ Contribution guide
├── SECURITY.md        # ✅ Security policy
└── api.md             # ✅ API reference
```

### **What Was Removed During Cleanup:**
- `classical-planning.md` ❌ (conflicted with `families/planning.md`)
- `dmps.md` ❌ (conflicted with `families/dmps.md`)
- `gaussian-process.md` ❌ (conflicted with `families/gaussian-process.md`)
- `hrl.md` ❌ (conflicted with `families/hierarchical-rl.md`)
- `mpc.md` ❌ (conflicted with `families/mpc.md`)
- `real-time-control.md` ❌ (conflicted with `families/real-time-control.md`)
- `reinforcement-learning.md` ❌ (conflicted with `families/reinforcement-learning.md`)
- `classic-dp.md` ❌ (conflicted with `families/dynamic-programming.md`)

### **Why Cleanup Was Necessary:**
- **Duplicate content**: Old files conflicted with new family pages
- **Wrong locations**: Family content was in root instead of `families/` subdirectory
- **Broken links**: Family pages linked to non-existent algorithm files
- **Confusion**: Multiple sources of truth for same information

### **Important Note About Algorithm Files:**
- **No existing algorithm files were moved** - they were created fresh during our work
- **Template files were discovered and removed** to ensure clean foundation
- **Current approach**: Create algorithm files from scratch with proper canonical template
- **This ensures quality** and consistency across all algorithm documentation

### **Next Steps:**
1. **Complete Phase 1.2**: Create fresh algorithm files in family subdirectories
2. **Execute Phase 3**: Apply canonical template to all algorithm pages
3. **Continue with remaining phases**: Cross-linking, navigation, testing

---

## 🚀 Implementation Checklist

### Phase 1: Directory Restructuring
- [x] **Create new directory structure**
  - [x] Create `docs/families/` directory
  - [x] Create `docs/algorithms/` directory with subdirectories for each family
  - [ ] Create fresh algorithm files in appropriate family subdirectories
  - [ ] Verify new file structure is properly organized

- [ ] **Create and organize algorithm files by family**
  - [ ] Dynamic Programming: `coin-change.md`, `fibonacci.md`, `longest-common-subsequence.md`, `matrix-chain-multiplication.md`, `edit-distance.md`, `knapsack.md`
  - [ ] Reinforcement Learning: `q-learning.md`, `dqn.md`, `actor-critic.md`, `policy-gradient.md`, `ppo.md`
  - [ ] Hierarchical RL: `hierarchical-actor-critic.md`, `hierarchical-policy-gradient.md`, `hierarchical-q-learning.md`, `hierarchical-task-networks.md`, `feudal-networks.md`, `option-critic.md`
  - [ ] Control: `pid-control.md`, `adaptive-control.md`, `sliding-mode-control.md`, `h-infinity-control.md`, `robust-control.md`
  - [ ] Model Predictive Control: `model-predictive-control.md`, `linear-mpc.md`, `nonlinear-mpc.md`, `robust-mpc.md`, `economic-mpc.md`, `distributed-mpc.md`, `learning-mpc.md`
  - [ ] Planning: `a-star-search.md`, `graphplan.md`, `partial-order-planning.md`, `fast-forward.md`
  - [ ] Gaussian Processes: `gp-classification.md`, `gp-optimization.md`, `gp-regression.md`, `deep-gps.md`, `sparse-gps.md`, `multi-output-gps.md`
  - [ ] Dynamic Movement Primitives: `basic-dmps.md`, `adaptive-dmps.md`, `multi-dimensional-dmps.md`, `probabilistic-dmps.md`, `dmps-obstacle-avoidance.md`

### Phase 2: Family Page Creation
- [x] **Create family overview pages**
  - [x] `docs/families/dynamic-programming.md` - Overview, key concepts, comparison table, algorithm links
  - [x] `docs/families/reinforcement-learning.md` - Overview, key concepts, comparison table, algorithm links
  - [x] `docs/families/hierarchical-rl.md` - Overview, key concepts, comparison table, algorithm links
  - [x] `docs/families/control.md` - Overview, key concepts, comparison table, algorithm links
  - [x] `docs/families/mpc.md` - Overview, key concepts, comparison table, algorithm links, cross-link to Real-Time Control
  - [x] `docs/families/planning.md` - Overview, key concepts, comparison table, algorithm links
  - [x] `docs/families/gaussian-process.md` - Overview, key concepts, comparison table, algorithm links
  - [x] `docs/families/dmps.md` - Overview, key concepts, comparison table, algorithm links
  - [x] `docs/families/real-time-control.md` - Overview, key concepts, comparison table, algorithm links, cross-link to MPC family

### Phase 3: Algorithm Page Standardization
- [ ] **Apply canonical template to all algorithm pages**
  - [ ] Ensure each algorithm page has: Overview, Mathematical Formulation, Pseudocode, Python Implementation, Complexity, Use Cases, References
  - [ ] Add family link at top of each algorithm page
  - [ ] Add YAML frontmatter with tags
  - [ ] Standardize formatting and structure across all algorithm docs
  - [ ] Fill in missing content with appropriate stubs where needed

### Phase 4: Cross-Linking Implementation
- [ ] **Add bidirectional navigation links**
  - [ ] Link each algorithm page back to its family page
  - [ ] Link each family page to all its algorithm pages
  - [ ] **Add cross-links between Real-Time Control and MPC families**
  - [ ] Add cross-references between related algorithms
  - [ ] Verify all relative links work correctly (`../` navigation)

### Phase 5: Navigation Configuration
- [ ] **Update mkdocs.yml navigation**
  - [ ] Restructure navigation to match new directory layout
  - [ ] Add family overview pages to navigation
  - [ ] Organize algorithms by family in navigation
  - [ ] Ensure MPC and Real-Time Control are distinct families
  - [ ] Test navigation structure locally

### Phase 6: Metadata and Enhancement
- [ ] **Add YAML frontmatter**
  - [ ] Add tags to each document for categorization
  - [ ] Include creation/update dates where appropriate
  - [ ] Add author information for documentation tracking

- [ ] **Create enhanced index page**
  - [ ] Update `docs/index.md` with family → algorithm diagram
  - [ ] Add visual navigation elements
  - [ ] Include quick reference tables

- [ ] **Add glossary and comparisons**
  - [ ] Create `docs/glossary.md` with key terms
  - [ ] Add per-family comparison tables (strengths/weaknesses)
  - [ ] Include complexity and application matrices

### Phase 7: Quality Assurance
- [ ] **Test documentation site**
  - [ ] Build site locally with `mkdocs serve`
  - [ ] Verify all links work correctly
  - [ ] Check navigation structure is intuitive
  - [ ] Validate markdown formatting

- [ ] **Content review**
  - [ ] Ensure consistent terminology across all docs
  - [ ] Verify mathematical notation is consistent
  - [ ] Check that all algorithm implementations are properly documented
  - [ ] Validate that family groupings make logical sense
  - [ ] **Verify MPC and Real-Time Control cross-links work correctly**

### Phase 8: Finalization
- [ ] **Update project documentation**
  - [ ] Update `README.md` to reflect new documentation structure
  - [ ] Update `CONTRIBUTING.md` with new documentation standards
  - [ ] Create documentation contribution templates
  - [ ] Document the new structure for future contributors

- [ ] **Commit and deploy**
  - [ ] Commit all changes with descriptive commit messages
  - [ ] Test deployment to ensure site builds correctly
  - [ ] Update any CI/CD pipelines if necessary

---

## ✅ Acceptance Criteria

* ✅ `families/mpc.md` and `families/real-time-control.md` both exist.
* ✅ MPC algorithms only appear under `algorithms/mpc/`.
* ✅ Real-Time Control algorithms appear under `algorithms/control/`.
* ✅ Cross-links connect Real-Time Control ↔ MPC.
* ✅ Navigation shows both families distinctly.
* ✅ No algorithm file is duplicated across families.

---

**Estimated Time**: 4-6 hours for complete restructuring
**Priority**: High - Foundation for all future documentation
**Dependencies**: None - can be done independently
**Risk**: Low - primarily file organization and content standardization
