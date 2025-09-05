Iris here. I put together a **step-by-step, battle-ready plan** your local gen-AI can follow to implement the full “generate pages from data” system with **mkdocs-gen-files + mkdocs-macros**, families/algorithms YAML, shared tags/refs, validation, nav, and polish. It’s phased, exhaustive, and loaded with checkboxes so you can track progress.

---

# Phase 0 — Pre-Flight (Repo Hygiene & Decisions)

* [ ] **Decide final data layout** (family-scoped):

  ```
  mkdocs_plugins/
    data/
      shared/
        tags.yaml
        refs.bib
      families/
        dp/
          family.yaml
          algorithms/
            fibonacci.yaml
            value-iteration.yaml
        rl/
          family.yaml
          algorithms/
            q-learning.yaml
            ppo.yaml
  docs/
    index.md
    templates/
      algo.md.j2
      family.md.j2
  mkdocs.yml
  ```
* [ ] **Confirm plugin set**: `gen-files`, `macros`, `search` (optional: `literate-nav`, `i18n`, `minify`, `redirects`)
* [ ] **Confirm theme** (Material recommended) and extensions (`pymdownx.*`, `admonition`, `details`, `superfences`, `arithmatex` if you want LaTeX)
* [ ] **Establish Python tooling**: `uv` for execution, `ruff` + `mypy` + `pytest` ready
* [ ] **Pick templating approach**: render via macros + `env.render("templates/*.j2")` (recommended)

**Definition of Done (DoD):** repo compiles, `uv run mkdocs --version` works; `mkdocs.yml` present; theme chosen.

---

# Phase 1 — MkDocs Config & Plugin Order

* [ ] Add to `mkdocs.yml` (order matters):

  ```yaml
  plugins:
    - gen-files:
        scripts:
          - mkdocs_plugins/gen_pages.py
    - macros:
        modules:
          - mkdocs_plugins.macros.main
    - search
  theme:
    name: material
  markdown_extensions:
    - admonition
    - toc:
        permalink: true
    - pymdownx.superfences
    - pymdownx.details
    - pymdownx.arithmatex
  ```
* [ ] Verify plugin load order (gen-files before macros).
* [ ] Add `site_url`, `repo_url`, and `edit_uri` (enables “Edit on GitHub”).

**DoD:** Running `uv run mkdocs serve` serves a minimal site without errors.

---

# Phase 2 — Data Schema (Families, Algorithms, Tags, Refs)

## 2.1 Family YAML

* [ ] Create `data/families/dp/family.yaml` sample:

  ```yaml
  id: dp
  name: Dynamic Programming
  summary: >
    DP solves problems by reusing overlapping subproblems with optimal substructure.
  algorithms:
    order_mode: by_algo_order   # by_algo_order | by_name | by_slug | pinned_then_alpha
    include: []                  # optional explicit allowlist
    exclude: []                  # optional hide list
    pinned_order: []             # optional: these slugs first, in this order
  key_characteristics:
    - Optimal Substructure
    - Overlapping Subproblems
    - Memoization / Tabulation
  common_applications:
    - Coin change, knapsack, LCS, edit distance
  concepts:
    - Base cases
    - Recurrence relation
    - State transition
  status: stable
  tags: [dp, planning]
  see_also: [rl, divide-and-conquer]
  ```
* [ ] Validate: `id` matches folder; sensible defaults when arrays are omitted.

## 2.2 Algorithm YAML

* [ ] Create `data/families/dp/algorithms/fibonacci.yaml`:

  ```yaml
  slug: fibonacci
  name: Fibonacci Sequence
  family_id: dp
  order: 10
  aliases: [fib]
  summary: >
    Classic sequence; excellent for demonstrating recursion, memoization, and iterative DP.
  tags: [dp, sequence, recursion]
  formulation: |
    F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2)
  properties:
    - Golden ratio asymptotics
    - Binet’s closed form
  implementations:
    - type: iterative
      code: |
        def fibonacci(n: int) -> int:
            if n <= 1: return n
            a, b = 0, 1
            for _ in range(2, n+1):
                a, b = b, a + b
            return b
    - type: memoized
      code: |
        def fibonacci_memoized(n: int, memo=None):
            if memo is None: memo = {}
            if n in memo: return memo[n]
            if n <= 1: return n
            memo[n] = fibonacci_memoized(n-1, memo) + fibonacci_memoized(n-2, memo)
            return memo[n]
  complexity:
    naive: {time: "O(2^n)", space: "O(n)"}
    iterative: {time: "O(n)", space: "O(1)"}
    memoized: {time: "O(n)", space: "O(n)"}
  applications:
    - Computer Science: recursion & DP pedagogy
    - Finance: Fibonacci retracements
    - Biology: spiral growth patterns
  refs: [knuth1997, cormen2009]
  status: stable
  ```
* [ ] Ensure **every tag ID** exists in `shared/tags.yaml` (or plan a fallback).

## 2.3 Shared: Tags & Refs

* [ ] `data/shared/tags.yaml`:

  ```yaml
  tags:
    - id: dp
      name: Dynamic Programming
      description: Store and reuse solutions to overlapping subproblems.
    - id: planning
      name: Planning
      description: Algorithms that reason about future actions/states.
    - id: recursion
      name: Recursion
      description: Self-referential definitions used in algorithm design.
    - id: sequence
      name: Sequences
      description: Integer sequences and their properties.
  ```
* [ ] `data/shared/refs.bib` (or `refs.yaml` if you prefer YAML):

  ```bibtex
  @book{knuth1997,
    author={Donald E. Knuth},
    title={The Art of Computer Programming, Volume 1},
    year={1997},
    publisher={Addison-Wesley}
  }
  @book{cormen2009,
    author={Thomas H. Cormen and others},
    title={Introduction to Algorithms},
    year={2009},
    publisher={MIT Press}
  }
  ```

**DoD:** Sample family/algorithm files load and pass a JSON/YAML parse check.

---

# Phase 3 — Page Generation Script (`gen_pages.py`)

* [ ] Implement disk discovery:

  * Find families at `data/families/*/family.yaml`
  * For each family, find `algorithms/*.yaml`
* [ ] Apply family filters:

  * `include` allowlist, `exclude` blocklist
  * `order_mode`:

    * `by_algo_order`: sort by `algo["order"]` default 9999
    * `by_name`: `a["name"].lower()`
    * `by_slug`: `a["slug"]`
    * `pinned_then_alpha`: list from `pinned_order` then alphabetical remainder
* [ ] Generate **virtual pages**:

  * `algorithms/<family_id>/index.md` → `{{ family_page('<fid>') }}`
  * `algorithms/<family_id>/<slug>.md` → `{{ algo_page(family_id='<fid>', slug='<slug>') }}`
* [ ] Set edit paths:

  * `mkdocs_gen_files.set_edit_path(family_index, path_to_family_yaml)`
  * `mkdocs_gen_files.set_edit_path(algo_page, path_to_algo_yaml)`
* [ ] Optionally write a `SUMMARY.md` that links all families/algorithms for nav.

**DoD:** After `mkdocs serve`, generated pages appear and link correctly. “Edit on GitHub” points to YAML.

---

# Phase 4 — Macros & Rendering (`macros/main.py`)

* [ ] Load shared vocab:

  * `tags_index = {id: {name, description}}` from `shared/tags.yaml`
  * Provide a helper `resolve_tags(list[str]) -> list[str]` (names)
* [ ] Family macro:

  * `family_page(fid)` loads family metadata
  * Discovers algorithms (again, to avoid drift)
  * Sorts list per `order_mode`
  * Renders via `env.render("templates/family.md.j2", **ctx)`
* [ ] Algorithm macro:

  * `algo_page(family_id, slug)` loads algorithm YAML + family
  * Resolves tags to human-readable names
  * Renders via `env.render("templates/algo.md.j2", **ctx)`
* [ ] (Optional) Filters:

  * `@env.filter("badge")` for status chips (`draft`, `stable`, `deprecated`)
  * `@env.filter("codeblock")` to safely wrap code with language fences

**DoD:** Macros return fully rendered Markdown for both page types without exceptions.

---

# Phase 5 — Templates (Jinja in `docs/templates/`)

## 5.1 `family.md.j2`

* [ ] Title, summary, characteristics, concepts, applications
* [ ] Auto-list algorithms (name → link)
* [ ] Optional status/tags badges
* [ ] Footer “Generated from YAML”

## 5.2 `algo.md.j2`

* [ ] H1 name + badges (family, status)
* [ ] Tag badges (resolved names)
* [ ] Summary paragraph
* [ ] Math/formulation section (render with `pymdownx.arithmatex` if LaTeX)
* [ ] Properties list
* [ ] Implementations:

  * Subheading per variant
  * Code block with language `python`
* [ ] Complexity section
* [ ] Applications (support either dict “Domain: details” or strings)
* [ ] Links section (external refs)
* [ ] References list: render IDs as citations (Phase 7)
* [ ] Footer “Generated from YAML”

**DoD:** Templates render without missing keys, and whitespace/heading levels look clean.

---

# Phase 6 — Navigation

* [ ] **Option A (simple):** keep a static `nav:` with just `Home: index.md` + `SUMMARY.md`:

  ```yaml
  nav:
    - Home: index.md
    - Algorithms: SUMMARY.md
  ```
* [ ] **Option B (literate-nav):** add plugin and generate nested `SUMMARY.md` with sections per family
* [ ] **Option C (manual):** generate a `nav.yml` with `gen-files` and include it via `mkdocs.yml` (advanced)

**DoD:** Sidebar/nav shows families → algorithms with correct relative links.

---

# Phase 7 — Tags & References Pages (Optional but Powerful)

## 7.1 Tag index pages

* [ ] Iterate `shared/tags.yaml`
* [ ] For each tag, generate `/tags/<id>.md` that lists every algorithm carrying that tag
* [ ] Add a global `/tags/index.md` with a styled grid of tag cards

## 7.2 References

* **Option A (lightweight):** switch to `shared/refs.yaml` with pre-formatted citations

  * [ ] Create `/refs/index.md` listing all citations alphabetically
  * [ ] Render per-page refs by filtering to `algo.refs`
* **Option B (BibTeX):** use `pybtex` to format citations

  * [ ] Load `.bib`
  * [ ] Add a formatter to produce human-readable strings
  * [ ] Cache parsed entries between pages

**DoD:** Clicking a tag shows relevant algorithms; references are readable and consistent.

---

# Phase 8 — Validation & Error Handling

* [ ] Define Pydantic models for Family & Algorithm (optional but recommended)
* [ ] Validate each YAML; on failure:

  * [ ] fail build with a helpful error (file path, field name, expected type)
* [ ] Validate relationships:

  * [ ] `algo.family_id == family.id`
  * [ ] `refs` exist in `refs.(yaml|bib)`
  * [ ] `tags` exist in `shared/tags.yaml` (or log a warning and display raw id)

**DoD:** Bad YAML fails fast with clear messages; warnings logged for soft failures.

---

# Phase 9 — Content Migration (From Existing Markdown)

* [ ] Identify legacy markdown pages (e.g., `fibonacci.md`, `dynamic-programming.md`)
* [ ] For each:

  * [ ] Extract metadata into `family.yaml` / `algo.yaml`
  * [ ] Move math to `formulation`
  * [ ] Move code blocks into `implementations[].code`
  * [ ] Convert references → `refs` IDs
  * [ ] Map old tags → canonical IDs in `shared/tags.yaml`
* [ ] Leave a short legacy stub that links to the new generated page (optional)

**DoD:** Old content represented in YAML; generated pages are richer and DRY.

---

# Phase 10 — Developer Ergonomics

* [ ] Add `README.md` under `mkdocs_plugins/` explaining the build pipeline
* [ ] Add `justfile` or `make` targets:

  * [ ] `just docs-serve` → `uv run mkdocs serve`
  * [ ] `just docs-build` → `uv run mkdocs build`
  * [ ] `just docs-validate` → run a small validator script over YAML
* [ ] Pre-commit hooks: YAML lint, ruff, mypy on `mkdocs_plugins/`

**DoD:** One-command local dev; contributors understand how to add an algorithm.

---

# Phase 11 — CI/CD & Quality Gates

* [ ] GitHub Actions workflow:

  * [ ] `uv python` setup cache
  * [ ] Install MkDocs + plugins
  * [ ] Validate YAML (schema + referential)
  * [ ] `mkdocs build` (fail on warnings if you like)
  * [ ] (Optional) Deploy to GitHub Pages on `main`
* [ ] Broken link checker:

  * [ ] Enable `link-checker` step or run `mkdocs build -v` + a custom checker

**DoD:** PRs must pass validation + build; website deploys automatically on merge.

---

# Phase 12 — Advanced Enhancements (Backlog)

* [ ] **Status badges**: colored chips for `draft/stable/deprecated`
* [ ] **Search facets**: generate a simple tags page with quick filters
* [ ] **Cross-family references**: render `see_also` as links
* [ ] **Auto-diagrams**: support `media/` per family and embed via macros
* [ ] **Algorithm diffing**: optional tool to compare two algos’ fields in a table
* [ ] **Versioning**: a `version:` field per algorithm + changelog rendering
* [ ] **Localization**: optional `i18n` if you need multi-language docs
* [ ] **Performance**: cache YAML loads across macros; memoize BibTeX parse

**DoD:** Each enhancement documented; toggled via config flags.

---

## Acceptance Checklist (Per Feature)

* [ ] **Data discovery** works with any number of families/algorithms
* [ ] **Filtering & ordering** behave per `family.yaml`
* [ ] **Generated pages** link correctly, no orphaned routes
* [ ] **Edit links** map to YAML sources
* [ ] **Tags** resolve to names; unknown tags warn gracefully
* [ ] **Refs** render; unknown IDs reported
* [ ] **Templates** render clean headings, math, code blocks
* [ ] **Build** completes on CI; broken links fail PRs
* [ ] **Docs** include “How to add a new algorithm”

---

## What the local gen-AI should implement first (fast path)

1. **Scaffold** `gen_pages.py` with discovery + page generation (+ set\_edit\_path).
2. **Implement** `macros/main.py` with `family_page` + `algo_page` rendering via Jinja templates.
3. **Create** Jinja templates (`templates/family.md.j2`, `templates/algo.md.j2`) with clean sections.
4. **Hook up** shared tags; show tag badges on algo pages.
5. **(Optional)** render basic references from a YAML list; upgrade to BibTeX later.

That flow gets you end-to-end output quickly. Then layer on validation, tag indexes, and refs pages.
