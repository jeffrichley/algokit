# Spec: add "Build & Serve the Docs Site Locally" README section (issue #21)

## Goal
Add a short, self-contained section to `README.md` that tells a new contributor exactly how to install the docs dependency group, serve the mkdocs site locally with live reload, and produce a static build. This addresses the gap noted in issue #21: the existing Quick Start covers package install but never explains how to preview docs changes, even though the project ships an mkdocs-material site under `docs/` and configures it via `mkdocs.yml`.

## Acceptance criteria
- [ ] `README.md` contains a new top-level (`##`) section whose title contains the phrase **"Build & Serve the Docs Site Locally"** (emoji prefix permitted to match the existing heading style).
- [ ] The new section is inserted immediately AFTER the existing `## 📚 Documentation` section (currently `README.md` lines 112-115) and BEFORE the existing `## 🛠️ Development` section. No other section is reordered.
- [ ] The section names, in order, (a) the dep-install command, (b) the live-reload serve command and its URL, (c) the static build command and its output directory, and (d) a pointer to `mkdocs.yml`.
- [ ] The install command is `uv sync --group docs` (NOT `--extra docs`) — `docs` lives in `pyproject.toml`'s `[dependency-groups]` block (PEP 735), not under `[project.optional-dependencies]`. This matches the `install-docs` recipe in the `justfile`.
- [ ] The serve command is `uv run mkdocs serve` and the documented URL is `http://127.0.0.1:8000/` (mkdocs default; `mkdocs.yml` does not override it).
- [ ] The build command is `uv run mkdocs build` and the documented output directory is `site/` (mkdocs default; the `justfile`'s `clean` recipe confirms it via `rm -rf site`).
- [ ] The justfile shortcuts `just docs-serve` and `just docs` are mentioned in one line as an idiomatic alternative — they already exist (`justfile:54-64`) and are the in-repo canonical wrappers.
- [ ] Total added content is **8-15 lines of prose + fenced code blocks**, matching the issue's length budget.
- [ ] No other text in `README.md` is rewritten, reordered, removed, or reformatted. A `git diff README.md` against `origin/main` shows ONLY additions inside the new section.
- [ ] `pyproject.toml`, `mkdocs.yml`, `justfile`, `docs/`, badges, and the hosted-docs URL block are NOT modified.

## Approach
This is a single-file README patch — no design pattern applies; it is straightforward documentation. The discipline here is scope containment: stay inside `README.md`, add one section, change nothing else. The Google engineering principle that fits is "make the right thing easy" — a contributor opening the README should be able to copy three commands and preview docs without hunting through `justfile` or `pyproject.toml`.

The new section slots between the existing `## 📚 Documentation` (which lists API/contributing links) and the existing `## 🛠️ Development` (which covers quality checks and project layout). That placement keeps doc-related content adjacent without modifying either neighbour. Heading style matches the README's emoji-prefixed `##` pattern; `📖` is a natural fit because it is distinct from `📚` already used by the immediately preceding section.

The single non-obvious decision is the install command. The issue body claims the docs deps live in `[project.optional-dependencies]`, which would imply `uv sync --extra docs`. That is wrong: `pyproject.toml` puts `docs = [...]` under `[dependency-groups]` (lines 111-129), and the `justfile`'s `install-docs` recipe canonically uses `uv sync --group docs` (`justfile:143-144`). The acceptance criterion "Commands are accurate to algokit's actual layout — verified against `pyproject.toml` + `mkdocs.yml`" governs here: follow the layout the repo actually has, not the framing in the issue body.

The Worker should also mention the existing justfile shortcuts (`just docs-serve`, `just docs`) in one sentence, because they are the in-repo idiomatic wrappers and contributors will see them in the `justfile` regardless. This stays inside the 8-15 line budget and adds zero new files or recipes.

## Sub-requests (topologically sorted)
1. **Compose the new section content** as a single contiguous markdown block. Required elements, in this order:
   - An `##`-level heading whose text contains "Build & Serve the Docs Site Locally" (emoji prefix `📖` recommended).
   - One short sentence of lead-in prose explaining what the section covers.
   - A fenced ```` ```bash ```` block with three commands and inline `#`-comments, exactly:
     ```
     # Install the docs dependency group
     uv sync --group docs

     # Serve with live reload at http://127.0.0.1:8000/
     uv run mkdocs serve

     # Build a static site into ./site/
     uv run mkdocs build
     ```
   - One short sentence noting the justfile shortcuts: `just docs-serve` (live reload) and `just docs` (build).
   - One short sentence pointing readers to `mkdocs.yml` for further configuration (theme, plugins, nav).
2. **Insert the composed block into `README.md`** at the exact insertion point: the blank line between the end of the existing `## 📚 Documentation` section (after line 115, which reads `- **[Contributing Guide](CONTRIBUTING.md)**: How to contribute to the project`) and the start of the existing `## 🛠️ Development` section (currently line 117, `## 🛠️ Development`). Preserve a single blank line separating the new section from each neighbour.
3. **Verify the diff is additive only.** Run `git diff README.md` and confirm there are zero `-` lines (no deletions, no reflow) — only `+` lines inside the new section. If any other line of `README.md` shows up in the diff, revert it before committing.

## File-level changes
| File | Change | Description |
|------|--------|-------------|
| `README.md` | Modify | Insert one new `##`-level section ("Build & Serve the Docs Site Locally") between the existing `## 📚 Documentation` and `## 🛠️ Development` sections. 8-15 lines of prose + a single fenced bash block. No other lines touched. |

No other files are created or modified.

## Alternatives considered
1. **Make the new content an `###` subsection of the existing `## 📚 Documentation` section.** Rejected because the issue's Approach explicitly asks for a new section, the acceptance criterion specifies the section title as a heading (not a subheading), and demoting it would require restructuring the existing Documentation section — which the issue's Out-of-Scope list forbids.
2. **Use `uv sync --extra docs` as the issue body suggests.** Rejected because `docs` is defined under `[dependency-groups]` in `pyproject.toml` (PEP 735), not under `[project.optional-dependencies]` (PEP 621). The `--extra` flag would fail with "no extra named 'docs'". The `justfile` already canonically uses `--group docs`. The acceptance criteria require accuracy over fidelity to the issue's framing.
3. **Add only the raw `mkdocs` commands and skip the justfile shortcuts.** Rejected because the `justfile` already exposes `docs-serve` / `docs` / `docs-dev` recipes and they are the idiomatic in-repo entry points; omitting them would leave a new contributor confused when they discover the `justfile` later. Mentioning them in one sentence costs nothing against the 8-15 line budget.
4. **Replace the existing `## 📚 Documentation` section entirely with a combined build-and-serve + links section.** Rejected because the issue's Out-of-Scope list forbids rewriting other sections.

## Open questions
None. The dependency group location, default serve URL, default build output directory, justfile shortcut names, and heading style are all verifiable from files already in the repo (`pyproject.toml`, `mkdocs.yml`, `justfile`, `README.md`).

## Out of scope
- Editing `docs/` content of any kind.
- Editing `mkdocs.yml`, `justfile`, `pyproject.toml`, or any other config file.
- Adding badges to `README.md`, linking the hosted docs site, or modifying the existing badge block (`README.md:3-13`).
- Rewriting, reordering, or reformatting any other README section, including the existing `## 📚 Documentation` section that the new section will sit beside.
- Adding new justfile recipes for docs (e.g., a `docs-install` recipe) — the existing `install-docs` recipe is sufficient.
- Changing the README's overall structure (table of contents, section ordering, etc.).
