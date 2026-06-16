# Spec: add "Build & Serve the Docs Site Locally" section to README (issue #21)

## Goal
Add a short, self-contained section to `README.md` that tells a new contributor exactly how to install the docs dependency group, serve the mkdocs site locally with live reload, build a static site, and where to look for further configuration. See [issue #21](https://github.com/jeffrichley/algokit/issues/21). This closes a discoverability gap: the project ships a full mkdocs-material site (`mkdocs.yml`, `docs/`) but the README never tells the reader how to preview it.

## Acceptance criteria
- [ ] `README.md` contains a new `##`-level section whose heading contains the phrase **"Build & Serve"** and the word **"Docs"** (recommended exact heading: `## 📖 Build & Serve the Docs Site Locally`). The build-and-serve framing from the issue is preserved.
- [ ] The new section is inserted between the existing `## 📚 Documentation` section (currently `README.md` lines 112-115) and the existing `## 🛠️ Development` section (currently line 117). No other section is reordered, removed, or rewritten.
- [ ] The section explicitly names, in code-fenced blocks the reader can copy-paste, all four of:
  1. The dep-install command — must be `uv sync --group docs` (NOT `uv sync --extra docs`; see Approach for why).
  2. The serve-with-live-reload command — `uv run mkdocs serve`.
  3. The static-build command — `uv run mkdocs build`.
  4. The default served URL — `http://127.0.0.1:8000/`.
- [ ] The section mentions the static-build output directory `site/` so the reader knows where the built site lands.
- [ ] The section includes a one-line pointer to `mkdocs.yml` for further configuration.
- [ ] Heading level uses `##` (not `###`) so it appears as a peer of the README's other top-level sections.
- [ ] Total section length (heading through last line, excluding the blank lines separating it from neighbours) is **8-20 lines**, including code blocks. The issue allows 8-15 lines "of content + code blocks"; we widen to 20 to allow for three small fenced blocks plus prose without forcing single-line collapse.
- [ ] `just quality` passes (no new lint, format, typecheck, or spellcheck failures).
- [ ] `git diff README.md` against `origin/main` shows ONLY additions inside the new section — no deletions, no reflow, no whitespace edits to neighbouring lines.
- [ ] No file other than `README.md` is created or modified.

## Approach
No design pattern applies — this is a single-file README documentation addition. The discipline is in (a) getting the install command right for THIS repo's dependency layout, and (b) staying inside the issue's scope guardrails.

**The issue body's example install command is wrong for this repo, and the spec must correct it.** The issue says "`uv sync --extra docs`, or the equivalent if the project uses dep-groups". algokit uses dep-groups — `pyproject.toml` declares `docs` under `[dependency-groups]` (lines 111-129), not under `[project.optional-dependencies]` (which contains only `viz = []`). The `[project.optional-dependencies]` block has no `docs` extra at all, so `uv sync --extra docs` would error. The canonical install command, already encoded in `justfile:144` as `install-docs`, is `uv sync --group docs`. The new README section MUST use this form.

**The serve / build commands and URL are verifiable from existing `justfile` recipes:**
- `justfile:63-64` `docs-serve: install-docs` → `uv run mkdocs serve` (mkdocs's built-in dev server serves at `http://127.0.0.1:8000/` by default and reloads on file change without needing extra flags).
- `justfile:54-55` `docs: install-docs` → `uv run mkdocs build` (mkdocs's default output directory is `site/`, confirmed by `justfile:158-163` `clean` recipe explicitly removing `site/`).
- `mkdocs.yml` is the site-config file; its presence at the repo root is what makes the bare `mkdocs serve` / `mkdocs build` calls work without flags.

**Insertion point: immediately after the existing `## 📚 Documentation` section, immediately before `## 🛠️ Development`.** The existing `## 📚 Documentation` section (lines 112-115) is a four-line index of static doc links (API Reference, Contributing Guide). The new "Build & Serve" section is topically a direct continuation of that — "and here's how to render those docs yourself" — so adjacency is correct. Inserting earlier would split the install instructions (Quick Start) from the related-content cluster (Documentation → Build-docs → Development → Contributing); inserting later would put it after Development (the wrong audience: Development is about running the code, Docs-build is about previewing the site).

**Heading style: emoji-prefixed `##`, recommended emoji `📖`.** Every existing top-level section in `README.md` uses this pattern (`🚀 Quick Start`, `📦 Usage`, `🧪 Testing`, `📚 Documentation`, `🛠️ Development`, `🤝 Contributing`, `📄 License`, `🎉 Acknowledgments`). `📖` (open book) is distinct from every emoji already used and is the documented choice the sibling spec for issue #23 expects from this spec ("The spec for issue #21 used `📖` for its docs section").

**Do NOT fix the stale `uv run dev docs` line in the Quick Start.** `README.md:46` shows `uv run dev docs` under "Development Setup", but there is no `dev` entry in `[project.scripts]` (only `algokit = "algokit.cli.main:app"` at `pyproject.toml:55`). This line is stale and broken. Fixing it is genuinely out of scope per the issue: "Nothing else in the README is rewritten or reorganized." The new section will supply correct, verified commands; the stale line can be cleaned up under a separate issue.

## Sub-requests (topologically sorted)
1. **Compose the new section** as a single contiguous markdown block at the structure shown below. The Worker may adjust wording but must keep each required element:
   - `## 📖 Build & Serve the Docs Site Locally` heading.
   - One sentence of lead-in prose introducing the section (e.g., "algokit ships a [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) site under `docs/`. To preview it locally:").
   - A small "Install the docs dependency group" subsection or paragraph with the fenced shell block:
     ````
     ```bash
     uv sync --group docs
     ```
     ````
   - A "Serve with live reload" paragraph with the fenced shell block:
     ````
     ```bash
     uv run mkdocs serve
     ```
     ````
     followed by one line naming the served URL: `http://127.0.0.1:8000/`.
   - A "Build a static site" paragraph with the fenced shell block:
     ````
     ```bash
     uv run mkdocs build
     ```
     ````
     followed by one line naming the output directory: `site/`.
   - A closing one-liner pointing at `mkdocs.yml` for configuration, e.g., "See `mkdocs.yml` for site config (theme, plugins, navigation)."
   - Total block length: 8-20 lines including heading, prose, and fenced code blocks; excluding the surrounding blank lines that separate the section from its neighbours.

2. **Insert the composed block into `README.md`** at the exact insertion point: between the close of `## 📚 Documentation` (after line 115's "[Contributing Guide](CONTRIBUTING.md): How to contribute to the project" bullet) and the start of `## 🛠️ Development` (line 117). Preserve exactly one blank line between the new section and each neighbour, matching the README's existing section-to-section spacing pattern.

3. **Verify the diff is additive only.** Run `git diff README.md` against `origin/main` and confirm there are zero `-` lines (no deletions, no whitespace edits, no reflow) — only `+` lines inside the new section. If any unrelated README line shows up in the diff, revert it before committing.

4. **Run the project's quality gate.** Run `just check` (the configured Foreman quality gate). Per `.github/workflows/` and project conventions, this should be the entrypoint. If `just check` is unavailable in this repo, fall back to `just quality` (`justfile:79`). Confirm exit zero. README-only changes should not trip lint/typecheck/tests, but the Worker contract requires the gate to pass before the impl PR opens.

5. **Commit and push.** Single commit with conventional-commit-style message: `docs: add "Build & Serve Docs Locally" section to README`.

## File-level changes
| File | Change | Description |
|------|--------|-------------|
| `README.md` | Modify | Insert one new `##`-level section (recommended heading `## 📖 Build & Serve the Docs Site Locally`) between the existing `## 📚 Documentation` section (ends line 115) and the existing `## 🛠️ Development` section (starts line 117). 8-20 lines total including three small fenced shell blocks: `uv sync --group docs`, `uv run mkdocs serve` (with URL `http://127.0.0.1:8000/`), `uv run mkdocs build` (with output dir `site/`), plus a closing pointer to `mkdocs.yml`. No other lines of `README.md` are touched. |

No other files are created or modified.

## Alternatives considered
1. **Use `uv sync --extra docs` as the issue body suggests.** Rejected: `pyproject.toml` declares `docs` under `[dependency-groups]` (line 111), not `[project.optional-dependencies]`. The only `optional-dependencies` entry is `viz = []`. `--extra docs` would fail; `--group docs` is the correct invocation, already encoded in `justfile:144`.
2. **Recommend the `just` recipes (`just docs-serve`, `just docs`) instead of bare `uv run mkdocs ...` commands.** Rejected: the just recipes are equivalent, but bare `uv run mkdocs serve` / `uv run mkdocs build` are more discoverable to a reader who doesn't yet know what `just` is. The issue's example commands are also written in the bare `uv run mkdocs ...` form. A future README pass can add the `just` aliases as a one-line "or, if you have `just` installed" mention; keep this spec focused.
3. **Also fix the stale `uv run dev docs` line in the existing Quick Start (`README.md:46`).** Rejected: the issue's Out-of-Scope is explicit — "Nothing else in the README is rewritten or reorganized" — and the issue's acceptance criteria say to add a section, not to edit existing ones. The stale line is a real bug, but it deserves its own issue so the diff stays small and reviewable.
4. **Insert the section earlier (between Quick Start and Usage) so it's higher in the page.** Rejected: the existing `## 📚 Documentation` section is the natural topical anchor — it already discusses docs — so placing the build/serve instructions immediately after it preserves topical clustering and avoids splitting the install/usage/testing/docs/development band. The issue says "between Quick Start and Contributing", which the chosen insertion point satisfies (Quick Start ends at line 47; Contributing starts at line 144; the chosen insertion is at line 116).
5. **Make the new content a `###` subsection of `## 📚 Documentation` instead of its own `##` section.** Rejected: the issue's first acceptance criterion says "a new section titled..." and the suggested heading form is `##`-level. Demoting to `###` would also reduce discoverability in the README's auto-generated table of contents.

## Open questions
None. Every command, URL, directory, file path, line number, and dep-group name in this spec is verifiable from files already in the repo: `README.md` (insertion point), `pyproject.toml` (dep-group location), `mkdocs.yml` (site config), `justfile` (canonical install/build/serve commands), and `docs/superpowers/specs/foreman-issue-23-spec.md` (house spec style + emoji-choice precedent).

## Out of scope
- Editing any other section of `README.md`, including the stale `uv run dev docs` reference at `README.md:46` (this is a real bug but belongs to a separate issue).
- Modifying `mkdocs.yml`, any file under `docs/`, or any docs-build configuration.
- Adding new badges, links to the hosted docs site at `https://jeffrichley.github.io/algokit`, or any other content in the README beyond the new section.
- Adding new `just` recipes, new `[project.scripts]` entries, or fixing the missing `dev` command referenced in the existing Quick Start.
- Reformatting, reordering, or reflowing any line of `README.md` outside the inserted block.
- Adding linkcheck steps, CI workflows, or pre-commit hooks for the docs site.
