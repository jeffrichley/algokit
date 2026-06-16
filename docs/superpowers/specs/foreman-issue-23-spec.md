# Spec: add a `## Roadmap` placeholder section to README (issue #23)

## Goal
Add a short, forward-looking `## Roadmap` section to `README.md` so a new contributor landing on the repo can see what's planned at a glance without digging through the issue tracker. See [issue #23](https://github.com/jeffrichley/algokit/issues/23). This is a placeholder — not a milestone commitment — and the issue explicitly bounds the change at 4-10 lines of README-only content.

## Acceptance criteria
- [ ] `README.md` contains a new top-level (`##`) section whose title contains the word **"Roadmap"** (an emoji prefix is permitted and recommended to match the existing heading style; `🗺️` is suggested but not required).
- [ ] The new section is inserted between the existing `## 🤝 Contributing` section (currently `README.md` lines 144-159) and the existing `## 📄 License` section (currently line 161). No other section is reordered, removed, or rewritten.
- [ ] The section content is **4-10 lines** of prose/bullets total (the issue allows 4-8 in its Approach, 4-10 in its acceptance list; we use the looser bound). No fenced code blocks are required; if any are added, they must NOT contain shell commands or code that needs to stay in sync with the repo.
- [ ] The section contains a markdown link to the GitHub issues page at `https://github.com/jeffrichley/algokit/issues`. The link text is the Worker's choice (`open issues`, `the issue tracker`, etc.) but the URL must be exact.
- [ ] The section names at least 2-4 concrete forward-looking themes. Suggested seeds from the issue body: "more adaptive control variants (e.g., MIT rule, gain-scheduled LQR)", "expanded swarm intelligence catalog", "additional examples + notebooks per algorithm family". The Worker may adapt the exact wording but must keep the themes consistent with the directions the repo already shows (adaptive control under `src/algokit/`, swarm algorithms, example notebooks under `examples/` / `docs/`).
- [ ] The section ends with a one-line invitation for contributions or issue proposals (the issue suggests "Contributions and issue proposals welcome." — the Worker may use that verbatim or paraphrase).
- [ ] `just quality` passes (no new lint, format, typecheck, or test failures).
- [ ] `git diff README.md` against `origin/main` shows ONLY additions inside the new section — no deletions, no reflow, no edits to any neighbouring line.
- [ ] No file other than `README.md` is created or modified. In particular, no separate `ROADMAP.md` file is added (the issue's Out-of-Scope list forbids this).

## Approach
No design pattern applies — this is a single-file README documentation addition. The work is in scope discipline, not architecture: stay inside `README.md`, add one short section, change nothing else, and pick an insertion point that satisfies the issue's "visible without burying the install commands" constraint.

The chosen insertion point is the blank line between the end of `## 🤝 Contributing` (which closes at `README.md:159` with the trailing code-fence of its Quick Setup block) and the start of `## 📄 License` (`README.md:161`). The issue's source-file-pointer note says "between the existing 'Documentation' / 'Development' section and the License section near the end of the file"; the trailing third of the README (Documentation → Development → Contributing → License → Acknowledgments) is the band the issue is describing, and the Contributing-to-License gap is the most natural slot in that band because:

- It sits well below the Quick Start install block (`README.md:17-47`), so it never pushes install commands below the fold for a first-time visitor.
- It is immediately adjacent to the existing call-to-action ("contributions welcome") at the end of `## 🤝 Contributing`, so a forward-looking "here's where we're heading" section reinforces — rather than duplicates — that invitation.
- It does not split any existing topical group: Documentation, Development, and Contributing form a contiguous "how to engage with this project" cluster, and Roadmap belongs next to that cluster, not inside it.

Heading style matches the README's emoji-prefixed `##` pattern. `🗺️` is the recommended emoji because it is semantically obvious (a map = a roadmap) and is distinct from every other emoji already used in the README (`🚀`, `📦`, `🧪`, `📚`, `🛠️`, `🤝`, `📄`, `🎉`). The spec for issue #21 used `📖` for its docs section; using `🗺️` here keeps the README's emojis distinct.

The section content stays at the placeholder level the issue demands. No version numbers, no milestone targets, no dated commitments — just 2-4 themed bullets pointing at directions the repo already visibly invests in (adaptive control, swarm intelligence, examples/notebooks), plus a link to the live issues page so the README can stay stable while priorities shift. This is exactly the "static placeholder, not a complete roadmap" framing the issue body asks for.

## Sub-requests (topologically sorted)
1. **Compose the new section content** as a single contiguous markdown block. Required elements, in this order:
   - An `##`-level heading containing the word "Roadmap" (recommended: `## 🗺️ Roadmap`).
   - One sentence of lead-in prose that (a) frames the section as forward-looking, and (b) contains a markdown link to `https://github.com/jeffrichley/algokit/issues` for live priorities.
   - A short bulleted list (2-4 bullets) of forward-looking themes. Suggested seeds: "More adaptive control variants (e.g., MIT rule, gain-scheduled LQR)", "Expanded swarm intelligence catalog", "Additional examples + notebooks per algorithm family". The Worker may adapt wording but must keep each bullet at placeholder granularity — no specific dates, version numbers, or milestone names.
   - One closing line inviting contributions or issue proposals (e.g., "Contributions and issue proposals welcome.").
   - Total length: 4-10 lines including the heading and bullets, excluding the blank lines separating the section from its neighbours.
2. **Insert the composed block into `README.md`** at the exact insertion point: the blank line between the close of `## 🤝 Contributing` (the closing ` ``` ` of its Quick Setup block at line 159) and the start of `## 📄 License` (line 161). Preserve exactly one blank line between the new section and each neighbour.
3. **Verify the diff is additive only.** Run `git diff README.md` and confirm there are zero `-` lines (no deletions, no whitespace edits, no reflow) — only `+` lines inside the new section. If any unrelated line of `README.md` shows up in the diff, revert it before committing.
4. **Run the project's quality gate.** Run `just quality` (or `just check` if that is the configured Foreman gate) and confirm it exits zero. README-only changes should not trip lint, typecheck, or tests, but Foreman's Worker contract requires the gate to pass before the impl PR opens.

## File-level changes
| File | Change | Description |
|------|--------|-------------|
| `README.md` | Modify | Insert one new `##`-level section (recommended title `## 🗺️ Roadmap`) between the existing `## 🤝 Contributing` (ends line 159) and `## 📄 License` (starts line 161) sections. 4-10 lines total: a lead-in sentence with a link to the GitHub issues page, 2-4 forward-looking bullets, and a one-line invitation for contributions. No other lines touched. |

No other files are created or modified.

## Alternatives considered
1. **Add a separate top-level `ROADMAP.md` file and link to it from the README.** Rejected because the issue's Out-of-Scope list explicitly forbids adding a separate `ROADMAP.md`: "Adding a separate ROADMAP.md file — keep it inline for now." Inline is the requested shape.
2. **Insert the section higher up — for example, between `## 📚 Documentation` and `## 🛠️ Development`.** Rejected because the issue's source-file-pointer guidance points at the band "between the existing 'Documentation' / 'Development' section and the License section near the end of the file." A higher insertion point would (a) split the existing "how to engage with this project" cluster (Docs → Dev → Contributing) and (b) push install commands closer to the fold for visitors with shorter viewports. The Contributing→License gap satisfies both the visibility and the don't-bury-install constraints.
3. **Make the Roadmap a `###` subsection of `## 🤝 Contributing`.** Rejected because the issue's first acceptance criterion specifies a `## Roadmap` section (heading, not subheading) and the Approach explicitly calls it a "section." Demoting it would also blur the line between "how to contribute" and "what's planned," which are distinct concerns.
4. **Populate the Roadmap with concrete version targets and dated milestones (e.g., "v0.16 — MRAC variants by Q3").** Rejected because the issue's Out-of-Scope list explicitly excludes "Detailed milestone planning or version targets — this is a placeholder, not a commitment." The Worker must keep the bullets at placeholder granularity.

## Open questions
None. The insertion point, recommended emoji, suggested theme bullets, link target, length budget, and quality-gate command are all verifiable from the issue body and files already in the repo (`README.md`, `justfile`, `docs/superpowers/specs/foreman-issue-21-spec.md` for house style).

## Out of scope
- Adding a separate `ROADMAP.md` file (explicitly forbidden by the issue).
- Detailed milestone planning, version targets, dates, or anything that would turn the placeholder into a commitment.
- Any code, test, configuration, or `pyproject.toml` change — this is README-only.
- Rewriting, reordering, reformatting, or even whitespace-editing any other section of `README.md`, including the immediately adjacent `## 🤝 Contributing` and `## 📄 License` sections.
- Adding new badges, hosted-docs links, or modifying the existing badge block (`README.md:3-13`).
- Restructuring the README (table of contents, section ordering, emoji palette swap, etc.).
- Adding new `just` recipes, CI workflows, or docs pages related to the roadmap.
