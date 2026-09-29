# Documentation and source organization

The same navigation grammar is used in all three repositories; the source trees need not be artificially symmetric.

```text
README.md                      question, idea, present capabilities, selected result links
AGENTS.md                      concise edit/authority contract
CONTRIBUTING.md                one development entry point, not another rulebook
docs/
  index.md                     choose a reading path
  design/overview.md           concepts, mechanisms, boundaries, open questions
  design/program.md            common three-project story
  development/start.md         minimal build/run/validation instructions
  development/source-map.md    a few canonical entry points, not an exhaustive tree
  status/current.md            dated synopsis; source facts vs target intent
  results/index.md             selected studies and what they show
  results/<study>.md           generated study page with bounded claim, method, chart/table
  results/data/<study>.json    portable chart input with evidence provenance
  results/assets/              generated figure files
  archive/README.md            historical reading, labels and migration index
```

Keep detailed existing technical documents if useful. Link them from the appropriate landing page, classify them, and move only when the move materially reduces confusion. Do not create a second authoritative explanation beside an equally prominent old one. Prefer replacement/consolidation or an explicit archived/superseded header. QMD is not inferior to Markdown; retain documents needing its build features. New GitHub entry pages can be plain Markdown.

## Three reading paths

A visitor: README → design overview → selected result → one entry-point example.
A developer: source map → minimal build/use → relevant subsystem note/test.
An agent: AGENTS → current scoped task → only the relevant design/source/evidence.

## Inventories and move proposals

`tools/inventory.py` enumerates tracked files and separately reports untracked files when run locally. Its categories are initial routing suggestions, not deletion authority. For each important document record audience, current/historical status, replacement or destination, incoming links, and source references. The provided per-project migration maps seed the known problematic files. QMD contents and full tree inventories unavailable through the observer must be read locally before a move is accepted.

Use one `inputs/moves.json` with explicit source/destination/reason/consumers and approval per actual move. Never mass-move .todo-orchestrator, generated projections, vendor/submodule trees or evidence-linked paths. Stable evidence can remain physically under bench/ or experiments/ if it has a readable public index. Source organization is judged by development utility, not alphabetical or aesthetic symmetry.
