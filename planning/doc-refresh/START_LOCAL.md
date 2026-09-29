# Local implementation entry

## Continuity of existing work

This documentation epic is additive. It does not supersede, retire, or cancel
any active or scheduled Todo. Before accepting the epic, verify that those
tasks remain continuable with their claims, source paths, interfaces, and
required validation routes intact. If a useful source move affects them,
coordinate the owning task and repair its references within that authority.
Do not record this epic complete while an existing task is stranded by its
changes.

This directory is the prepared documentation package for this one repository. Its `project.json` identifies the authority; do not apply a sibling plan here.

From the repository root:

```sh
# Observe first: current source, AGENTS, actual overlapping claims/leases.
python3 planning/doc-refresh/tools/inventory.py --repo . --out planning/doc-refresh/inputs/inventory.json
project-control plan validate --project YOUR_REGISTERED_PROJECT --file planning/doc-refresh/plan.v2.json
# Inspect the native diff, then use the current Project Control operator route to apply.
project-control plan apply --project YOUR_REGISTERED_PROJECT --file planning/doc-refresh/plan.v2.json
```

Replace the single project token with the registered id in `project.json` (`cellerator`, `baseplane` or `glasshelix`). No second Todo bootstrap is needed: all three authorities already exist. CLI spelling should be checked against the installed deployment if it changed; do not install another copy just to match a historical command.

The four outcome tasks are described in `shared/TASKS.md`. Most prose lives in `templates/`; replace only the final path/status blocks in `inputs/bindings.json`. The prepared migration map and benchmark shortlist state what to retain, consolidate or verify.

```sh
python3 planning/doc-refresh/tools/render.py --project YOUR_REGISTERED_PROJECT --repo . --out planning/doc-refresh/staged --figures
python3 planning/doc-refresh/tools/check.py --local --project YOUR_REGISTERED_PROJECT --repo .
```

Staging does not copy anything over README/AGENTS/source. Review the diff and copy the intended files locally using the normal scoped workflow. Keep the managed AGENTS block byte-for-byte. Source moves are manual, recorded and validated; no deletion or move script is supplied.

After real review, fill `inputs/review.json` and the chosen result records, remove resolved review notes, rerender with `--final`, and apply the staged public files. Then:

```sh
python3 planning/doc-refresh/tools/check.py --final --project YOUR_REGISTERED_PROJECT --repo .
```

The optional GH probe is an analytical specification plus oracle, not product evidence. Benchmark execution uses the existing scheduler/resources. Finish with `inputs/handoff.json` containing actual commands/results, qualified HEAD, moves and selections. The program-level checker consumes all three receipts read-only.

## Handoff record

Copy `inputs/handoff.template.json` to `inputs/handoff.json` only when recording the completed work. Each validation command entry uses:

```json
{"argv":["the","actual","command"],"returncode":0,"expected_exit_code":0,"basis":"executed","evidence_path":"planning/doc-refresh/evidence/build-check.txt"}
```

Use `basis: "verified_existing_receipt"` only when an existing receipt was actually verified and still qualifies the unchanged source; do not relabel it as executed. A source move requires fresh affected build/test checks. Create the handoff before the final documentation gate; record that gate's own output afterward, avoiding a receipt that claims its own future success.

The qualified HEAD identifies the checked source, with pending local edits stated in the receipt. Do not commit merely to satisfy a documentation script. After an intentional commit changes HEAD, review the delta and update the qualification record. This is a local implementation review, not an additional user-approval ceremony.

Cellerator-specific activation note: the full remote preview returned `bounded_read_failed`, while context reads worked. Run local native validation first and act on its actual diagnostics; do not apply on the strength of package-only tests. Keep the existing authority.
