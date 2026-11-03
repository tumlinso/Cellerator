# Planning and contract-review validation

Historical planning/review record. See [the realized implementation](IMPLEMENTATION.md) and [actual type map](implemented-type-map.csv) for the realized cutover and its evidence.

Validated 2026-11-01. The revised proposal includes [SCOPE.md](SCOPE.md), the
clean source cutover, permitted format migration and required multi-device
first milestone. Document checks concern the plan. The five separately listed
host tests concern the inspected Cell redesign, not the proposed data runtime.

| Check | Result |
|---|---|
| Protected `cellular_geometry.md` | SHA-256 before and after is `d37f504e5c1ab629334f4127dcaa60d008afe94e560d41aa936d8ba161124254`; unchanged. |
| Type map | 71 unique row IDs; all current source files exist and contain the named symbol anchor. This is a lexical source-reference check, not AST or exhaustive consumer validation. |
| Markdown references | Checked local links in DESIGN/MIGRATION/SCOPE resolve to existing files. Proposed destination paths in the CSV are intentionally future paths. |
| Document structure | Balanced code fences and no trailing whitespace in the planning documents. |
| Rendering | DESIGN, MIGRATION and SCOPE rendered successfully to standalone HTML with `quarto pandoc`; resulting structure and substantive text were checked. |
| Source/design review | The initial proposal received independent contract review. A second review inspected the typed Cell redesign and revised cleanup/distribution scope; findings and checks are recorded below. |
| Workspace preservation | The 32 pre-existing tracked dirty files captured in `/tmp/cellerator-data-scope-iy89zfdi/existing-work.json` remain byte-identical. All edits made for this review are confined to this planning directory. No source/build/generated-map/workflow mutation was made. |

Review resolved missing operation/condition/order scoping in omission rules,
ambiguity between entity and occurrence identity, the v1 restriction on
zero-participant support edges, and the required warp-tile width/mask/count/
offset semantics. These corrections are in the proposal, not in library code.

C++ design sketches were **not compiled**. No proposed data implementation,
full software build, installed-package/module qualification, CUDA execution,
GPU admission, benchmark, profiling, registration, migration or performance
promotion was performed. Future implementation acceptance is specified in
MIGRATION.md, including actual two-device forward/backward.

## Cell/model review

Independent review found no confirmed defect in the inspected typed Cell/model
contract. It identified three requirements for data integration: keep data
types usable without Cell, obtain revision/readiness from owners rather than
Cell calls, and bind required semantic identity/order separately from generic
instance views. These are incorporated in DESIGN, SCOPE and the type map.

The reviewer compiled and ran the following five focused host checks from the
repository root with GCC `c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`.
Each compile and executable returned exit 0 with empty output. No tests were
rerun merely to restate those results.

```sh
c++ -std=c++17 -Wall -Wextra -pedantic -Iinclude tests/cell/cell_core_test.cc -o /tmp/cellerator-contract-review-cell-core
/tmp/cellerator-contract-review-cell-core
c++ -std=c++17 -Wall -Wextra -pedantic -Iinclude tests/cell/cell_views_bridge_test.cc src/cell.cc -o /tmp/cellerator-contract-review-bridge
/tmp/cellerator-contract-review-bridge
c++ -std=c++20 -Wall -Wextra -pedantic -Iinclude tests/cell/cell_adapters_test.cc src/cell.cc -o /tmp/cellerator-contract-review-adapters
/tmp/cellerator-contract-review-adapters
c++ -std=c++17 -Wall -Wextra -pedantic -fno-rtti -Iinclude tests/cell/typed_core_no_rtti_test.cc -o /tmp/cellerator-contract-review-no-rtti
/tmp/cellerator-contract-review-no-rtti
c++ -std=c++17 -Wall -Wextra -pedantic -Iinclude tests/framework/native/model_contract_test.cc -o /tmp/cellerator-contract-review-dynamics
/tmp/cellerator-contract-review-dynamics
```

These tests exercise the current other-agent redesign. They do not establish
the future cache/publication integration, distributed numerical behavior or
native NN package/module qualification. The binaries remain under `/tmp`;
stdout/exit statuses were observed in tool results, not redirected to log files.

## Scope-review findings addressed

The independent scope review identified four acceptance gaps, corrected in
SCOPE section 4 and MIGRATION M6:

- Require staged transfer; qualify direct device transfer on capable hardware
  and record unsupported paths without claiming a pass.
- Include nonzero remote-condition, participant and parameter VJPs, plus an
  inactive case with an independently nonzero baseline.
- Compare a deterministic shared-parameter update and the next forward after
  replica refresh against the reference.
- Require the same connected distributed example to consume specialist views
  and prepared dependency/condition/omission contracts directly on both devices.

The final bounded review reported no remaining material findings. Reviewed
SHA-256 values were `c6ae08a19593a797b807f0f0a8eb8e817d265483f3947657e655d8f28554fddd`
for SCOPE and `4f2df03d353c1ff559d5387a07a20cc23446ca7bc7a35f13ae214179e4bf13b4`
for MIGRATION.

The source inventory also distinguished redundant participant-posting builds
from separate condition/read indexes and identified the concrete mixed-TU
extraction seams in SCOPE. It performed no source changes or GPU execution.

Revised rendering commands (output kept outside the checkout):

```sh
quarto pandoc planning/native-data-geometry-2026-11-01/DESIGN.md \
  --from gfm --to html --standalone --metadata title='Native C++ data design' \
  --output /tmp/cellerator-data-scope-iy89zfdi/design.html
quarto pandoc planning/native-data-geometry-2026-11-01/MIGRATION.md \
  --from gfm --to html --standalone --metadata title='Native C++ data migration plan' \
  --output /tmp/cellerator-data-scope-iy89zfdi/migration.html
quarto pandoc planning/native-data-geometry-2026-11-01/SCOPE.md \
  --from gfm --to html --standalone --metadata title='Native data scope' \
  --output /tmp/cellerator-data-scope-iy89zfdi/scope.html
```
