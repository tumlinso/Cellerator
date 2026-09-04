# Compiler Part One milestone M00 receipt

## Milestone

- Task: `CE-CCP1-M00`
- Run: `CE-CCP1-RUN-V1`
- Result: architecture, ownership, migration, and source-layout authority frozen
- Integration base: `31e491ed29de0fcde70259cbeab8c5c7ad353485`

## Integrated P00 workstreams

Project Control integrated the four isolated lane artifacts in rendezvous order.
Each artifact was applied as its material source delta, validated in the managed
M00 workspace, and frozen as an integration commit.

| Lane | Terminal task | Frozen integration commit | Material content hash |
|---|---|---|---|
| `CE-CCP1-L-A01` | `CE-CCP1-A01-009` | `3ce19eed28c5246fe641dff196e6fa46dd720889` | `3a49e1f4f31a84b03af6573bdfff02ed2b09035029e93026a59248dcec50867b` |
| `CE-CCP1-L-A02` | `CE-CCP1-A02-012` | `f322aefa7d621c9fc1f21873f29b8217e26f4802` | `cc50992ca16b24dc0297c24643a2714aef46b1fbf0b97e6448dd9e9c8f5cffc4` |
| `CE-CCP1-L-A03` | `CE-CCP1-A03-014` | `065a1d48e392f45820266f804272315ce04eb42a` | `616de04eef8db305a8e86a970b8a78e4e396cf092bfbd0a5989dbb62fc95d13d` |
| `CE-CCP1-L-A04` | `CE-CCP1-A04-010` | `e30174410d329659ffc1325b0c296084db531476` | `9a01679bf98921263cf6ecda4e2a826a53369a4c34588ee57a8f011b8757c73f` |

The integrated tree preserves the A01 authority evidence, the 979-row A02 JBC
migration inventory, the A03 compiler-ownership rehoming contract, and the A04
source-layout contract. No Part Two JIT or deep CellShard runtime ownership was
introduced.

## Frozen interfaces

| Interface | Version | Project Control content hash |
|---|---:|---|
| `CE-CCP1-I01-AUTHORITY-BASELINE` | 1 | `7863a75faca6d53df3d0ac17735bb8f73f67839315a62da4ac0c87377895bca0` |
| `CE-CCP1-I02-JBC-MIGRATION-MANIFEST` | 1 | `66109e87b3aad1b71a01a15031ca3296f125e807cd50eaccfda5c3494571587e` |
| `CE-CCP1-I03-COMPILER-OWNERSHIP` | 1 | `3478e9787fbee8e66f1c12dba0d69641d01605ef2316420f7d74cac02421b1d0` |
| `CE-CCP1-I04-SOURCE-LAYOUT` | 1 | `98245e15975ccf7a311d6edb65ffd2257bb5d971db331ea9ef50cbb6648bb9da` |

All four interfaces are frozen in the live Project Control authority. Their
source-linked receipts remain in the integrated tree; downstream lanes must
consume the frozen versions and may not silently reinterpret them.

## Validation

- The repository configured successfully with
  `cmake -S . -B build -DBASEPLANE_SOURCE_DIR=/home/tumlinson/Baseplane`.
- The full configured tree built successfully with
  `cmake --build build -j "$(nproc)"` on the local Tesla V100 `sm_70` system.
- Project Control ran `CE-CCP1-M00-INTEGRATION-GATE` after each lane merge and
  recorded passed evidence IDs `4f128ac0-6f71-4283-8804-e3c022be5173`,
  `ebc944f4-d8cd-4667-b880-07c37bca49bf`,
  `d478c8c7-4f7f-4fa4-bdb7-d9d4cafca4d1`, and
  `149cebf0-468d-4400-b7ca-80663366306a`.
- The declared command was
  `ctest --test-dir build --output-on-failure -L ce_ccp1_m00`. It returned zero,
  but the current build registers no tests under that label; the focused A01-A04
  receipt validators therefore remain the substantive lane evidence.

## Boundary

M00 freezes the authority needed by downstream Part One implementation. It does
not claim that the compiler products already exist, does not transfer storage or
application mechanics from CellShard, and does not authorize deferred Part Two
JIT or deep runtime work.
