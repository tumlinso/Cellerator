# CE-CCP1-M90: Part One compiler family final acceptance

> **PROPOSED PRE-LEDGER RECORD.** This file is not managed by Todo Orchestrator and does not authorize implementation until the apply-ready plan is manually applied and the run is explicitly activated.

## Objective

Integrate and validate all P90 workstreams, freeze shared interfaces, and publish milestone M90.

## State

- Lifecycle: proposed / planned
- Execution: inactive
- Kind: `validation_task`
- Parallel policy: `exclusive_integration`

## Program role

Integration-lane-only task. It may repair temporary structural worktree breakage but may not absorb Part Two.

## Ownership

- exclusive: `CMakeLists.txt`
- exclusive: `cmake`
- exclusive: `include/Cellerator`
- exclusive: `src`
- exclusive: `tools`
- exclusive: `stdlib`
- exclusive: `profiles`
- exclusive: `docs`
- exclusive: `README.md`
- exclusive: `components/CellShard`
- read: `.`
- forbidden: `.todo-orchestrator`
- shared lock: `ce-ccp1-root-build`
- shared lock: `ce-ccp1-package-exports`
- shared lock: `ce-ccp1-doc-authority`
- shared lock: `ce-ccp1-cellshard-gitlink`
- claim lock: `ce-ccp1-root-build`
- claim lock: `ce-ccp1-package-exports`
- claim lock: `ce-ccp1-doc-authority`
- claim lock: `ce-ccp1-cellshard-gitlink`

## Dependencies

- task CE-CCP1-J01-012 state done
- task CE-CCP1-J02-014 state done
- task CE-CCP1-J03-013 state done
- checkpoint CE-CCP1-MILESTONE-M80 state reached

## Interfaces

- None directly.

## Invariants

- `CE-CCP1-INV-CENTRAL-INTEGRATION`
- `CE-CCP1-INV-PRESERVE-JBC`

## Completion contract

- required: All workstreams in P90 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P90 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Produced artifacts

- `docs/compiler/milestones/m90_receipt.md` (integration_receipt)

## Gates and resources

- gate `CE-CCP1-M90-INTEGRATION-GATE`: `ctest --test-dir build --output-on-failure -L ce_ccp1_m90`
- resource `accelerator:any` during `gate`

## Completion criteria

- required: All workstreams in P90 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P90 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Next action

Collect isolated lane receipts for P90, verify hashes and interfaces, integrate central files, run label ce_ccp1_m90, and publish CE-CCP1-MILESTONE-M90.
