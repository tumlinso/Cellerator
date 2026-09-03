# CE-CCP1-M40: Planning IR and Cellerator-owned JBC compiler logic integrated

> **PROPOSED PRE-LEDGER RECORD.** This file is not managed by Todo Orchestrator and does not authorize implementation until the apply-ready plan is manually applied and the run is explicitly activated.

## Objective

Integrate and validate all P40 workstreams, freeze shared interfaces, and publish milestone M40.

## State

- Lifecycle: proposed / planned
- Execution: inactive
- Kind: `validation_task`
- Parallel policy: `exclusive_integration`

## Program role

Integration-lane-only task. It may repair temporary structural worktree breakage but may not absorb Part Two.

## Ownership

- exclusive: `include/Cellerator/compiler/discovery`
- exclusive: `include/Cellerator/compiler/composition`
- exclusive: `include/Cellerator/compiler/program`
- exclusive: `include/Cellerator/compiler/planning`
- exclusive: `components/CellShard`
- exclusive: `docs/compiler/migration`
- read: `.`
- forbidden: `.todo-orchestrator`
- shared lock: `ce-ccp1-jbc-migration`
- shared lock: `ce-ccp1-ceir-registry`
- claim lock: `ce-ccp1-jbc-migration`
- claim lock: `ce-ccp1-ceir-registry`

## Dependencies

- task CE-CCP1-E01-016 state done
- task CE-CCP1-E02-018 state done
- task CE-CCP1-E03-018 state done
- task CE-CCP1-E04-018 state done
- checkpoint CE-CCP1-MILESTONE-M30 state reached

## Interfaces

- None directly.

## Invariants

- `CE-CCP1-INV-CENTRAL-INTEGRATION`
- `CE-CCP1-INV-PRESERVE-JBC`

## Completion contract

- required: All workstreams in P40 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P40 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Produced artifacts

- `docs/compiler/milestones/m40_receipt.md` (integration_receipt)

## Gates and resources

- gate `CE-CCP1-M40-INTEGRATION-GATE`: `ctest --test-dir build --output-on-failure -L ce_ccp1_m40`

## Completion criteria

- required: All workstreams in P40 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P40 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Next action

Collect isolated lane receipts for P40, verify hashes and interfaces, integrate central files, run label ce_ccp1_m40, and publish CE-CCP1-MILESTONE-M40.
