# CE-CCP1-M30: Common CEIR, Semantic IR, and representative profile environment integrated

> **PROPOSED PRE-LEDGER RECORD.** This file is not managed by Todo Orchestrator and does not authorize implementation until the apply-ready plan is manually applied and the run is explicitly activated.

## Objective

Integrate and validate all P30 workstreams, freeze shared interfaces, and publish milestone M30.

## State

- Lifecycle: proposed / planned
- Execution: inactive
- Kind: `validation_task`
- Parallel policy: `exclusive_integration`

## Program role

Integration-lane-only task. It may repair temporary structural worktree breakage but may not absorb Part Two.

## Ownership

- exclusive: `include/Cellerator/compiler/ir`
- exclusive: `include/Cellerator/compiler/profile`
- exclusive: `src/compiler/ir`
- exclusive: `src/compiler/profile`
- exclusive: `cmake/compiler`
- read: `.`
- forbidden: `.todo-orchestrator`
- shared lock: `ce-ccp1-ceir-registry`
- claim lock: `ce-ccp1-ceir-registry`

## Dependencies

- task CE-CCP1-D01-014 state done
- task CE-CCP1-D02-016 state done
- task CE-CCP1-D03-015 state done
- checkpoint CE-CCP1-MILESTONE-M20 state reached

## Interfaces

- None directly.

## Invariants

- `CE-CCP1-INV-CENTRAL-INTEGRATION`

## Completion contract

- required: All workstreams in P30 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P30 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Produced artifacts

- `docs/compiler/milestones/m30_receipt.md` (integration_receipt)

## Gates and resources

- gate `CE-CCP1-M30-INTEGRATION-GATE`: `ctest --test-dir build --output-on-failure -L ce_ccp1_m30`

## Completion criteria

- required: All workstreams in P30 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P30 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Next action

Collect isolated lane receipts for P30, verify hashes and interfaces, integrate central files, run label ce_ccp1_m30, and publish CE-CCP1-MILESTONE-M30.
