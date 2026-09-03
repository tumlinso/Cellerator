# CE-CCP1-M80: celleratord core and Cellerator-aware semantic tooling integrated

> **PROPOSED PRE-LEDGER RECORD.** This file is not managed by Todo Orchestrator and does not authorize implementation until the apply-ready plan is manually applied and the run is explicitly activated.

## Objective

Integrate and validate all P80 workstreams, freeze shared interfaces, and publish milestone M80.

## State

- Lifecycle: proposed / planned
- Execution: inactive
- Kind: `validation_task`
- Parallel policy: `exclusive_integration`

## Program role

Integration-lane-only task. It may repair temporary structural worktree breakage but may not absorb Part Two.

## Ownership

- exclusive: `tools/celleratord`
- exclusive: `include/Cellerator/compiler/tooling`
- exclusive: `src/compiler/tooling`
- exclusive: `cmake/package`
- read: `.`
- forbidden: `.todo-orchestrator`
- shared lock: `ce-ccp1-celleratord-protocol`
- shared lock: `ce-ccp1-package-exports`
- claim lock: `ce-ccp1-celleratord-protocol`
- claim lock: `ce-ccp1-package-exports`

## Dependencies

- task CE-CCP1-I01-014 state done
- task CE-CCP1-I02-014 state done
- checkpoint CE-CCP1-MILESTONE-M70 state reached

## Interfaces

- None directly.

## Invariants

- `CE-CCP1-INV-CENTRAL-INTEGRATION`

## Completion contract

- required: All workstreams in P80 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P80 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Produced artifacts

- `docs/compiler/milestones/m80_receipt.md` (integration_receipt)

## Gates and resources

- gate `CE-CCP1-M80-INTEGRATION-GATE`: `ctest --test-dir build --output-on-failure -L ce_ccp1_m80`

## Completion criteria

- required: All workstreams in P80 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P80 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Next action

Collect isolated lane receipts for P80, verify hashes and interfaces, integrate central files, run label ce_ccp1_m80, and publish CE-CCP1-MILESTONE-M80.
