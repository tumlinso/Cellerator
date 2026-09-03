# CE-CCP1-M50: Realization IR and CPU/NVIDIA backend foundation integrated

> **PROPOSED PRE-LEDGER RECORD.** This file is not managed by Todo Orchestrator and does not authorize implementation until the apply-ready plan is manually applied and the run is explicitly activated.

## Objective

Integrate and validate all P50 workstreams, freeze shared interfaces, and publish milestone M50.

## State

- Lifecycle: proposed / planned
- Execution: inactive
- Kind: `validation_task`
- Parallel policy: `exclusive_integration`

## Program role

Integration-lane-only task. It may repair temporary structural worktree breakage but may not absorb Part Two.

## Ownership

- exclusive: `include/Cellerator/compiler/backend`
- exclusive: `src/compiler/backend`
- exclusive: `cmake/compiler`
- exclusive: `cmake/providers`
- exclusive: `include/Cellerator/compiler/ir/realization`
- read: `.`
- forbidden: `.todo-orchestrator`
- shared lock: `ce-ccp1-backend-registry`
- claim lock: `ce-ccp1-backend-registry`

## Dependencies

- task CE-CCP1-F01-018 state done
- task CE-CCP1-F02-014 state done
- task CE-CCP1-F03-015 state done
- task CE-CCP1-F04-013 state done
- checkpoint CE-CCP1-MILESTONE-M40 state reached

## Interfaces

- None directly.

## Invariants

- `CE-CCP1-INV-CENTRAL-INTEGRATION`

## Completion contract

- required: All workstreams in P50 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P50 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Produced artifacts

- `docs/compiler/milestones/m50_receipt.md` (integration_receipt)

## Gates and resources

- gate `CE-CCP1-M50-INTEGRATION-GATE`: `ctest --test-dir build --output-on-failure -L ce_ccp1_m50`
- resource `accelerator:any` during `gate`

## Completion criteria

- required: All workstreams in P50 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P50 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Next action

Collect isolated lane receipts for P50, verify hashes and interfaces, integrate central files, run label ce_ccp1_m50, and publish CE-CCP1-MILESTONE-M50.
