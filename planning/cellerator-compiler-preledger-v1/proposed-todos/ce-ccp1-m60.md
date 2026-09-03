# CE-CCP1-M60: Reflection, open passes, self-transforms, validation modes, and provenance integrated

> **PROPOSED PRE-LEDGER RECORD.** This file is not managed by Todo Orchestrator and does not authorize implementation until the apply-ready plan is manually applied and the run is explicitly activated.

## Objective

Integrate and validate all P60 workstreams, freeze shared interfaces, and publish milestone M60.

## State

- Lifecycle: proposed / planned
- Execution: inactive
- Kind: `validation_task`
- Parallel policy: `exclusive_integration`

## Program role

Integration-lane-only task. It may repair temporary structural worktree breakage but may not absorb Part Two.

## Ownership

- exclusive: `include/Cellerator/compiler/reflection`
- exclusive: `include/Cellerator/compiler/pass`
- exclusive: `include/Cellerator/compiler/diagnostics`
- exclusive: `src/compiler`
- exclusive: `include/Cellerator/compiler.hh`
- read: `.`
- forbidden: `.todo-orchestrator`
- shared lock: `ce-ccp1-ceir-registry`
- shared lock: `ce-ccp1-umbrella-headers`
- claim lock: `ce-ccp1-ceir-registry`
- claim lock: `ce-ccp1-umbrella-headers`

## Dependencies

- task CE-CCP1-G01-016 state done
- task CE-CCP1-G02-018 state done
- task CE-CCP1-G03-016 state done
- checkpoint CE-CCP1-MILESTONE-M50 state reached

## Interfaces

- None directly.

## Invariants

- `CE-CCP1-INV-CENTRAL-INTEGRATION`

## Completion contract

- required: All workstreams in P60 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P60 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Produced artifacts

- `docs/compiler/milestones/m60_receipt.md` (integration_receipt)

## Gates and resources

- gate `CE-CCP1-M60-INTEGRATION-GATE`: `ctest --test-dir build --output-on-failure -L ce_ccp1_m60`

## Completion criteria

- required: All workstreams in P60 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P60 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Next action

Collect isolated lane receipts for P60, verify hashes and interfaces, integrate central files, run label ce_ccp1_m60, and publish CE-CCP1-MILESTONE-M60.
