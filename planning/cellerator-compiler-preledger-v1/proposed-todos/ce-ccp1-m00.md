# CE-CCP1-M00: Architecture, ownership, and migration authority frozen

> **PROPOSED PRE-LEDGER RECORD.** This file is not managed by Todo Orchestrator and does not authorize implementation until the apply-ready plan is manually applied and the run is explicitly activated.

## Objective

Integrate and validate all P00 workstreams, freeze shared interfaces, and publish milestone M00.

## State

- Lifecycle: proposed / planned
- Execution: inactive
- Kind: `validation_task`
- Parallel policy: `exclusive_integration`

## Program role

Integration-lane-only task. It may repair temporary structural worktree breakage but may not absorb Part Two.

## Ownership

- exclusive: `docs/compiler/architecture`
- exclusive: `docs/compiler/migration`
- exclusive: `docs/compiler/source-layout`
- exclusive: `planning/cellerator-compiler-preledger-v1`
- read: `.`
- forbidden: `.todo-orchestrator`
- shared lock: `ce-ccp1-jbc-migration`
- shared lock: `ce-ccp1-doc-authority`
- claim lock: `ce-ccp1-jbc-migration`
- claim lock: `ce-ccp1-doc-authority`

## Dependencies

- task CE-CCP1-A01-009 state done
- task CE-CCP1-A02-012 state done
- task CE-CCP1-A03-014 state done
- task CE-CCP1-A04-010 state done

## Interfaces

- None directly.

## Invariants

- `CE-CCP1-INV-CENTRAL-INTEGRATION`
- `CE-CCP1-INV-PRESERVE-JBC`

## Completion contract

- required: All workstreams in P00 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P00 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Produced artifacts

- `docs/compiler/milestones/m00_receipt.md` (integration_receipt)

## Gates and resources

- gate `CE-CCP1-M00-INTEGRATION-GATE`: `ctest --test-dir build --output-on-failure -L ce_ccp1_m00`

## Completion criteria

- required: All workstreams in P00 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P00 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Next action

Collect isolated lane receipts for P00, verify hashes and interfaces, integrate central files, run label ce_ccp1_m00, and publish CE-CCP1-MILESTONE-M00.
