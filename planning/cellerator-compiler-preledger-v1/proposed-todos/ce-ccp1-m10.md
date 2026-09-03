# CE-CCP1-M10: Host-only build, driver, source pipeline, and C++ bridge integrated

> **PROPOSED PRE-LEDGER RECORD.** This file is not managed by Todo Orchestrator and does not authorize implementation until the apply-ready plan is manually applied and the run is explicitly activated.

## Objective

Integrate and validate all P10 workstreams, freeze shared interfaces, and publish milestone M10.

## State

- Lifecycle: proposed / planned
- Execution: inactive
- Kind: `validation_task`
- Parallel policy: `exclusive_integration`

## Program role

Integration-lane-only task. It may repair temporary structural worktree breakage but may not absorb Part Two.

## Ownership

- exclusive: `CMakeLists.txt`
- exclusive: `src/CMakeLists.txt`
- exclusive: `tools/CMakeLists.txt`
- exclusive: `cmake/compiler`
- exclusive: `include/Cellerator/compiler/build`
- read: `.`
- forbidden: `.todo-orchestrator`
- shared lock: `ce-ccp1-root-build`
- claim lock: `ce-ccp1-root-build`

## Dependencies

- task CE-CCP1-B01-012 state done
- task CE-CCP1-B02-014 state done
- task CE-CCP1-B03-015 state done
- task CE-CCP1-B04-014 state done
- checkpoint CE-CCP1-MILESTONE-M00 state reached

## Interfaces

- None directly.

## Invariants

- `CE-CCP1-INV-CENTRAL-INTEGRATION`

## Completion contract

- required: All workstreams in P10 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P10 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Produced artifacts

- `docs/compiler/milestones/m10_receipt.md` (integration_receipt)

## Gates and resources

- gate `CE-CCP1-M10-INTEGRATION-GATE`: `ctest --test-dir build --output-on-failure -L ce_ccp1_m10`

## Completion criteria

- required: All workstreams in P10 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P10 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Next action

Collect isolated lane receipts for P10, verify hashes and interfaces, integrate central files, run label ce_ccp1_m10, and publish CE-CCP1-MILESTONE-M10.
