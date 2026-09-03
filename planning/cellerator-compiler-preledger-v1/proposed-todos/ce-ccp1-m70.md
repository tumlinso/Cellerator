# CE-CCP1-M70: Cross-TU/LTO, libCellerator, standard library, and installable SDK integrated

> **PROPOSED PRE-LEDGER RECORD.** This file is not managed by Todo Orchestrator and does not authorize implementation until the apply-ready plan is manually applied and the run is explicitly activated.

## Objective

Integrate and validate all P70 workstreams, freeze shared interfaces, and publish milestone M70.

## State

- Lifecycle: proposed / planned
- Execution: inactive
- Kind: `validation_task`
- Parallel policy: `exclusive_integration`

## Program role

Integration-lane-only task. It may repair temporary structural worktree breakage but may not absorb Part Two.

## Ownership

- exclusive: `CMakeLists.txt`
- exclusive: `cmake/package`
- exclusive: `include/Cellerator`
- exclusive: `stdlib`
- exclusive: `profiles/reference`
- exclusive: `tools/CMakeLists.txt`
- read: `.`
- forbidden: `.todo-orchestrator`
- shared lock: `ce-ccp1-package-exports`
- shared lock: `ce-ccp1-stdlib-manifest`
- shared lock: `ce-ccp1-umbrella-headers`
- claim lock: `ce-ccp1-package-exports`
- claim lock: `ce-ccp1-stdlib-manifest`
- claim lock: `ce-ccp1-umbrella-headers`

## Dependencies

- task CE-CCP1-H01-016 state done
- task CE-CCP1-H02-016 state done
- task CE-CCP1-H03-018 state done
- checkpoint CE-CCP1-MILESTONE-M60 state reached

## Interfaces

- None directly.

## Invariants

- `CE-CCP1-INV-CENTRAL-INTEGRATION`

## Completion contract

- required: All workstreams in P70 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P70 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Produced artifacts

- `docs/compiler/milestones/m70_receipt.md` (integration_receipt)

## Gates and resources

- gate `CE-CCP1-M70-INTEGRATION-GATE`: `ctest --test-dir build --output-on-failure -L ce_ccp1_m70`

## Completion criteria

- required: All workstreams in P70 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P70 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Next action

Collect isolated lane receipts for P70, verify hashes and interfaces, integrate central files, run label ce_ccp1_m70, and publish CE-CCP1-MILESTONE-M70.
