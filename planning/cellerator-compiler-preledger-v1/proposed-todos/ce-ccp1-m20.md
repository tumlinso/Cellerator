# CE-CCP1-M20: Source language parser, AST, Sema, and execution-field semantics integrated

> **PROPOSED PRE-LEDGER RECORD.** This file is not managed by Todo Orchestrator and does not authorize implementation until the apply-ready plan is manually applied and the run is explicitly activated.

## Objective

Integrate and validate all P20 workstreams, freeze shared interfaces, and publish milestone M20.

## State

- Lifecycle: proposed / planned
- Execution: inactive
- Kind: `validation_task`
- Parallel policy: `exclusive_integration`

## Program role

Integration-lane-only task. It may repair temporary structural worktree breakage but may not absorb Part Two.

## Ownership

- exclusive: `include/Cellerator/compiler.hh`
- exclusive: `include/Cellerator/compiler/frontend`
- exclusive: `include/Cellerator/compiler/ast`
- exclusive: `include/Cellerator/compiler/sema`
- exclusive: `src/compiler/CMakeLists.txt`
- read: `.`
- forbidden: `.todo-orchestrator`
- shared lock: `ce-ccp1-grammar-registry`
- shared lock: `ce-ccp1-umbrella-headers`
- claim lock: `ce-ccp1-grammar-registry`
- claim lock: `ce-ccp1-umbrella-headers`

## Dependencies

- task CE-CCP1-C01-016 state done
- task CE-CCP1-C02-012 state done
- task CE-CCP1-C03-016 state done
- task CE-CCP1-C04-016 state done
- checkpoint CE-CCP1-MILESTONE-M10 state reached

## Interfaces

- None directly.

## Invariants

- `CE-CCP1-INV-CENTRAL-INTEGRATION`

## Completion contract

- required: All workstreams in P20 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P20 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Produced artifacts

- `docs/compiler/milestones/m20_receipt.md` (integration_receipt)

## Gates and resources

- gate `CE-CCP1-M20-INTEGRATION-GATE`: `ctest --test-dir build --output-on-failure -L ce_ccp1_m20`

## Completion criteria

- required: All workstreams in P20 are integrated
- required: Focused and milestone validation passes
- required: All interfaces published by P20 owners are frozen
- integration: Resolve source fragments only through source-linked receipts
- integration: Leave main coherent and buildable at the milestone boundary

## Next action

Collect isolated lane receipts for P20, verify hashes and interfaces, integrate central files, run label ce_ccp1_m20, and publish CE-CCP1-MILESTONE-M20.
