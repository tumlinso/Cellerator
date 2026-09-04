# Frozen compiler-ownership rehoming contract

Todo `CE-CCP1-A03-014` freezes the complete Cellerator/CellShard compiler split
against `CE-CCP1-I02-JBC-MIGRATION-MANIFEST` v1, frozen with content hash
`66109e87b3aad1b71a01a15031ca3296f125e807cd50eaccfda5c3494571587e`,
and the Cellerator scope/architecture boundary.

| Before: CellShard JBC family | After: authoritative target | Owner |
|---|---|---|
| evidence and discovery | Cellerator compiler profiles/discovery | Cellerator |
| exact certification | Cellerator Planning IR validators | Cellerator |
| atom semantics | Cellerator atom IR; resident instances below | Cellerator / CellShard split |
| typed composition and grammar | Cellerator Planning IR extensions | Cellerator |
| basis selection | Cellerator planner | Cellerator |
| superatom promotion | optional Cellerator planner composition | Cellerator |
| partials | algebra/legality in Cellerator; bytes/recovery in CellShard | Split |
| global graph/program | Cellerator Semantic and Planning IR | Cellerator |
| portable schedule | Cellerator Planning and Realization IR | Cellerator |
| atom-store/materialization/runtime | CellShard concrete application | CellShard |
| legacy compiler includes/tests | versioned one-way adapters to public Cellerator APIs | Temporary |
| compiled ruleset delivery | immutable Cellerator export consumed later by CellShard | Part Two seam |

Every compiler-semantic subsystem has an owner. CellShard retains storage,
materialization, placement, residency, transport, leases, recovery and delivery,
but none decide biology or compiler semantics. Temporary adapters own nothing
and carry named retirement proofs. The Part Two seam contains immutable rules,
profile identity, exact coverage and realization requirements only; deep
CellShard integration and general JIT remain deferred.

The accompanying machine contract has twelve exhaustive rows and includes no
unowned state. Gate `CE-CCP1-A03-014-GATE` compiles all A03 contracts together,
checks each subsystem map and cross-contract invariant, confirms I02 is frozen,
and scans the receipt against Cellerator scope. Publication freezes
`CE-CCP1-I03-COMPILER-OWNERSHIP` v1 and reaches `CE-CCP1-CP-A03`.
