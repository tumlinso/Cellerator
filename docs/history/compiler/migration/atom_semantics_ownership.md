# Cellerator ownership of atom semantics

Todo `CE-CCP1-A03-003` retains **atom** as the compiler term while assigning
each state exactly one owner. Source review is pinned to CellShard
`b9749ad3e5146a04f847533d8c6f1a54146aed20`, especially `atom/level_v1.hh`,
`species_v1.hh`, the atom planes, certification, basis, superatom, partial, and
schedule contracts.

Candidate, certified, basis, super, physical, logical-replica and partial atom
semantics belong to Cellerator compiler levels. Candidate remains uncertain;
certified adds exact proof; basis is planner selection; super is an optional
exact derivation; physical describes realization requirements; replica states
logical ownership/consistency; partial states algebra and persistence legality.
Resident atom instances belong to CellShard because residency is concrete
materialization, placement, encoding, leasing and storage—not biological or
compiler identity.

The eight-row header contract is exhaustive and preserves current JBC behavior
without retaining the old monotonic `resident` level as compiler authority.
Species remain provider-qualified and extensible. Physical views and replica
requirements stay compiler semantics even when CellShard later instantiates
them. No state is deleted; the deliberate change is an explicit split between
logical compiler state and concrete application instances.

Gate `CE-CCP1-A03-003-GATE` checks all eight required states occur once, only
resident is CellShard-owned, and the pinned source still exposes the historical
level, species, partial, superatom and physical/replica behavior being mapped.
