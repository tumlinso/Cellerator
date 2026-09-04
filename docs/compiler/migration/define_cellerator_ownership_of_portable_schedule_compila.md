# Cellerator ownership of portable schedule compilation

Todo `CE-CCP1-A03-009` moves machine-independent schedule and ruleset
selection from the JBC schedule family at CellShard
`b9749ad3e5146a04f847533d8c6f1a54146aed20` into Cellerator Planning and
Realization IR. Portable commands, dependencies, bindings, transient-workspace
requirements, order transforms, barriers, publication intent, replay mode and
distributed logical certificates remain useful behavior.

Schedule identity is derived from the compiled ruleset, representative profile,
exact coverage, realization family and a portable target capability class. It
excludes file/object paths, runtime pointers, allocation addresses, lease
tokens, current device ordinals and concrete routes. CellShard may later bind
the immutable schedule to placement, data delivery, residency and transport;
those bindings do not alter compiler identity.

The deliberate change replaces CellShard-owned strong IDs with public
Cellerator identities and treats the old portable image as compatibility
evidence, not final wire doctrine. Gate `CE-CCP1-A03-009-GATE` validates every
identity component and scans the contract for concrete runtime/storage fields.
