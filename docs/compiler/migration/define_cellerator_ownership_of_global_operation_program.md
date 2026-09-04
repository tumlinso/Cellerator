# Cellerator ownership of global operation/program IR

Todo `CE-CCP1-A03-008` recasts the CellShard JBC graph family at commit
`b9749ad3e5146a04f847533d8c6f1a54146aed20` as Cellerator program-level
Semantic and Planning IR. The preserved behaviors include operation providers,
typed nodes and ports, access effects, atom dependencies and flows, graph
families/recipes, partial-result trees, rewrite descriptors, physical
realization requirements and portable serialization.

The thin-waist record has explicit stable identities for the entity, biological
domain and order, an entity kind (field, operation, atom flow or profile
family), and a structure generation. It contains no path, file descriptor,
object key, chunk address, lease, resident pointer, device ordinal, transport
route or CellShard include. Ordinary CellShard APIs consume compiled program
artifacts later; they are absent from compiler meaning.

Current JBC graph algorithms and tests remain migration evidence. The target
uses Cellerator fields and operation contracts rather than CellShard strong-ID
types, and physical realization becomes compiler requirements rather than a
resident instance. Gate `CE-CCP1-A03-008-GATE` checks all four entity kinds,
rejects unqualified biological identity/generation, and scans the public
contract for storage vocabulary and CellShard dependencies.
