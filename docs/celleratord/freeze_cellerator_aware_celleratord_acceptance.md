# Cellerator-aware celleratord acceptance

The terminal I02 acceptance fixture represents an installed, multi-profile relation
project and freezes the baseline semantic query surface. It covers completion and
hover, profile state, Semantic and Planning IR, candidate costs, mutation generation
staleness, realization decomposition, and source-to-native navigation.

The focused integration gate links the independently implemented query providers into
one celleratord-facing process. Stable string snapshots prove that each query returns a
serializable result, while the LSP probe proves ordinary completion and biological
hover remain available. This is a tooling acceptance boundary; it does not introduce
Part Two JIT execution or transfer compiler ownership to CellShard.

The frozen public contract is
`include/Cellerator/compiler/tooling/cellerator_queries_v1.hh`. It names every
supported query independently of the server transport and exposes an acceptance
snapshot for installed profiles, LSP integration, and stable serialization.
