# Cellerator documentation

The documentation describes Cellerator as an early high-performance PyTorch
extension for cellular model building. PyTorch remains the general tensor,
autograd, optimizer, and standard-layer foundation. Cellerator adds reusable
operations and execution semantics shaped by cellular structure. The
[architecture](architecture.qmd) is the target ownership model; the
[implementation record](current_implementation.qmd) describes dated current
capabilities and evidence.

## Reader route

1. [Book entry](index.qmd) and [getting started](getting_started.qmd)
2. [Scope](scope.qmd) and [architecture](architecture.qmd)
3. [Model building](model_building.qmd), [operations](operations.qmd), and
   [layout/execution](layout_execution.qmd)
4. [Python bindings](development/python-bindings.md)
5. [Developer reference](developer_reference.qmd) and
   [agent-facing source routing](development/routing.md)
6. [Experiments and history](experiments_and_history.qmd)

The current book navigation is in `_quarto.yml`. Subsystem pages remain
accessible under their subsystem and history part, but they are not a
substitute for the current implementation record or the exact source map.

## Evidence and change policy

Keep current source placement, target architecture, experimental status, and
historical evidence distinct. The generated source map and routing rules are
maintained by `tools/maintenance/source_map/`; live task status belongs to
Project Control. Do not duplicate that status in multiple documentation
tables. Preserve source-bound reports and snapshot bytes; add current
interpretation in a dated summary or wrapper instead of rewriting frozen
records.

Keep biological identity, structure epochs, activity, values, physical layout,
and runtime bindings distinct. Claims about correctness, scientific validity,
package usability, and performance need separate evidence. Update the
relevant contract and its consumers when code changes; a target path in an
architecture diagram is not an implemented API.
