# Working on Cellerator preprocessing

Follow the root [repository work contract](../../AGENTS.md) and protected
[C++ style](../../coding_style.md). This component supplies
`import cellerator.preprocess;` and `Cellerator::preprocess`.

Use typed Cellerator views and caller-owned scratch and outputs. Preserve signed
dense strides, sparse orientation and index types, and observation and gene
ordering. Keep acquisition and AnnData I/O at the caller boundary.

See the [method contract](../../docs/development/preprocess/operations.md) and
[server validation guide](../../docs/development/single-cell-methods.md).
Preserve historical source identities and qualification records.
