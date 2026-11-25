# Working on Cellerator RNA velocity

Follow the root [repository work contract](../../AGENTS.md) and protected
[C++ style](../../coding_style.md). This component supplies
`import cellerator.velocity;` and `Cellerator::velocity`.

Use typed Cellerator views and caller-owned scratch and outputs. Preserve signed
dense strides, sparse orientation and index types, and observation and gene
ordering. Keep acquisition and AnnData I/O at the caller boundary.

See the [method contract](../../docs/development/velocity/api.md) and
[server validation guide](../../docs/development/single-cell-methods.md).
Preserve historical source identities and qualification records.
