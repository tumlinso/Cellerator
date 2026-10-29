# Generation and staleness queries

This page specifies query semantics; it is not independent evidence that the current server exposes the described query. Check the [current implementation boundary](../current_implementation.qmd) and [source routing guide](../development/routing.md) for current support.

Queries distinguish structure, value, support, and order generations and report the exact cached candidates, plans, projections, or value bindings invalidated by a statement.
