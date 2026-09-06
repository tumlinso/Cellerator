# Cellerator source units

The existing umbrella is now `library/cellerator/cellerator.cell`; its pragma,
comment and six core includes are preserved byte for byte from the I01 inventory.
No direct source references required changes. Ordinary native `.hh` headers remain.

`.cell` is the intended self-describing, importable source format. Extension-driven
dialect activation, semantic imports, cross-unit optimization and complete `.cell`
execution remain future work. This conversion supplies none of those capabilities
and does not expand the source library or installed SDK. Historical planning and
review references to `.ceh` remain historical evidence.
