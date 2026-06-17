# Cellerator Components

This directory holds staging components for Cellerator surfaces that are being
split away from the native base math library.

Components here should depend on Cellerator primitives instead of redefining
them. A component belongs here when it is still close enough to migrate with the
Cellerator checkout but is not part of Cellerator sensu stricto.

Current components:

- `CellPack/`: static semantic sparse-packing layout compiler substrate.
- `CelleraTorch/`: Torch and libtorch integration boundary.
