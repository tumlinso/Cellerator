# Sparse spatial neighborhoods

`SpatialIndex` maintains a dynamic reference cell list for stable entity IDs in
one, two, or three Euclidean dimensions. Radius queries inspect occupied nearby
bins and exact-filter candidates. Moving an entity updates only its old/new
bins. Its coordinates, IDs, and returned neighbor relations stay distinct from
cell state and values.

```python
from cellerator.spatial import SpatialIndex, diffuse

positions = SpatialIndex({"a": (0.0, 0.0), "b": (0.2, 0.0)}, search_radius=0.5)
neighbors = positions.query("a")
next_field = diffuse(positions, {"a": 1.0, "b": 0.0}, dt=0.1, diffusivity=0.2)
```

`interaction_graph` builds a sparse snapshot from the current position map.
Rebuild it after movement when the model needs the new neighborhood. The graph
operation and optional gate are supplied by the model. `diffuse` demonstrates
an explicit conservative exchange over undirected neighbor pairs; callers
choose units, weights, radius and stable time step. It is a graph update, not a
continuum diffusion solver or a biological signaling assumption.

Current limitations: uniform grid, exact Euclidean radius queries in at most
three dimensions, one configured maximum radius, no periodic boundaries, and
Python reference execution. Native/CUDA acceleration and larger geometry
families need separate contracts and measurements.
