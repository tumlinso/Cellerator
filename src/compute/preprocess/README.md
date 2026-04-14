# `compute/preprocess`

Sparse scRNA preprocessing operators for row-sharded sparse inputs.

This area owns:

- row-local cell metrics
- in-place normalization transforms
- gene metric accumulation
- gene filter mask construction
- per-device and fleet workspaces

## Backend Contract

The preprocess backend split is explicit in `backend_policy.cuh`.

Current operator classification:

| Operator | `blocked_ell_view` | `compressed_view` | Notes |
| --- | --- | --- | --- |
| `cell_metrics` | custom kernel | custom kernel | QC math is row-local and does not map cleanly to cuSPARSE |
| `normalize_log1p` | custom kernel | custom kernel | in-place transform and row masks are custom work |
| `gene_metrics` | custom kernel | cuSPARSE transpose `SpMV` | CSR remains the better temporary analysis layout for this reduction path today |
| `gene_filter_mask` | custom kernel | custom kernel | thresholding over aggregate feature stats |

Repository intent:

- Blocked-ELL is the native persisted and runtime execution layout.
- cuSPARSE should be used aggressively when the operation is genuinely sparse linear algebra on a layout it accelerates well.
- Preprocess should stay honest about where CSR is still the better temporary analysis layout instead of pretending every stage is Blocked-ELL-native or cuSPARSE-backed.

On V100 this means:

- keep Blocked-ELL-native custom kernels for row-local QC and transforms
- keep the CSR transpose-reduction path where it still gives the cleanest cuSPARSE mapping
- replace CSR temporary analysis paths only when a real Blocked-ELL library-backed or clearly faster custom path is proven
