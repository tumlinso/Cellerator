# Feature-major tiled sparse training benchmark

The benchmark performs the same width-16 training step through six routes.
Each uses binary16 sparse execution values, FP32 weights and gradients, a
squared-error loss scaled by the output width, and SGD. FP32 sparse training
parameters are separate from the execution values. The benchmark checks
forward, both gradients, and updated masters and weights for three steps
against independent double-precision loops over the original CSR coordinates.

| Route | Forward | Weight gradient | Sparse gradient |
| --- | --- | --- | --- |
| `csr` | Native full-width CSR | Native transpose with atomic accumulation | Native CSR sampled dot |
| `csr-transpose` | Native full-width CSR | Prepared per-entry directory with shared sparse values | Native CSR sampled dot |
| `cusparse` | Prepared cuSPARSE CSR ALG1 | Prepared cuSPARSE transpose ALG1 | Prepared cuSPARSE SDDMM |
| `cusparse-alg2` | cuSPARSE CSR ALG2 | cuSPARSE transpose ALG2 | Prepared cuSPARSE SDDMM |
| `contiguous` | Feature-major tiles | Feature-to-tile directory | Feature-major sampled dot |
| `grouped` | Feature-major tiles with local row grouping | Feature-to-tile directory | Feature-major sampled dot |

cuSPARSE uses FP32 sparse operands expanded from the same binary16 bit values
used by the other routes. Its workspace and FP32 sparse plane count toward the
live footprint. The external refresh run alternates two CSR-ordered FP16 arrays.
The CSR routes can borrow the arrays directly. Tiled routes gather each source
into the shared packed plane. cuSPARSE expands the selected source into its
resident FP32 plane.

## Build and run

Configure a native CUDA build for `sm_70` with
`CELLERATOR_BUILD_TESTS=ON` and `CELLERATOR_BUILD_NATIVE_DATA_BENCHMARKS=ON`.
Build `celleratorFeatureMajorTilesTrainingTest` and
`celleratorFeatureMajorTilesBench`. Run the training test before the campaign
and use Compute Sanitizer `memcheck` and `synccheck` for direct CUDA validation.
Run GPU programs under the project and host execution rules for the session.
The campaign driver does not grant access to a device.

```sh
python3 bench/feature_major_tiles/run_campaign.py \
  --binary /path/to/celleratorFeatureMajorTilesBench \
  --traces bench/architecture_evidence/real_traces \
  --kda-root /path/to/kda \
  --expected-gpu GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --mode learn --output-dir /path/to/new-run
```

Run again with `--mode refresh` and `--mode frozen` for external input refresh
and fixed sparse input, respectively. `--case` selects one or more cases for a
small check. The two biological fixtures record original support with generated
numerical values. The other two fixtures cover low feature sharing and skew.
The optional `--case shared-tiles` control has exactly shared support in each
full tile and tests the high-occupancy limit of the representation.
For larger original-support batches, select `--case gse147520-4096` or
`--case gse147520-16384` and pass `--matrix-market CASE /path/to/support.mtx`
for each selected case. These inputs require NumPy and SciPy and a matching
`.rows.npy` sidecar containing the distinct source row IDs. The driver reads the
full 26,587-column support and rejects duplicate coordinates. Source, sidecar,
and derived CSR hashes are recorded independently.

## Interpret results

The driver checks the GPU UUID and SM version reported by each process. It
records the executable, driver, KDA selection code, and exact fixture hashes.
Each round varies the route order. After screening, the fastest conventional and
tiled routes for each workload are remeasured in fresh paired rounds at 64 steps.
The driver calls KDA's paired bootstrap confirmation with 4,000 resamples and
reports the 95% interval and a 5% complete-time reduction threshold.

The primary metric is host-wall mean milliseconds per step over each horizon
plus one-time preparation divided by the horizon. Setup includes host packing,
initial device upload, value packing, and vendor preparation where applicable.
For dataset training, the reuse horizon is the number of steps across all
epochs that share the prepared model structure. Charge layout preparation once
and report the per-example gather with the resident training step. The 64-step
comparison is one measured horizon. The refresh case alternates two
preassembled arrays on fixed CSR support. A dataset-level benchmark needs the
declared biological structure and the actual sequence of example gathers,
including batch order and any per-example activity.

CUDA event timings include complete-step execution and isolated forward,
backward, and sparse update measurements. Isolated measurements are diagnostic.
The byte metric covers explicitly provisioned live device arrays and cuSPARSE
workspace during training. It excludes CUDA context, library internal host
allocations, and validation-only buffers. Read `summary.txt` or `summary.json`
first. `run_config.json` and `results.json` retain workload identities, settings,
and raw samples.

## Shared regulatory structure with real PBMC3K cells

The dataset benchmark reads all 2,700 cells from the local raw PBMC3K assay.
It selects unambiguous measured TF and target symbols from the published
human CollecTRI table and excludes autoregulatory edges to avoid predicting a
gene directly from itself. The graph has 1,160 TF inputs, 6,330 target genes,
and 40,089 supported coefficients. Each observed count is transformed by
`log1p(10000 * count / cell library size)`, with library size computed over
the entire assay. The initial coefficient is its regulatory sign divided by
the retained target's in-degree. [CollecTRI archive](https://zenodo.org/records/8192729)

The four routes share the same graph, initial coefficients, ordered training
examples, and squared-error objective. They gather TF expression and measured
target expression into batches of up to 16 cells from one resident,
cell-major observation table. Each route prepares its sparse operator once.
Forward multiplies by the shared graph, and backward computes the input
gradient and the shared coefficient gradient on the declared edges. The
complete training measurement also includes SGD and casting from the FP32
master to the execution precision. All sparse values used in multiplication
are rounded to FP16, including the values expanded to FP32 for cuSPARSE.

Build `celleratorFeatureMajorDatasetTrainingBench`. Use
`prepare_dataset_training.py` with the local PBMC3K H5AD and
`CollecTRI_regulons.csv` to create the fixture and its checksum manifest.
`run_dataset_training.py` takes the fixture, manifest, executable and its
SHA-256, KDA checkout, and output path. It checks an FP64 reference on the
first and final batch of each route. The final batch masks its four padding
channels. The driver randomizes route order within each of seven paired
rounds, with five samples of three full epochs per route and round.

The primary gather, forward, and backward time uses one timer around the
whole measurement interval. We hold the sparse coefficients fixed for
that measurement and still compute both derivatives. A separate whole-run
measurement times the full training step with SGD. Phase timing and the CUDA
trace are diagnostic. Dataset reading and normalization are common offline
steps. Dataset upload and route preparation are reported separately.

### Results on V100

Times are medians in microseconds per batch of up to 16 real cells. The
one-time route setup includes preparation and initial value placement.
The common dataset and batch-buffer setup took about 13 ms.

| Route | Gather + forward + backward | Full step with SGD | Route setup (ms) |
| --- | ---: | ---: | ---: |
| Current tiled operators | 277.73 | 280.13 | 2.80 |
| Tiled with experimental fused backward | 47.99 | 50.37 | 2.77 |
| Native CSR | 67.09 | 69.59 | 0.58 |
| Prepared cuSPARSE CSR ALG2 + SDDMM | 49.55 | 51.85 | 4.81 |

An earlier CUDA trace of the same workload put about 267 µs into the two
tiled input-gradient kernels and 14 µs into the sampled coefficient
gradient. The fused candidate visits each packed feature record once.
It reuses the cotangent for both derivatives, writes each sampled
coefficient gradient to its packed slot, and atomically accumulates the
input gradient. It retains the same prepared biological structure and
packed execution values. The fused implementation is benchmark-only.
Floating-point summation order can vary across runs.

The paired speedup for gather, forward, and backward was 5.79× against the
current tiled operators (95% interval 5.75–5.83×), 1.40× against native CSR
(1.39–1.41×), and 1.032× against cuSPARSE (1.025–1.045×). All 28 runs passed
the FP64 oracle. Compute Sanitizer memcheck and synccheck each found zero
errors in the fused route. These are results for one graph, dataset, batch
width, and GPU. The execution batches consist of shuffled individual
cells. A neighbor-pooling experiment would keep one output per focal cell
and gather context from overlapping neighborhoods of other cells.

Measurements ran directly under the temporary host override after the
Scheduler submission was cancelled without an attempt or lease. The
V100-SXM2-16GB showed zero utilization and allocated memory before and after
the run. CUDA 12.9, driver `580.178.04`, worktree base `fe12934a`, and KDA
`277c79b8cdd1` were used. The CollecTRI MD5 matched the archive's
`cee4a3943c059e6dd8796ced6dec44f6`. Assay, regulatory CSV, and fixture
SHA-256 prefixes are `89a96f1beaa2`, `4473c9189dd5`, and `191572c67537`.
The final benchmark source and executable prefixes are `16e2785b2cfe`
and `a47f25bc1091`. Raw paired results, profiler summaries, and sanitizer
logs are under `/tmp/cellerator-dataset-training`.

### Overlap in a candidate neighbor graph

A structural probe used the same 2,700 normalized PBMC3K cells. It took the
1,000 most variable features of the resident assay table, computed 24
principal components, and selected exact Euclidean neighbors with self
excluded. For each graph, both placements use the same directed edges.
The locality order is the leaf order of Ward clustering in the same PCA
space. Each record is a unique source cell within a 32-focal-cell tile.

| Neighbors per focal cell | Records in shuffled order | Records in locality order | Edges per locality-ordered record |
| --- | ---: | ---: | ---: |
| 16 | 37,176 | 15,491 | 2.79 |
| 32 | 65,712 | 23,042 | 3.75 |

At 32 neighbors, 43.2% of edges in locality order belong to records shared
by at least eight focal cells in one tile. Those records can reuse source
feature loads in forward and combine gradient contributions in backward.
These counts are structural evidence. A timed comparison needs the same
neighborhood graph, focal-cell schedule, and pooling law on both routes,
plus the cost of gathering, reductions, and one-time preparation.

On the same 32-neighbor graph, a partial densification probe grouped the
locality order into 16 focal rows. Within each group it sorted source
columns by occupancy and emitted only complete 16-column dense panels
above the cutoff. All remaining edges stayed sparse.

| Minimum active focal rows per source | Edges in dense panels | Dense slots occupied | Total multiply terms relative to sparse |
| --- | ---: | ---: | ---: |
| 6 of 16 | 27.13% | 54.82% | 1.224× |
| 8 of 16 | 8.91% | 66.82% | 1.044× |

The term count includes dense zero slots and the sparse remainder. A
32-row variant that densified every source used by eight or more focal
cells and padded each panel to a multiple of 16 columns covered 43.18% of
edges at 1.949× total multiply terms. These are layout estimates. Compute
pooling normalization over the full neighborhood, and restrict any
trainable edge gradients to the original support.

## V100 comparison, 2026-11-21

Direct, serial runs on Tesla V100-SXM2-16GB
(`GPU-21131915-1488-23af-38dd-1743ae1f5cc8`), CUDA 12.9, driver
`580.178.04`, with idle-device observations before each campaign. Every
measured route passed the independent FP64 oracle. The initial tile-serial
traversal took 6.400 ms per PBMC3K step at 64 uses. Sharding CUDA work across
records and replacing host entry sorting with tile-mask preparation reduced
the revised measurement to 0.526 ms. The corresponding GSE147520 measurement
fell from 9.803 ms to 0.407 ms. These are medians from separate campaigns.

These timings train dense weights and sparse master values and include
preparation amortized over 64 uses. Times are milliseconds per step.

| Support | Revised tiled | Fastest conventional | Fresh paired tiled time / conventional |
| --- | ---: | ---: | ---: |
| PBMC3K, 512 rows | 0.526 | 0.284, cuSPARSE ALG2 | 1.885× |
| GSE147520, 256 rows | 0.407 | 0.270, cuSPARSE ALG2 | 1.544× |
| GSE147520, 4,096 rows | 4.978 | 1.747, cuSPARSE ALG2 | 2.981× |
| GSE147520, 16,384 rows | 31.244 | 7.394, native CSR | 4.205× |
| Low sharing, 2,049 rows | 0.140 | 0.053, native CSR | 2.735× |
| Skewed, 2,049 rows | 0.253 | 0.086, native CSR | 2.888× |
| Fully shared tiles, 8,193 rows | 0.171 | 0.115, native CSR | 1.468× |

The all-column 4,096- and 16,384-row runs contain 5,131,947 and 27,553,941
stored coordinates. They use distinct selections of real source rows, so
they do not isolate batch size from support density.

| GSE147520 route | Forward ms | Weight gradient ms | Sparse gradient ms | Setup ms | Live arrays MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 rows, contiguous tiles | 0.407 | 0.853 | 0.547 | 196.3 | 117.2 |
| 4,096 rows, cuSPARSE ALG2 | 0.507 | 0.761 | 0.162 | 12.9 | 105.3 |
| 16,384 rows, contiguous tiles | 8.143 | 3.898 | 2.736 | 1,022.9 | 593.5 |
| 16,384 rows, native CSR | 1.956 | 1.839 | 2.468 | 41.2 | 479.3 |

The 16,384-row table uses the earlier dispatch, which returned to one warp per
tile at 512 tiles. Two separate AB/BA order pairs tested the installed rule
that also checks the number of records per tile. Forward fell from
8.13–8.15 ms to 2.01 ms, and the resident step fell from 15.25–15.47 ms to
9.14–9.15 ms. At 64 uses with preparation, the revised pairs took
29.32–29.49 ms per step. Native CSR took 6.75 ms per resident step in the
earlier paired campaign. The dispatch check used executable SHA-256 prefix
`fc2273f9a553` and saved its raw results under
`/tmp/cellerator-feature-tiles-density.dyc7s07k`.

PBMC3K's prepared per-entry transpose alone took 0.053 ms, while tiled
transpose took 0.120 ms.

For the two original small traces, the best tiled route took 1.766× and
1.432× the fastest conventional time with external refresh, and 1.894×
and 1.504× with fixed sparse values. The tiled FP16 gather took about
5.25 µs on each trace. cuSPARSE's update to its expanded plane took about
4.61 µs. Native CSR borrowed the incoming FP16 array without a refresh
launch. Both modes updated dense weights and evaluated both gradients.

The measured cases leave the tiled format available for explicit use. None
clears the complete-step promotion threshold.

The original support trace SHA-256 prefixes are `e8ddfb1e439a` for PBMC3K,
`4f00eb8d42e5` for small GSE147520, and `017255c36f27` and `b9e365a0ec72`
for the 4,096- and 16,384-row Matrix Market sources. The experiments start
from worktree base `fe12934a` and use KDA `277c79b8cdd1`. Initial, revised,
and phase-diagnostics executable SHA-256 prefixes are `39f97dda4c99`,
`84c82a343a63`, and `cf559d8c33f3`. Initial and revised task source archive
prefixes are `81f0b77fed4b` and `90acb6131bce`. Source archives, executable
hashes, build commands, paired samples, and confidence intervals are in
the local reports
`/tmp/cellerator-feature-tiles-campaign.ER9kkF` and
`/tmp/cellerator-feature-tiles-revised.GjS6hc`.

## Partial packing observations

For each contiguous group of 32 rows, count the number of stored coordinates
at each feature. The table shows what a structural cutoff of eight occupied
rows per feature record would select. Every other stored coordinate would
still participate through the residual CSR path.

| Original support | Entries selected | Records selected | Mean occupied rows per selected record |
| --- | ---: | ---: | ---: |
| PBMC3K, 512 rows | 47.88% | 12.06% | 15.45 |
| GSE147520, 256 rows | 53.98% | 18.18% | 14.20 |
| GSE147520, 4,096 rows | 48.78% | 13.94% | 14.57 |
| GSE147520, 16,384 rows | 55.98% | 19.25% | 14.43 |

The cutoff concentrates tiled records on reusable feature loads. The counts
measure structural coverage only. A training comparison would also charge
the support scan, emission of both layouts, value mapping and refresh, and
the reductions of overlapping outputs and dense weight gradients.
