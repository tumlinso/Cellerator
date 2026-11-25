# Built-in gene catalogs

The mm10 manifest pins GENCODE M25 comprehensive annotation for the reference
chromosomes of GRCm38.p6. The two dictionaries select all 37 chrM gene features
and the 13 chrM features annotated as protein-coding in that release.
[Release and annotation download](https://www.gencodegenes.org/mouse/release_M25.html)

Stage `gencode.vM25.annotation.gtf.gz` under
`${CELLERATOR_DATA_ROOT}/reference/biology/` (default
`~/data/Cellerator/reference/biology/`) or pass its local path with
`-DCELLERATOR_ANNOTATION_FILE=/path/to/gencode.vM25.annotation.gtf.gz`.
From the repository root, build the optional catalog with:

```sh
cmake -S components/annotation -B build/annotation \
  -DCELLERATOR_ANNOTATION_CATALOG_MANIFEST="$PWD/components/annotation/catalogs/mm10_m25.cmake"
cmake --build build/annotation
ctest --test-dir build/annotation --output-on-failure
```

For a root Cellerator build, enable `CELLERATOR_ENABLE_ANNOTATION` and set the
same catalog options. The build checks SHA-256 before decompressing locally
with `gzip`. Generation uses the same annotation reader as runtime annotation
loading. The build writes `generated/gene_catalog.gen.inc` in the component's
build directory and compiles it into one translation unit.

Use `builtin_gene_catalog()` from `<Cellerator/biology/gene_catalog.hh>` or
`import cellerator.annotation;`. It returns the configured release and its
static arrays, or null when the manifest option is unset. The arrays can be
borrowed throughout the process lifetime. Pass `reference->catalog` to
`GeneVocabulary::from_catalog()` to assign a model vocabulary.

Each dictionary is a UTF-8 `.ids` file with one exact, versioned ID per line,
plus optional blank lines and comments beginning with `#`. The file stem names
the subset. Add the file to `catalog_subsets` in the manifest. Generation
preserves dictionary order and rejects unknown or repeated IDs. The catalog
itself is ordered by canonical ID, with compatible duplicate rows merged.

## Why this exists

The main use for these catalogs is to support reusable, precomputed packings
for Cellerator sparse support. A packing should carry its assembly, annotation
release, and exact canonical gene IDs in packing order. When binding an AnnData
dataset or another source, resolve its feature metadata and compare the ordered
IDs once. Matching axes can reuse the feature positions directly. If the same
IDs occur in a different order, prepare a column map once and reuse it across
batches. Resolve missing or ambiguous required IDs before binding.

Catalog rows, packing positions, dataset columns, and model token IDs each
have their own ordering. A packing that depends on observed sparse topology
also needs that topology validated. This is the intended binding contract.
The catalog currently supplies the reference metadata and named subsets.
