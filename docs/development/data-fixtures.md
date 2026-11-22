# Local test and benchmark data

Dataset-backed test and benchmark tools read local payloads from
`CELLERATOR_DATA_ROOT`. The default is `~/data/Cellerator`. Set the variable to
an absolute path to use a different fixture store. An explicit fixture argument
such as `--source` takes precedence for that input.

```sh
export CELLERATOR_DATA_ROOT="$HOME/data/Cellerator"
```

Paths in the fixture store are relative to this root:

| Path | Contents |
| --- | --- |
| `data/test/reference/` | Long-lived inputs such as `pbmc3k_raw.h5ad`, `pbmc3k_raw.csh5`, and the GSE147520 H5AD files. |
| `data/test/generated/csh5/` | Converted CSH5 outputs. |
| `data/test/generated/cache/` | Warmed cache trees and CSPACK files. |
| `data/test/generated/blocked_ell_optimization/` | Matrix subsets and related metadata for the blocked ELL study. |
| `data/test/generated/logs/` and `data/test/generated/tmp/` | Local diagnostics and scratch space. |

Preserve source fixture names and the source checksums recorded in the
manifests. Keep existing datasets and generated files in place. When a recipe
needs a new fixture output directory, create it under `CELLERATOR_DATA_ROOT`.

The [architecture benchmark manifests](../../bench/architecture_evidence/manifests/)
describe workload inputs and smoke traces. The
[quantitative PBMC3K manifest](../../tests/live/fixture/pbmc3k_quantitative_v1.json)
pins the source and the expected extracted fixture. Verify it from the checkout
root with:

```sh
python3 tests/live/fixture/quantitative_relation_fixture.py verify \
  --manifest tests/live/fixture/pbmc3k_quantitative_v1.json
```
