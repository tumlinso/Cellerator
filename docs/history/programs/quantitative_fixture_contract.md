# CE-LIVE quantitative fixture contract

CE-LIVE uses one checksum-pinned PBMC3K artifact as a **computational fixture**.
It is not evidence for a biological claim, and the fixture tooling is not a
general H5AD reader or a Cellerator runtime input path.

## Frozen source and axes

The authoritative manifest is
`data/manifests/ce_live/pbmc3k_quantitative_v1.json`. It fixes the local source
artifact by SHA-256, byte size, `/X` matrix path, legacy CSR encoding, stored
`float32` dtype, and observations-by-features orientation. The complete feature
axis and selected observation axis are bound by unambiguous length-prefixed
UTF-8 digests. Source row offsets are also committed explicitly.

The 512 observations use the existing `splitmix64-rank-v1` structural sampling
rule with seed 7: rank every source row by `splitmix64(seed xor row)`, retain the
512 smallest ranks, then restore ascending source-row order. This is exactly the
selection already used by the PBMC3K structure trace; CE-LIVE does not define a
second subset.

The extracted CSR contract preserves source edge order within each selected
row. It hashes canonical little-endian `uint64` row offsets, `uint32` feature
indices, and IEEE-754 `float32` values separately. Domain, order, geometry,
partition, structure, and value-generation identities are deterministic tagged
digests of those frozen inputs. These fixture identities are evidence inputs;
they do not introduce a second production identity ABI.

## Stored values and generations

The source-wide `/X/data` scan—not merely the selected rows—establishes that all
2,286,884 stored values are finite, non-negative, integral in their stored
`float32` representation, and range from 1 through 419. Their `float64` audit
sum is 6,390,631. Generation 1 uses the selected stored values through an
explicit `float32` identity cast. No normalization or inferred semantics are
attached to them.

Generation 2 keeps exactly the same CSR support and creates deterministic signed
`float32` stress values from source row, feature, and within-row edge ordinal.
It exists solely to catch stale-generation and numerical-path errors. It is not
normalization, transformation, imputation, or biological data.

## Reproduction and referees

Verify the local source and committed manifest with:

```bash
python scripts/ce_live_fixture.py verify \
  --source data/test/reference/pbmc3k_raw.h5ad \
  --manifest data/manifests/ce_live/pbmc3k_quantitative_v1.json
python scripts/ce_live_fixture.py smoke \
  --fixture tests/live/fixture/tiny_quantitative_fixture_v1.json
python -m unittest tests/live/fixture/test_ce_live_fixture.py
```

The tiny committed schema fixture is independently evaluated as row-oriented
CSR and as an edge-coordinate list for both forward and transpose products.
The representative 512-row arrays can be generated under the ignored
`bench/ce_live/fixture/local/` directory. Raw H5AD and derived large arrays stay
local; only their reproducibility contract is committed.

## Scientific limitation

No donor, sample, chemistry, species, normalization, biological comparison, or
scientific interpretation is asserted. The source metadata currently available
to CE-LIVE does not support those statements. Consumers may claim only that the
bytes, axes, support, values, selection, and deterministic fixture identities
match this contract.
