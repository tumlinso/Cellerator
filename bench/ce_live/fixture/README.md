# CE-LIVE quantitative fixture outputs

Large extracted arrays are local evidence, not repository or runtime formats.
Generate the representative fixture with:

```bash
python scripts/ce_live_fixture.py extract \
  --source data/test/reference/pbmc3k_raw.h5ad \
  --manifest data/manifests/ce_live/pbmc3k_quantitative_v1.json \
  --output bench/ce_live/fixture/local/pbmc3k-r512-s7.npz
```

The `local/` directory is ignored. The committed manifest is sufficient to
reproduce and verify its source rows, CSR arrays, and both value generations.
