# CE-LIVE quantitative fixture outputs

Generate the local extracted fixture from the PBMC3K source under
`CELLERATOR_DATA_ROOT` (default `~/data/Cellerator`):

```bash
python tests/live/fixture/quantitative_relation_fixture.py extract \
  --manifest tests/live/fixture/pbmc3k_quantitative_v1.json \
  --output bench/ce_live/fixture/local/pbmc3k-r512-s7.npz
```

The `local/` directory is ignored. The committed manifest is sufficient to
reproduce and verify its source rows, CSR arrays, and both value generations.
