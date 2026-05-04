# `core/quantized`

CelleratorCore quantized format, metadata, packing, and unpacking helpers for
Volta `sm_70`.

## Design

- One Core-owned format and pack/decode library.
- One packed CSR layout for all quantized sparse paths.
- Policy metadata decides where scale and offset come from.

Current metadata policies:

- `per_gene_affine<Real>`
- `column_scale_row_offset<Real>`

## Hot-path rule

This substrate is custom-kernel, not library-backed. The limiting factors are
sparse HBM traffic and irregular per-row work, not Tensor Core math. Sparse
matmul and model-facing use of these payloads remain in `src/compute/`.

## Future learned quantization hook

Learned per-gene quantization plugs in through `per_gene_affine<Real>`. A later
model only needs to own/update the gene scale and offset arrays, then pass that
metadata into `make_matrix(...)`.
