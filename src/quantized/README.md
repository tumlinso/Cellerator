# `quantized`

Single CUDA-first quantized CSR backend for Volta `sm_70`.

## Design

- One storage and kernel library.
- One packed CSR layout for all quantized sparse paths.
- Policy metadata decides where scale and offset come from.

Current metadata policies:

- `per_gene_affine<Real>`
- `column_scale_row_offset<Real>`

## Hot-path rule

This backend is custom-kernel, not library-backed. The limiting factors are sparse
HBM traffic and irregular per-row work, not Tensor Core math. The main throughput
win is keeping one pack/unpack/kernel path and compiling metadata differences away
through templates instead of runtime branching.

## Future learned quantization hook

Learned per-gene quantization plugs in through `per_gene_affine<Real>`. A later
model only needs to own/update the gene scale and offset arrays, then pass that
metadata into `make_matrix(...)`.
