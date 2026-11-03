# Low-bit sparse values

The representation views live under `tensor/quantized`; quantize and
dequantize CUDA launchers live under `ops/quantized`.

`tensor::quantized::CsrView` borrows ordinary CSR row pointers and column
indices while storing values as 1, 2, 4, or 8-bit codes. It is a quantized
value encoding over CSR topology, not a second CSR index implementation.
`tensor::quantized::BlockedEllView` borrows block-column indices and row-packed
codes, with affine metadata supplied as a separate view. Both formats use the
same low-bit lane order: the first code occupies the lowest bits of each byte.

The operations layer provides the existing portable and SM70-specific
quantize/dequantize backends. The view, metadata policy, and kernel selection
remain separate so stored layout does not own execution strategy.

`interop::formats::AffineBlockedEllCodec` isolates the older CellShard archive
owner and metadata envelope. Its packed-value subregion uses the same
row-packed code order as `BlockedEllView`, but its owner and archive metadata
remain a distinct format. `make_runtime_view` borrows its buffers without
copying or rewriting archive bytes.
