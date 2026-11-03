# PBMC ATAC replay format

`prepare_atac.cu` reads the official 10x Genomics ARC 1.0.0 filtered
feature-by-cell HDF5 matrix through CellShard's native 10x reader. It retains
only rows whose `feature_type` is exactly `Peaks`, in their supplied order.
Cell barcodes, peak IDs, names, and genomic intervals remain in the TSV
sidecars. The HDF5 source and preparation output carry SHA-256 identities.

The converter requires the exact pinned source digest and an explicit
40-character CellShard revision. It validates all sparse count values as
finite, nonnegative integers, rejects duplicate cell or peak IDs, and requires
every peak ID to retain a parseable `chromosome:start-end` interval. A bit is
set exactly when the stored count is positive. This is an assay observation;
an unset bit is not a biological exclusion.

`accessibility.cgatac` is little-endian. Its fixed 72-byte header is followed
by feature-major `uint32[peak, ceil(cells/32)]` words. Within a word, bit `j`
belongs to cell `32*word+j`; unused tail bits must be zero.

| Byte offset | Width | Value |
| ---: | ---: | --- |
| 0 | 8 | ASCII `CGATAC01` |
| 8 | 4 | Format version `1` |
| 12 | 4 | Endian marker `0x01020304` |
| 16 | 8 | Cell count |
| 24 | 8 | Peak count |
| 32 | 8 | Words per peak |
| 40 | 32 | Raw bytes of the source HDF5 SHA-256 |
| 72 | remaining | Feature-major packed observations |

For the pinned file, the dimensions are 12,016 cells and 63,751 peaks, so
there are 376 words per peak and 95,881,576 bytes including the header.

`cells.tsv` stores `cell_index, barcode`. `features.tsv` stores
`peak_index, source_feature_index, id, name, chromosome, start, end`. The
feature index preserves the original 10x feature row; the peak index gives its
compact bitplane position. `manifest.json` records the source URL and CC BY 4.0
attribution, source and preparation revisions, dimensions, payload FNV64,
SHA-256 of the replay and sidecars, identity-sidecar SHA-256, and wrapper
SHA-256. The wrapper digest covers the replay bytes followed by `cells.tsv`
then `features.tsv`; it excludes the manifest itself.

`verify_atac.cc` independently checks the header, source and CellShard
identities, checksums, tail padding, sidecar row counts and unique identities,
coordinate fields, and the zero-observation semantics.
