# `ingest`

Source-format ingest and series conversion live here.

This area owns MTX parsing, partition planning, manifest-driven series ingest,
and the shared feature/barcode/metadata tables needed to emit `CellShard`
series files. Keep parsing, staging, and file-format decisions explicit here.
