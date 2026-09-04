# Pointer-plus-count profile ingestion v1

The C ABI accepts four typed pointer-plus-count spans for relation, support,
value, and trace observations and forwards them synchronously to a caller sink.
It owns no storage, performs no allocation, and names no HDF5, AnnData, dataset,
or workflow file format. Both C and C++ producers consume the same bounded ABI.
