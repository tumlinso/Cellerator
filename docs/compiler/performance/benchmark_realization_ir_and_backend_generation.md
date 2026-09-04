# Realization IR and backend-generation benchmark v1

CPU, NVCC, Clang CUDA, and direct PTX (where installed and legal) report
realization, projection/packing planning, stage construction, generated-source
bytes, downstream compile time, ptxas registers/shared/local memory, object and
fatbinary bytes, and provenance bytes. Missing paths remain explicitly
unavailable and are never replaced. All available runs use the benchmark mutex;
GPU execution is outside this compiler-stage measurement.
