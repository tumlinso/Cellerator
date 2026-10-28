// One tiny generated region shared byte-for-byte by NVCC and NVRTC.
// This is a compiler-path probe, not a proposed sparse or softmax kernel.
extern "C" __global__ void cellerator_region(
    const long long* row_offsets,
    const long long* columns,
    const float* values,
    const float* x,
    float* row_scratch,
    float* output,
    int rows,
    const long long* segment_offsets,
    int segments) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (long long edge = row_offsets[row]; edge < row_offsets[row + 1]; ++edge)
      sum += values[edge] * x[columns[edge]];
    row_scratch[row] = sum;
  }
  for (int segment = 0; segment < segments; ++segment) {
    const long long begin = segment_offsets[segment];
    const long long end = segment_offsets[segment + 1];
    if (begin == end) continue;
    float maximum = row_scratch[begin];
    for (long long row = begin + 1; row < end; ++row)
      if (row_scratch[row] > maximum) maximum = row_scratch[row];
    float denominator = 0.0f;
    for (long long row = begin; row < end; ++row)
      denominator += expf(row_scratch[row] - maximum);
    for (long long row = begin; row < end; ++row)
      output[row] = expf(row_scratch[row] - maximum) / denominator;
  }
}
