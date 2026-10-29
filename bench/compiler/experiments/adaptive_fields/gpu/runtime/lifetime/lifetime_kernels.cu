// Tiny executable used only to exercise module ownership and stream lifetime.
// The scalar is rebound on every launch; it is not embedded in compiled code.
extern "C" __global__ void field_forward(const float* input, float* output,
                                          int count, float scale) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) output[index] = input[index] * scale;
}

extern "C" __global__ void field_backward(const float* upstream,
                                           float* gradient, int count,
                                           float scale) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) gradient[index] = upstream[index] * scale;
}
