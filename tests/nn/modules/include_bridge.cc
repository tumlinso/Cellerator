#include <Cellerator/nn.hh>

#include <cstddef>

// This ordinary-header translation unit shares real NN and ATen types with
// the importing and, optionally, NVCC-compiled translation units.
std::size_t header_owner_slots(const cellerator::nn::Parameters& owner) {
    return owner.size();
}

bool header_tensor_defined(const at::Tensor& value) {
    return value.defined();
}
