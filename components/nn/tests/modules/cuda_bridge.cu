#include <Cellerator/nn.hh>

#include <cstddef>

// NVCC sees only the established header contract. This host function reads
// the same C++ object and ATen tensor passed by the importing Clang TU; it
// performs no device launch.
std::size_t cuda_header_owner_slots(const cellerator::nn::Parameters& owner) {
    return owner.size();
}

at::Device cuda_header_parameter_device(
    const cellerator::nn::Parameters& owner) {
    return owner.device();
}

bool cuda_header_tensor_defined(const at::Tensor& value) {
    return value.defined();
}
