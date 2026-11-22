import cellerator.nn;

static_assert(sizeof(at::Tensor) > 0);
static_assert(sizeof(at::Device) > 0);
static_assert(sizeof(cellerator::memory::const_array_view<int>) > 0);
static_assert(sizeof(cellerator::memory::matrix_view<float>) > 0);

bool module_only_surface_works() {
    cellerator::nn::Parameters parameters(
        at::Device(at::DeviceType::CPU), at::ScalarType::Float);
    const at::Tensor undefined_tensor;
    if (undefined_tensor.defined()) {
        return false;
    }
    return parameters.size() == 0;
}
