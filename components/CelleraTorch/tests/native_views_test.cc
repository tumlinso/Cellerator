#include <CelleraTorch/native_views.hh>

#include <cuda_runtime.h>

#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

using namespace celleratorch::native_views;
namespace execution = cellerator::execution;

execution::axis_identity axis(std::uint32_t seed) {
    return {{seed, 1u}, {seed + 1u, 1u}, {seed + 2u, 1u},
        {seed + 3u, 1u}};
}

std::shared_ptr<const void> device_storage(std::size_t bytes, void **pointer) {
    void *data = nullptr;
    if (cudaMalloc(&data, bytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");
    *pointer = data;
    return std::shared_ptr<const void>(data, [](const void *allocation) {
        (void)cudaFree(const_cast<void *>(allocation));
    });
}

execution::dense_tensor_view dense(void *data, int device) {
    execution::dense_tensor_view view{};
    view.data = data;
    view.location = {execution::residency_kind::device, {}, device, 0u};
    view.value_type = execution::numeric_type::f32;
    view.rank = 2u;
    view.axes[0] = axis(1u);
    view.axes[1] = axis(11u);
    view.shape[0] = 3u;
    view.shape[1] = 4u;
    view.stride[0] = 4;
    view.stride[1] = 1;
    return view;
}

cellerator::native_parameter_descriptor parameter(void *data, int device) {
    cellerator::native_parameter_descriptor descriptor{};
    descriptor.storage.name = "relation_values";
    descriptor.storage.scalar_type = cellerator::parameter_scalar_type::f16;
    descriptor.storage.memory_space = cellerator::parameter_memory_space::device;
    descriptor.storage.device_ordinal = device;
    descriptor.storage.data = data;
    descriptor.storage.rank = 1u;
    descriptor.storage.shape[0] = 16;
    descriptor.storage.stride[0] = 1;
    descriptor.storage.writable = true;
    descriptor.kind = cellerator::native_parameter_kind::relation_values;
    descriptor.structure = {1u, 1u};
    descriptor.structure_epoch = {3u};
    descriptor.generation = {7u};
    descriptor.axes[0] = axis(21u);
    descriptor.axes[1] = axis(31u);
    descriptor.axis_count = 2u;
    return descriptor;
}

bool rejects(const std::function<void()> &operation) {
    try {
        operation();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 1;

    void *dense_data = nullptr;
    auto dense_owner = device_storage(12u * sizeof(float), &dense_data);
    auto dense_descriptor = dense(dense_data, device);
    native_view_requirements dense_requirements{};
    dense_requirements.expected_device_ordinal = device;
    dense_requirements.check_dtype = true;
    dense_requirements.expected_dtype = torch::kFloat32;
    auto dense_view = make_dense_tensor_view(
        dense_descriptor, native_storage_lease(dense_owner), dense_requirements);
    if (!dense_view.aliases(dense_data) || dense_view.tensor.dim() != 2
        || dense_view.tensor.size(0) != 3 || dense_view.tensor.size(1) != 4
        || dense_view.tensor.stride(0) != 4 || dense_view.tensor.stride(1) != 1
        || dense_view.tensor.device().index() != device
        || dense_view.metadata.axis_count != 2u)
        return 2;

    void *parameter_data = nullptr;
    auto parameter_owner = device_storage(16u * sizeof(std::uint16_t),
        &parameter_data);
    auto parameter_descriptor = parameter(parameter_data, device);
    native_view_requirements parameter_requirements{};
    parameter_requirements.expected_device_ordinal = device;
    parameter_requirements.check_dtype = true;
    parameter_requirements.expected_dtype = torch::kFloat16;
    parameter_requirements.access = native_view_access::read_write;
    auto parameter_view = make_parameter_tensor_view(parameter_descriptor,
        native_storage_lease(parameter_owner), parameter_requirements);
    if (!parameter_view.aliases(parameter_data)
        || parameter_view.tensor.numel() != 16
        || parameter_view.metadata.kind != native_view_kind::relation_values
        || parameter_view.metadata.generation.value != 7u
        || !parameter_view.metadata.writable
        || parameter_view.metadata.parameter_role
            != cellerator::parameter_role::learned
        || parameter_view.metadata.parameter_name != "relation_values")
        return 3;

    auto wrong_device = dense_requirements;
    wrong_device.expected_device_ordinal = device + 1;
    if (!rejects([&] { make_dense_tensor_view(dense_descriptor,
            native_storage_lease(dense_owner), wrong_device); })) return 4;

    auto invalid_dense = dense_descriptor;
    invalid_dense.location.device_ordinal = device + 1;
    if (!rejects([&] { make_dense_tensor_view(invalid_dense,
            native_storage_lease(dense_owner)); })) return 13;

    auto wrong_dtype = dense_requirements;
    wrong_dtype.expected_dtype = torch::kFloat16;
    if (!rejects([&] { make_dense_tensor_view(dense_descriptor,
            native_storage_lease(dense_owner), wrong_dtype); })) return 5;

    invalid_dense = dense_descriptor;
    invalid_dense.data = nullptr;
    if (!rejects([&] { make_dense_tensor_view(invalid_dense,
            native_storage_lease(dense_owner)); })) return 6;

    std::weak_ptr<const void> expired;
    {
        void *temporary = nullptr;
        auto temporary_owner = device_storage(sizeof(float), &temporary);
        expired = temporary_owner;
    }
    if (!rejects([&] { make_dense_tensor_view(dense_descriptor, expired); }))
        return 7;

    invalid_dense = dense_descriptor;
    invalid_dense.shape[0] = 0u;
    if (!rejects([&] { make_dense_tensor_view(invalid_dense,
            native_storage_lease(dense_owner)); })) return 8;

    invalid_dense = dense_descriptor;
    invalid_dense.stride[0] = 1;
    invalid_dense.stride[1] = 1;
    if (!rejects([&] { make_dense_tensor_view(invalid_dense,
            native_storage_lease(dense_owner)); })) return 9;

    auto invalid_parameter = parameter_descriptor;
    invalid_parameter.structure = {};
    if (!rejects([&] { make_parameter_tensor_view(invalid_parameter,
            native_storage_lease(parameter_owner), parameter_requirements); }))
        return 10;

    invalid_parameter = parameter_descriptor;
    invalid_parameter.axis_count = 1u;
    if (!rejects([&] { make_parameter_tensor_view(invalid_parameter,
            native_storage_lease(parameter_owner), parameter_requirements); }))
        return 14;

    invalid_parameter = parameter_descriptor;
    invalid_parameter.storage.writable = false;
    if (!rejects([&] { make_parameter_tensor_view(invalid_parameter,
            native_storage_lease(parameter_owner), parameter_requirements); }))
        return 11;

    void *leased_data = nullptr;
    auto leased_owner = device_storage(sizeof(float), &leased_data);
    auto leased_descriptor = dense(leased_data, device);
    leased_descriptor.rank = 1u;
    leased_descriptor.shape[0] = 1u;
    leased_descriptor.stride[0] = 1;
    auto lease_view = make_dense_tensor_view(leased_descriptor,
        native_storage_lease(leased_owner));
    std::weak_ptr<const void> lease_observer = leased_owner;
    leased_owner.reset();
    if (lease_observer.expired() || !lease_view.aliases(leased_data)) return 15;
    lease_view.tensor.reset();
    if (!lease_observer.expired()) return 16;

    std::cout << "CE-LIVE-40 native views passed\n";
    return 0;
}
