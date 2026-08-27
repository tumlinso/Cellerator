#include <CelleraTorch/native_views.hh>

#include <Cellerator/execution/validation.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celleratorch::native_views {
namespace {

using cellerator::execution::biological_operand_max_axes;

[[noreturn]] void reject(const char *message) {
    throw std::invalid_argument(message);
}

torch::ScalarType torch_type(cellerator::execution::numeric_type type) {
    using type_t = cellerator::execution::numeric_type;
    switch (type) {
    case type_t::u8: return torch::kUInt8;
    case type_t::i32: return torch::kInt32;
    case type_t::f16: return torch::kFloat16;
    case type_t::bf16: return torch::kBFloat16;
    case type_t::f32: return torch::kFloat32;
    case type_t::f64: return torch::kFloat64;
    default: reject("native dense dtype is not supported by CelleraTorch views");
    }
}

torch::ScalarType torch_type(cellerator::parameter_scalar_type type) {
    using type_t = cellerator::parameter_scalar_type;
    switch (type) {
    case type_t::u8: return torch::kUInt8;
    case type_t::i32: return torch::kInt32;
    case type_t::i64: return torch::kInt64;
    case type_t::f16: return torch::kFloat16;
    case type_t::bf16: return torch::kBFloat16;
    case type_t::f32: return torch::kFloat32;
    case type_t::f64: return torch::kFloat64;
    default: reject("native parameter dtype is not supported by CelleraTorch views");
    }
}

std::shared_ptr<const void> acquire(native_storage_lease lifetime) {
    std::shared_ptr<const void> owner = lifetime.lock();
    if (!owner) reject("native storage lifetime is empty or expired");
    return owner;
}

void validate_device_pointer(const void *data, int device_ordinal) {
    if (data == nullptr) reject("native storage pointer must be non-null");
    if (device_ordinal < 0) reject("native storage requires a CUDA device ordinal");

    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, data);
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        reject("native storage is not a CUDA allocation");
    }
    if (attributes.type != cudaMemoryTypeDevice)
        reject("native storage must be device-resident");
    if (attributes.device != device_ordinal)
        reject("native storage device does not match its descriptor");
}

template<typename Shape, typename Stride>
void validate_layout(const Shape *shape, const Stride *stride, std::uint8_t rank) {
    if (rank == 0u || rank > biological_operand_max_axes)
        reject("native tensor rank is unsupported");

    std::array<std::uint8_t, biological_operand_max_axes> order{};
    for (std::uint8_t axis = 0u; axis < rank; ++axis) {
        if (shape[axis] <= 0) reject("native tensor dimensions must be positive");
        if (stride[axis] <= 0) reject("native tensor strides must be positive");
        if (static_cast<std::uint64_t>(shape[axis])
            > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::overflow_error("native tensor dimension exceeds Torch int64 range");
        order[axis] = axis;
    }
    std::sort(order.begin(), order.begin() + rank,
        [stride](std::uint8_t lhs, std::uint8_t rhs) {
            return stride[lhs] < stride[rhs];
        });

    std::uint64_t required_span = 1u;
    for (std::uint8_t position = 0u; position < rank; ++position) {
        const std::uint8_t axis = order[position];
        const std::uint64_t axis_stride = static_cast<std::uint64_t>(stride[axis]);
        const std::uint64_t axis_size = static_cast<std::uint64_t>(shape[axis]);
        if (axis_stride < required_span)
            reject("native tensor layout overlaps or broadcasts storage");
        if (axis_size - 1u > (std::numeric_limits<std::uint64_t>::max()
                - required_span) / axis_stride)
            throw std::overflow_error("native tensor strided span overflows");
        required_span += (axis_size - 1u) * axis_stride;
    }
}

template<typename Shape, typename Stride>
torch::Tensor alias_tensor(void *data,
    const Shape *shape,
    const Stride *stride,
    std::uint8_t rank,
    int device_ordinal,
    torch::ScalarType dtype,
    std::shared_ptr<const void> lifetime) {
    std::vector<std::int64_t> sizes(rank);
    std::vector<std::int64_t> strides(rank);
    for (std::uint8_t axis = 0u; axis < rank; ++axis) {
        sizes[axis] = static_cast<std::int64_t>(shape[axis]);
        strides[axis] = static_cast<std::int64_t>(stride[axis]);
    }
    auto deleter = [owner = std::move(lifetime)](void *) mutable {
        owner.reset();
    };
    return torch::from_blob(data, sizes, strides, std::move(deleter),
        torch::TensorOptions()
            .dtype(dtype)
            .device(torch::Device(torch::kCUDA, device_ordinal))
            .requires_grad(false));
}

void validate_requirements(const native_view_requirements &requirements,
    int device_ordinal,
    torch::ScalarType dtype,
    bool writable) {
    if (requirements.expected_device_ordinal >= 0
        && requirements.expected_device_ordinal != device_ordinal)
        reject("native view device does not satisfy the requested device");
    if (requirements.check_dtype && requirements.expected_dtype != dtype)
        reject("native view dtype does not satisfy the requested dtype");
    if (requirements.access == native_view_access::read_write && !writable)
        reject("native storage is not writable");
}

native_view_kind parameter_kind(cellerator::native_parameter_kind kind) {
    switch (kind) {
    case cellerator::native_parameter_kind::relation_values:
        return native_view_kind::relation_values;
    case cellerator::native_parameter_kind::dense_bias:
        return native_view_kind::dense_bias;
    default: reject("native parameter kind is unsupported");
    }
}

void validate_parameter_kind_contract(
    const cellerator::native_parameter_descriptor &native) {
    switch (native.kind) {
    case cellerator::native_parameter_kind::relation_values:
        if (native.axis_count != 2u
            || native.storage.scalar_type != cellerator::parameter_scalar_type::f16)
            reject("native relation-value parameter metadata is incompatible");
        return;
    case cellerator::native_parameter_kind::dense_bias:
        if (native.axis_count != 1u
            || native.storage.scalar_type != cellerator::parameter_scalar_type::f32)
            reject("native dense-bias parameter metadata is incompatible");
        return;
    default: reject("native parameter kind is unsupported");
    }
}

} // namespace

native_tensor_view make_dense_tensor_view(
    const cellerator::execution::dense_tensor_view &native,
    native_storage_lease lifetime,
    const native_view_requirements &requirements) {
    if (cellerator::execution::validate_dense_tensor(native)
        != cellerator::execution::biological_validation_code::ok)
        reject("native dense tensor descriptor is invalid");
    if (native.location.residency != cellerator::execution::residency_kind::device)
        reject("CelleraTorch native views currently require device residency");
    validate_layout(native.shape, native.stride, native.rank);
    const torch::ScalarType dtype = torch_type(native.value_type);
    validate_requirements(requirements, native.location.device_ordinal, dtype,
        requirements.access == native_view_access::read_write);
    validate_device_pointer(native.data, native.location.device_ordinal);
    std::shared_ptr<const void> owner = acquire(std::move(lifetime));

    native_tensor_view result{};
    result.tensor = alias_tensor(native.data, native.shape, native.stride,
        native.rank, native.location.device_ordinal, dtype, std::move(owner));
    result.metadata.kind = native_view_kind::dense_operand;
    result.metadata.access = requirements.access;
    result.metadata.device_ordinal = native.location.device_ordinal;
    result.metadata.axis_count = native.rank;
    for (std::uint8_t axis = 0u; axis < native.rank; ++axis)
        result.metadata.axes[axis] = native.axes[axis];
    return result;
}

native_tensor_view make_parameter_tensor_view(
    const cellerator::native_parameter_descriptor &native,
    native_storage_lease lifetime,
    const native_view_requirements &requirements) {
    const cellerator::parameter_descriptor &storage = native.storage;
    if (storage.memory_space != cellerator::parameter_memory_space::device)
        reject("CelleraTorch native parameters must be device-resident");
    if (storage.name == nullptr || storage.name[0] == '\0')
        reject("native parameter requires a stable name");
    if (!cellerator::execution::valid_handle(native.structure)
        || native.structure_epoch.value == 0u || native.generation.value == 0u)
        reject("native parameter identity is incomplete");
    constexpr std::size_t native_parameter_axis_count =
        sizeof(native.axes) / sizeof(native.axes[0]);
    if (native.axis_count == 0u
        || native.axis_count > native_parameter_axis_count)
        reject("native parameter axis metadata is invalid");
    for (std::uint8_t axis = 0u; axis < native.axis_count; ++axis) {
        if (!cellerator::execution::valid_axis_identity(native.axes[axis]))
            reject("native parameter axis identity is invalid");
    }
    validate_parameter_kind_contract(native);
    validate_layout(storage.shape, storage.stride, storage.rank);
    const torch::ScalarType dtype = torch_type(storage.scalar_type);
    validate_requirements(requirements, storage.device_ordinal, dtype,
        storage.writable);
    validate_device_pointer(storage.data, storage.device_ordinal);
    std::shared_ptr<const void> owner = acquire(std::move(lifetime));

    native_tensor_view result{};
    result.tensor = alias_tensor(storage.data, storage.shape, storage.stride,
        storage.rank, storage.device_ordinal, dtype, std::move(owner));
    result.metadata.kind = parameter_kind(native.kind);
    result.metadata.access = requirements.access;
    result.metadata.device_ordinal = storage.device_ordinal;
    result.metadata.structure = native.structure;
    result.metadata.structure_epoch = native.structure_epoch;
    result.metadata.generation = native.generation;
    result.metadata.axis_count = native.axis_count;
    result.metadata.writable = storage.writable;
    result.metadata.parameter_role = storage.role;
    result.metadata.parameter_name = storage.name;
    for (std::uint8_t axis = 0u; axis < native.axis_count; ++axis)
        result.metadata.axes[axis] = native.axes[axis];
    return result;
}

} // namespace celleratorch::native_views
