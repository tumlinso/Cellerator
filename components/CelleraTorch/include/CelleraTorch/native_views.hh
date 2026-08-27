#pragma once

#include <Cellerator/execution/operands.hh>
#include <Cellerator/parameters.hh>

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>

namespace celleratorch::native_views {

enum class native_view_access : std::uint8_t {
    read_only = 0u,
    read_write = 1u
};

// A native owner supplies an aliasing shared_ptr whose control block keeps the
// backing allocation alive. The adapter never deletes or becomes the canonical
// owner of that allocation; the captured lease only prevents a dangling Torch
// view. An empty or expired lease is rejected.
using native_storage_lease = std::weak_ptr<const void>;

struct native_view_requirements {
    int expected_device_ordinal = -1;
    bool check_dtype = false;
    torch::ScalarType expected_dtype = torch::kFloat32;
    native_view_access access = native_view_access::read_only;
};

enum class native_view_kind : std::uint8_t {
    dense_operand = 0u,
    relation_values = 1u,
    dense_bias = 2u
};

struct native_view_metadata {
    native_view_kind kind = native_view_kind::dense_operand;
    native_view_access access = native_view_access::read_only;
    int device_ordinal = -1;
    cellerator::execution::structure_handle structure{};
    cellerator::execution::structure_epoch structure_epoch{};
    cellerator::execution::value_generation generation{};
    cellerator::execution::axis_identity axes[
        cellerator::execution::biological_operand_max_axes]{};
    std::uint8_t axis_count = 0u;
    bool writable = false;
    cellerator::parameter_role parameter_role =
        cellerator::parameter_role::learned;
    std::string parameter_name;
};

struct native_tensor_view {
    torch::Tensor tensor;
    native_view_metadata metadata;

    bool aliases(const void *native_storage) const noexcept {
        return tensor.defined() && tensor.data_ptr() == native_storage;
    }
};

// These factories allocate no tensor storage and perform no transfer, device
// selection, or synchronization. They validate the declared CUDA allocation,
// then create a strided Torch alias whose deleter only releases the lease.
native_tensor_view make_dense_tensor_view(
    const cellerator::execution::dense_tensor_view &native,
    native_storage_lease lifetime,
    const native_view_requirements &requirements = {});

native_tensor_view make_parameter_tensor_view(
    const cellerator::native_parameter_descriptor &native,
    native_storage_lease lifetime,
    const native_view_requirements &requirements = {});

} // namespace celleratorch::native_views
