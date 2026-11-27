#pragma once

#include <Cellerator/memory/domain.hh>
#include <Cellerator/ops/mechanism/access.hh>
#include <Cellerator/structure/identity.hh>

#include <cstddef>
#include <cstdint>
#include <span>

namespace cellerator::execution::dependency {

// Cold association for one physical arena. Format-specific maps establish
// which logical coordinates its bytes represent; a byte count is not coverage.
// This is a borrowed descriptor. The operation must retain actual contents
// and wait for the matching readiness event before dereferencing it.
struct DeviceExtentBinding {
    const void* data = nullptr;
    std::size_t bytes = 0;
    memory::placement where{};
    structure::domain_id domain{};
    structure::order_id logical_order{};
    structure::order_id physical_order{};
    structure::structure_id structure{};
    std::uint64_t structure_epoch = 0;
    std::uint64_t value_generation = 0;
    std::uint64_t placement_revision = 0;
    std::uint64_t ready_generation = 0;
    int ready_device = -1;
};

struct ExtentRequirement {
    structure::domain_id domain{};
    structure::order_id logical_order{};
    structure::order_id physical_order{};
    structure::structure_id structure{};
    std::uint64_t structure_epoch = 0;
    std::uint64_t value_generation = 0;
    std::uint64_t placement_revision = 0;
    std::size_t minimum_bytes = 0;
    std::size_t alignment = 1;
    int device = -1;
};

enum class DistributedBindingCode : std::uint8_t {
    success,
    invalid_identity,
    invalid_revision,
    invalid_extent,
    invalid_alignment,
    wrong_domain,
    wrong_order,
    wrong_structure,
    stale_structure,
    stale_values,
    stale_placement,
    wrong_device,
    stale_readiness,
    insufficient_capacity,
    count_mismatch
};

struct TransferRequirement {
    ops::mechanism::Phase phase = ops::mechanism::Phase::primal;
    geometry::dependency::ReadDependency source{};
    int source_device = -1;
    int destination_device = -1;
    std::size_t maximum_bytes = 0;
};

struct DistributedBindingStatus {
    DistributedBindingCode code = DistributedBindingCode::success;
    std::size_t index = 0;
    explicit operator bool() const noexcept {
        return code == DistributedBindingCode::success;
    }
};

// Pure host validation: no CUDA initialization, allocation, pointer discovery
// or dereference. This validates associations, not a provider's event or lease.
DistributedBindingStatus validate_extent(
    const ExtentRequirement&, const DeviceExtentBinding&) noexcept;
DistributedBindingStatus validate_extents(
    std::span<const ExtentRequirement>,
    std::span<const DeviceExtentBinding>) noexcept;

// Every transfer of a numerical read must be justified by the actual phase's
// semantic access contract. The physical byte projection is checked separately.
DistributedBindingStatus validate_transfer_requirements(
    const ops::mechanism::AccessContract&,
    std::span<const TransferRequirement>) noexcept;

} // namespace cellerator::execution::dependency
