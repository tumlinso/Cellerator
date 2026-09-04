#include <Cellerator/compiler/ir/realization/implement_atom_and_extent_bindings_v1.hh>

#include <algorithm>
#include <limits>

namespace cellerator::compiler::ir::realization::v1 {
namespace {

atom_extent_status_v1 fail(
    atom_extent_status_v1 status, std::string* error, const char* message) noexcept {
    if (error != nullptr) {
        *error = message;
    }
    return status;
}

bool power_of_two(std::uint32_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

std::uint64_t index_limit(local_index_width_v1 width) noexcept {
    switch (width) {
        case local_index_width_v1::bits_16:
            return std::numeric_limits<std::uint16_t>::max();
        case local_index_width_v1::bits_32:
            return std::numeric_limits<std::uint32_t>::max();
        case local_index_width_v1::bits_64:
            return std::numeric_limits<std::uint64_t>::max();
    }
    return 0u;
}

} // namespace

atom_extent_status_v1 validate_atom_extent_binding_v1(
    const atom_extent_binding_v1& binding, std::string* error) noexcept {
    if (!valid(binding.identity) || !valid(binding.atom_identity) ||
        !valid(binding.artifact_identity)) {
        return fail(atom_extent_status_v1::invalid_identity, error,
            "binding, atom, and artifact identities are required");
    }
    if (!power_of_two(binding.alignment)) {
        return fail(atom_extent_status_v1::invalid_alignment, error,
            "alignment must be a nonzero power of two");
    }
    if (binding.extents.empty() || binding.global_element_count == 0u) {
        return fail(atom_extent_status_v1::invalid_extent, error,
            "one or more extents and a global size are required");
    }

    auto global = binding.extents;
    std::sort(global.begin(), global.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.global_begin < rhs.global_begin;
    });
    auto local = binding.extents;
    std::sort(local.begin(), local.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.local_begin < rhs.local_begin;
    });
    std::uint64_t total = 0u;
    for (std::size_t index = 0; index < global.size(); ++index) {
        const auto& extent = global[index];
        if (extent.element_count == 0u || extent.stride == 0u ||
            extent.global_begin > binding.global_element_count ||
            extent.element_count > binding.global_element_count - extent.global_begin) {
            return fail(atom_extent_status_v1::invalid_extent, error,
                "extent lies outside the global object");
        }
        if (index != 0u && global[index - 1u].global_begin +
            global[index - 1u].element_count > extent.global_begin) {
            return fail(atom_extent_status_v1::overlapping_global_extent, error,
                "global extents overlap");
        }
        total += extent.element_count;
    }
    for (std::size_t index = 1u; index < local.size(); ++index) {
        if (local[index - 1u].local_begin + local[index - 1u].element_count >
            local[index].local_begin) {
            return fail(atom_extent_status_v1::overlapping_local_extent, error,
                "local extents overlap");
        }
    }
    const auto limit = index_limit(binding.local_index_width);
    if (total == 0u || total - 1u > limit) {
        return fail(atom_extent_status_v1::index_width_overflow, error,
            "local index width cannot represent the bound elements");
    }
    if (binding.canonical_recovery.size() != total) {
        return fail(atom_extent_status_v1::invalid_recovery, error,
            "canonical recovery length must equal the local element count");
    }
    auto recovery = binding.canonical_recovery;
    std::sort(recovery.begin(), recovery.end());
    for (std::size_t index = 0; index < recovery.size(); ++index) {
        if (recovery[index] != index) {
            return fail(atom_extent_status_v1::invalid_recovery, error,
                "canonical recovery must be a local permutation");
        }
    }
    if (error != nullptr) {
        error->clear();
    }
    return atom_extent_status_v1::valid;
}

bool equivalent_atom_extent_binding_v1(
    const atom_extent_binding_v1& lhs,
    const atom_extent_binding_v1& rhs) noexcept {
    if (!(lhs.identity == rhs.identity) || !(lhs.atom_identity == rhs.atom_identity) ||
        !(lhs.parent_atom_identity == rhs.parent_atom_identity) ||
        !(lhs.artifact_identity == rhs.artifact_identity) || lhs.role != rhs.role ||
        lhs.address_space != rhs.address_space ||
        lhs.local_index_width != rhs.local_index_width ||
        lhs.alignment != rhs.alignment ||
        lhs.global_element_count != rhs.global_element_count ||
        lhs.extents.size() != rhs.extents.size() ||
        lhs.canonical_recovery != rhs.canonical_recovery) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.extents.size(); ++index) {
        const auto& a = lhs.extents[index];
        const auto& b = rhs.extents[index];
        if (a.global_begin != b.global_begin || a.element_count != b.element_count ||
            a.local_begin != b.local_begin || a.stride != b.stride) {
            return false;
        }
    }
    return true;
}

} // namespace cellerator::compiler::ir::realization::v1
