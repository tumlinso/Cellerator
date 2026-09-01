#pragma once

#include <cstdint>

namespace cellerator::jbc::verification {

struct identity_v1 {
    std::uint64_t low = 0u;
    std::uint64_t high = 0u;
};

struct atom_fragment_record_v1 {
    identity_v1 atom_identity{};
    identity_v1 order_identity{};
    std::uint64_t logical_begin = 0u;
    std::uint64_t logical_extent = 0u;
    const std::uint64_t *local_to_global = nullptr;
    std::uint64_t local_count = 0u;
};

enum class verification_code_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_identity,
    extent_mismatch,
    overlap,
    gap,
    order_mismatch,
    recovery_out_of_range,
    duplicate_recovery,
};

struct verification_status_v1 {
    verification_code_v1 code = verification_code_v1::success;
    std::uint64_t fragment_index = 0u;
    std::uint64_t local_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == verification_code_v1::success;
    }
};

constexpr bool valid_identity_v1(identity_v1 identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

constexpr bool same_identity_v1(identity_v1 left, identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

inline verification_status_v1 verify_atom_fragments_v1(
    std::uint64_t canonical_extent, identity_v1 required_order,
    const atom_fragment_record_v1 *fragments,
    std::uint64_t fragment_count) noexcept {
    if (canonical_extent == 0u || !valid_identity_v1(required_order) ||
        fragments == nullptr || fragment_count == 0u) {
        return {verification_code_v1::invalid_argument};
    }
    std::uint64_t covered = 0u;
    for (std::uint64_t index = 0u; index < fragment_count; ++index) {
        const auto &fragment = fragments[index];
        if (!valid_identity_v1(fragment.atom_identity)) {
            return {verification_code_v1::invalid_identity, index};
        }
        if (!same_identity_v1(fragment.order_identity, required_order)) {
            return {verification_code_v1::order_mismatch, index};
        }
        if (fragment.logical_extent == 0u ||
            fragment.logical_extent > canonical_extent ||
            fragment.local_count != fragment.logical_extent ||
            fragment.local_to_global == nullptr ||
            fragment.logical_begin >
                canonical_extent - fragment.logical_extent) {
            return {verification_code_v1::extent_mismatch, index};
        }
        if (fragment.logical_begin < covered) {
            return {verification_code_v1::overlap, index};
        }
        if (fragment.logical_begin > covered) {
            return {verification_code_v1::gap, index};
        }
        for (std::uint64_t local = 0u; local < fragment.local_count; ++local) {
            const auto global = fragment.local_to_global[local];
            if (global >= canonical_extent) {
                return {verification_code_v1::recovery_out_of_range,
                    index, local};
            }
            for (std::uint64_t prior_fragment = 0u;
                 prior_fragment <= index; ++prior_fragment) {
                const auto prior_limit = prior_fragment == index ? local :
                    fragments[prior_fragment].local_count;
                for (std::uint64_t prior = 0u; prior < prior_limit; ++prior) {
                    if (fragments[prior_fragment].local_to_global[prior] ==
                        global) {
                        return {verification_code_v1::duplicate_recovery,
                            index, local};
                    }
                }
            }
        }
        covered += fragment.logical_extent;
    }
    if (covered != canonical_extent) {
        return {verification_code_v1::gap, fragment_count};
    }
    return {};
}

}  // namespace cellerator::jbc::verification
