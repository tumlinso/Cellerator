#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t decomposition_vocabulary_schema_version_v1 = 1u;

// A split axis names logical work, never a physical layout dimension.  In
// particular, semantic and hierarchy components remain stable compiler
// concepts even when a selected projection tiles them differently.
enum class split_axis_kind_v1 : std::uint8_t {
    none = 0u,
    source = 1u,
    destination = 2u,
    logical_edge = 3u,
    dense_channel = 4u,
    semantic_component = 5u,
    hierarchical_component = 6u,
    partition = 7u
};

// This vocabulary classifies why fragments exist.  It does not prescribe a
// kernel, projection, schedule, or merge algorithm.
enum class decomposition_kind_v1 : std::uint8_t {
    unsplit = 1u,
    disjoint = 2u,
    overlapping = 3u,
    replicated = 4u,
    staged = 5u
};

// Fragment roles make the completeness obligation explicit.  Owned work is
// counted once; halo and replica work can be consumed but cannot establish
// logical ownership by themselves.
enum class fragment_role_v1 : std::uint8_t {
    complete = 1u,
    owned = 2u,
    halo = 3u,
    replica = 4u
};

constexpr bool valid_split_axis_kind_v1(split_axis_kind_v1 kind) noexcept {
    return kind >= split_axis_kind_v1::none
        && kind <= split_axis_kind_v1::partition;
}

constexpr bool valid_decomposition_kind_v1(
    decomposition_kind_v1 kind) noexcept {
    return kind >= decomposition_kind_v1::unsplit
        && kind <= decomposition_kind_v1::staged;
}

constexpr bool valid_fragment_role_v1(fragment_role_v1 role) noexcept {
    return role >= fragment_role_v1::complete
        && role <= fragment_role_v1::replica;
}

constexpr bool decomposition_requires_split_axis_v1(
    decomposition_kind_v1 kind) noexcept {
    return kind != decomposition_kind_v1::unsplit;
}

constexpr bool fragment_role_owns_logical_work_v1(
    fragment_role_v1 role) noexcept {
    return role == fragment_role_v1::complete
        || role == fragment_role_v1::owned;
}

static_assert(std::is_trivially_copyable_v<split_axis_kind_v1>);
static_assert(std::is_trivially_copyable_v<decomposition_kind_v1>);
static_assert(std::is_trivially_copyable_v<fragment_role_v1>);

}  // namespace cellerator::compute::decomposition
