#pragma once

#include <Cellerator/compiler/sema/implement_domain_and_human_biological_tag_semantics_v1.hh>

#include <cstdint>

namespace cellerator::compiler::sema::v1 {

struct semantic_identity {
    std::uint64_t low = 0;
    std::uint64_t high = 0;
};

struct axis_type {
    domain_type domain{};
    std::uint64_t global_extent = 0;
    semantic_identity logical_order{};
    semantic_identity geometry{};
    semantic_identity partition{};
    std::uint64_t local_extent = 0;
    semantic_identity recovery_identity{};
};

enum class axis_compatibility : std::uint8_t {
    exact = 0,
    domain_mismatch,
    extent_mismatch,
    order_mismatch,
    geometry_mismatch,
    partition_mismatch,
    local_extent_mismatch,
    recovery_mismatch
};

struct explicit_axis_mapping {
    axis_type source{};
    axis_type destination{};
    bool total = false;
    bool one_to_one = false;
};

axis_compatibility compare_axes(const axis_type &left,
                                const axis_type &right) noexcept;
bool valid_explicit_axis_mapping(const explicit_axis_mapping &mapping) noexcept;

}  // namespace cellerator::compiler::sema::v1
