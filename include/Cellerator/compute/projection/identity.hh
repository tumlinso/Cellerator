#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::math {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 feature_order_identity_schema_version = 1u;
inline constexpr u32 sparse_structure_identity_schema_version = 1u;

enum class feature_order_kind : u32 {
    canonical = 1u,
    packed = 2u
};

struct feature_order_identity {
    u32 schema_version = feature_order_identity_schema_version;
    feature_order_kind kind = feature_order_kind::canonical;
    u32 feature_count = 0u;
    u32 feature_axis_identity_version = 0u;
    u64 feature_axis_identity = 0u;
    u64 packing_geometry_identity = 0u;
};

struct sparse_structure_identity {
    u32 schema_version = sparse_structure_identity_schema_version;
    u32 identity_version = 0u;
    u64 value = 0u;
};

static_assert(std::is_trivially_copyable<feature_order_identity>::value,
    "feature order identity must remain serializable");
static_assert(std::is_trivially_copyable<sparse_structure_identity>::value,
    "sparse structure identity must remain serializable");

} // namespace cellerator::compute::math
