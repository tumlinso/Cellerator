#pragma once

#include <cstddef>
#include <cstdint>

namespace cellpack {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 abi_version = 1u;
inline constexpr u32 invalid_id = 0xffffffffu;
inline constexpr u32 default_residual_module_id = 0xfffffffeu;

enum class layout_kind : u32 {
    unknown = 0u,
    blocked_ell = 1u,
    sliced_ell = 2u,
    bcsr = 3u,
    bitmap_tile = 4u,
    dense_tile = 5u,
    quantized_blocked_ell = 6u,
    residual_csr = 7u
};

enum class region_role : u32 {
    unknown = 0u,
    primary = 1u,
    shared = 2u,
    conditional = 3u,
    residual = 4u,
    discarded = 5u
};

enum region_flags : u32 {
    region_flag_none = 0u,
    region_flag_conditional = 1u << 0u,
    region_flag_residual = 1u << 1u,
    region_flag_quantized = 1u << 2u,
    region_flag_dense_rhs_local = 1u << 3u
};

enum module_flags : u32 {
    module_flag_none = 0u,
    module_flag_residual = 1u << 0u
};

enum permutation_flags : u32 {
    permutation_flag_none = 0u,
    permutation_flag_identity = 1u << 0u
};

struct alignas(16) packed_region_desc {
    u32 region_id;
    u32 parent_id;
    u32 flags;
    u32 layout;

    u32 role;
    u32 module_id;
    u32 signature_id;
    u32 reserved0;

    u32 row_begin;
    u32 row_count;
    u32 feature_begin;
    u32 feature_count;

    u32 block_size;
    u32 width_class;
    u32 index_offset;
    u32 value_offset;

    u32 aux_offset;
    u32 weight_offset;
    u32 output_offset;
    u32 nnz_count;
};

struct alignas(16) feature_module_desc {
    u32 module_id;
    u32 feature_begin;
    u32 feature_count;
    u32 flags;
};

struct alignas(16) row_group_desc {
    u64 signature_hash;
    u32 row_begin;
    u32 row_count;
    u32 signature_offset;
    u32 signature_count;
    u32 flags;
    u32 reserved0;
};

struct alignas(16) permutation_desc {
    u32 count;
    u32 permutation_offset;
    u32 inverse_offset;
    u32 flags;
};

struct alignas(16) plan_desc {
    u32 version;
    u32 flags;
    u32 row_count;
    u32 feature_count;

    permutation_desc row_permutation;
    permutation_desc feature_permutation;

    u32 module_count;
    u32 row_group_count;
    u32 region_count;
    u32 residual_region_count;

    u32 module_desc_offset;
    u32 row_group_desc_offset;
    u32 region_desc_offset;
    u32 signature_offset;
};

inline constexpr bool is_valid_layout(layout_kind kind) {
    return kind == layout_kind::blocked_ell
        || kind == layout_kind::sliced_ell
        || kind == layout_kind::bcsr
        || kind == layout_kind::bitmap_tile
        || kind == layout_kind::dense_tile
        || kind == layout_kind::quantized_blocked_ell
        || kind == layout_kind::residual_csr;
}

inline constexpr bool is_valid_region_role(region_role role) {
    return role == region_role::primary
        || role == region_role::shared
        || role == region_role::conditional
        || role == region_role::residual
        || role == region_role::discarded;
}

inline constexpr u32 to_u32(layout_kind kind) {
    return static_cast<u32>(kind);
}

inline constexpr u32 to_u32(region_role role) {
    return static_cast<u32>(role);
}

} // namespace cellpack
