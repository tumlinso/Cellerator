#pragma once

#include <cstdint>

namespace cellerator::compiler::sema::v1 {

inline constexpr std::uint32_t semantic_type_category_schema_version = 1u;

// These are the only source-level categories whose contracts participate
// directly in semantic analysis or planning.  Storage and convenience types
// remain library concerns.
enum class semantic_type_category : std::uint8_t {
    domain = 1,
    axis,
    state,
    relation_structure,
    relation_values,
    relation,
    support,
    active_support,
    order,
    structure,
    value_plane,
    profile_state,
    execution_field,
    candidate,
    ir_handle
};

enum class type_ownership : std::uint8_t {
    compiler_intrinsic = 1,
    standard_library
};

struct semantic_type_descriptor {
    semantic_type_category category{};
    const char *spelling = nullptr;
    type_ownership ownership = type_ownership::compiler_intrinsic;
    bool owns_storage = false;
    bool fixes_physical_layout = false;
};

const semantic_type_descriptor *semantic_type_categories() noexcept;
std::uint32_t semantic_type_category_count() noexcept;
const semantic_type_descriptor *find_semantic_type_category(
    semantic_type_category category) noexcept;
bool is_valid_semantic_type_descriptor(
    const semantic_type_descriptor &descriptor) noexcept;

}  // namespace cellerator::compiler::sema::v1
