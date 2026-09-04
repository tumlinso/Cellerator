#include <Cellerator/compiler/sema/freeze_compiler_semantic_type_categories_v1.hh>

#include <array>

namespace cellerator::compiler::sema::v1 {
namespace {

constexpr std::array<semantic_type_descriptor, 15> categories{{
    {semantic_type_category::domain, "domain"},
    {semantic_type_category::axis, "axis"},
    {semantic_type_category::state, "state"},
    {semantic_type_category::relation_structure, "relation_structure"},
    {semantic_type_category::relation_values, "relation_values"},
    {semantic_type_category::relation, "relation"},
    {semantic_type_category::support, "support"},
    {semantic_type_category::active_support, "active_support"},
    {semantic_type_category::order, "order"},
    {semantic_type_category::structure, "structure"},
    {semantic_type_category::value_plane, "value_plane"},
    {semantic_type_category::profile_state, "profile"},
    {semantic_type_category::execution_field, "field"},
    {semantic_type_category::candidate, "candidate"},
    {semantic_type_category::ir_handle, "ir"},
}};

}  // namespace

const semantic_type_descriptor *semantic_type_categories() noexcept {
    return categories.data();
}

std::uint32_t semantic_type_category_count() noexcept {
    return static_cast<std::uint32_t>(categories.size());
}

const semantic_type_descriptor *find_semantic_type_category(
    semantic_type_category category) noexcept {
    for (const auto &descriptor : categories) {
        if (descriptor.category == category)
            return &descriptor;
    }
    return nullptr;
}

bool is_valid_semantic_type_descriptor(
    const semantic_type_descriptor &descriptor) noexcept {
    return descriptor.spelling != nullptr && descriptor.spelling[0] != '\0'
        && descriptor.ownership == type_ownership::compiler_intrinsic
        && !descriptor.owns_storage && !descriptor.fixes_physical_layout;
}

}  // namespace cellerator::compiler::sema::v1
