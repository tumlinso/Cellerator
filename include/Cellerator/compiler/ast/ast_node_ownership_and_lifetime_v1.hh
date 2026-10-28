#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::ast {

using ast_arena_id_v1 = std::uint32_t;
using source_identity_v1 = std::uint64_t;

inline constexpr std::uint32_t invalid_ast_slot_v1 = UINT32_MAX;

struct ast_node_handle_v1 {
    ast_arena_id_v1 arena = 0;
    std::uint32_t slot = invalid_ast_slot_v1;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return arena != 0 && slot != invalid_ast_slot_v1;
    }
};

struct ast_region_handle_v1 {
    ast_arena_id_v1 arena = 0;
    std::uint32_t slot = invalid_ast_slot_v1;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return arena != 0 && slot != invalid_ast_slot_v1;
    }
};

[[nodiscard]] constexpr bool operator==(ast_node_handle_v1 left,
                                        ast_node_handle_v1 right) noexcept {
    return left.arena == right.arena && left.slot == right.slot;
}

[[nodiscard]] constexpr bool operator==(ast_region_handle_v1 left,
                                        ast_region_handle_v1 right) noexcept {
    return left.arena == right.arena && left.slot == right.slot;
}

enum class ast_node_class_v1 : std::uint16_t {
    unknown = 0,
    translation_unit = 1,
    declaration = 2,
    expression = 3,
    policy = 4,
    native_fragment = 5,
};

// This is the fixed hot header shared by all source-level node families. It is
// pointer-free and has no virtual dispatch. Family-specific data lives in
// separate arena-owned tables introduced by their defining compiler layer.
struct ast_node_record_v1 {
    ast_node_handle_v1 handle{};
    ast_node_handle_v1 parent{};
    ast_region_handle_v1 region{};
    source_identity_v1 source_identity = 0;
    ast_node_class_v1 node_class = ast_node_class_v1::unknown;
    std::uint16_t flags = 0;
};

struct ast_region_record_v1 {
    ast_region_handle_v1 handle{};
    ast_region_handle_v1 parent{};
    ast_node_handle_v1 lexical_owner{};
};

struct ast_arena_metrics_v1 {
    std::size_t node_count = 0;
    std::size_t region_count = 0;
    std::size_t node_bytes = 0;
    std::size_t region_bytes = 0;
    std::size_t allocation_count = 0;
    std::size_t reserved_bytes = 0;
};

class ast_snapshot_v1 {
public:
    ast_snapshot_v1() = default;

    [[nodiscard]] ast_arena_id_v1 arena_id() const noexcept;
    [[nodiscard]] std::size_t node_count() const noexcept;
    [[nodiscard]] std::size_t region_count() const noexcept;
    [[nodiscard]] const ast_node_record_v1* node(ast_node_handle_v1 handle) const noexcept;
    [[nodiscard]] const ast_region_record_v1* region(ast_region_handle_v1 handle) const noexcept;
    [[nodiscard]] ast_arena_metrics_v1 metrics() const noexcept;
    [[nodiscard]] bool shares_storage_with(const ast_snapshot_v1& other) const noexcept;

private:
    struct storage_v1;
    explicit ast_snapshot_v1(std::shared_ptr<const storage_v1> storage) noexcept;

    std::shared_ptr<const storage_v1> storage_;
    friend class ast_arena_v1;
};

class ast_arena_v1 {
public:
    explicit ast_arena_v1(ast_arena_id_v1 arena_id);

    ast_arena_v1(const ast_arena_v1&) = delete;
    ast_arena_v1& operator=(const ast_arena_v1&) = delete;
    ast_arena_v1(ast_arena_v1&&) noexcept;
    ast_arena_v1& operator=(ast_arena_v1&&) noexcept;
    ~ast_arena_v1();

    [[nodiscard]] ast_arena_id_v1 arena_id() const noexcept;
    [[nodiscard]] bool sealed() const noexcept;
    [[nodiscard]] std::optional<ast_region_handle_v1>
    append_region(ast_region_handle_v1 parent = {}, ast_node_handle_v1 lexical_owner = {});
    [[nodiscard]] std::optional<ast_node_handle_v1>
    append_node(ast_node_class_v1 node_class, ast_node_handle_v1 parent,
                ast_region_handle_v1 region, source_identity_v1 source_identity,
                std::uint16_t flags = 0);
    [[nodiscard]] ast_arena_metrics_v1 metrics() const noexcept;
    [[nodiscard]] ast_snapshot_v1 freeze() &&;

private:
    struct builder_storage_v1;
    std::unique_ptr<builder_storage_v1> storage_;
};

static_assert(sizeof(ast_node_handle_v1) == 8);
static_assert(sizeof(ast_region_handle_v1) == 8);

} // namespace Cellerator::compiler::ast
