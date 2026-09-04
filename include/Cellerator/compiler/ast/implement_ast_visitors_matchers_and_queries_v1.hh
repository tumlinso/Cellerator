#pragma once

#include <Cellerator/compiler/ast/freeze_ast_node_ownership_and_lifetime_v1.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::ast {

enum class ast_query_kind_v1 : std::uint8_t {
    field = 1,
    relation,
    operation,
    effect,
    other,
};

struct ast_query_record_v1 {
    ast_node_handle_v1 node{};
    ast_query_kind_v1 kind = ast_query_kind_v1::other;
    std::uint16_t form = 0;
    std::uint64_t semantic_identity = 0;
    std::uint64_t source_file = 0;
    std::uint64_t source_begin = 0;
    std::uint64_t source_end = 0;
};

class ast_record_view_v1 {
public:
    [[nodiscard]] const ast_query_record_v1* const* begin() const noexcept { return begin_; }
    [[nodiscard]] const ast_query_record_v1* const* end() const noexcept { return begin_ + size_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] const ast_query_record_v1& operator[](std::size_t index) const noexcept {
        return *begin_[index];
    }

private:
    ast_record_view_v1(const ast_query_record_v1* const* begin, std::size_t size) noexcept
        : begin_(begin), size_(size) {}
    const ast_query_record_v1* const* begin_ = nullptr;
    std::size_t size_ = 0;
    friend class ast_query_index_v1;
};

struct ast_matcher_v1 {
    std::optional<ast_query_kind_v1> kind;
    std::optional<std::uint16_t> form;
    std::optional<std::uint64_t> source_file;
};

using ast_query_visitor_v1 = bool (*)(const ast_query_record_v1&, void*) noexcept;

class ast_query_index_v1 {
public:
    ast_query_index_v1() = default;
    ast_query_index_v1(const ast_query_index_v1&) = delete;
    ast_query_index_v1& operator=(const ast_query_index_v1&) = delete;
    ast_query_index_v1(ast_query_index_v1&&) noexcept = default;
    ast_query_index_v1& operator=(ast_query_index_v1&&) noexcept = default;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] ast_record_view_v1 records(ast_query_kind_v1 kind) const noexcept;
    [[nodiscard]] const ast_query_record_v1* find(ast_node_handle_v1 node) const noexcept;
    [[nodiscard]] ast_record_view_v1 at_source(std::uint64_t file,
                                               std::uint64_t begin) const noexcept;
    [[nodiscard]] bool visit_matching(ast_matcher_v1 matcher,
                                      ast_query_visitor_v1 visitor,
                                      void* context) const noexcept;

private:
    std::vector<ast_query_record_v1> records_;
    std::array<std::vector<const ast_query_record_v1*>, 5> by_kind_;
    std::vector<const ast_query_record_v1*> by_source_;
    friend std::optional<ast_query_index_v1>
    freeze_ast_query_index_v1(ast_arena_id_v1, std::vector<ast_query_record_v1>, std::string*);
};

[[nodiscard]] std::optional<ast_query_index_v1>
freeze_ast_query_index_v1(ast_arena_id_v1 arena,
                          std::vector<ast_query_record_v1> records,
                          std::string* error = nullptr);

} // namespace Cellerator::compiler::ast
