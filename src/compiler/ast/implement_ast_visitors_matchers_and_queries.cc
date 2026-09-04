#include <Cellerator/compiler/ast/implement_ast_visitors_matchers_and_queries_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::ast {
namespace {

std::size_t kind_index(ast_query_kind_v1 kind) noexcept {
    return static_cast<std::size_t>(kind) - 1;
}

} // namespace

std::size_t ast_query_index_v1::size() const noexcept { return records_.size(); }

ast_record_view_v1 ast_query_index_v1::records(ast_query_kind_v1 kind) const noexcept {
    const auto index = kind_index(kind);
    if (index >= by_kind_.size()) return {nullptr, 0};
    return {by_kind_[index].data(), by_kind_[index].size()};
}

const ast_query_record_v1* ast_query_index_v1::find(ast_node_handle_v1 node) const noexcept {
    const auto found = std::lower_bound(records_.begin(), records_.end(), node,
                                        [](const auto& record, auto sought) {
                                            return record.node.slot < sought.slot;
                                        });
    return found != records_.end() && found->node == node ? &*found : nullptr;
}

ast_record_view_v1 ast_query_index_v1::at_source(std::uint64_t file,
                                                 std::uint64_t begin) const noexcept {
    const auto key = std::pair{file, begin};
    const auto first = std::lower_bound(by_source_.begin(), by_source_.end(), key,
                                        [](const auto* record, const auto& sought) {
                                            return std::pair{record->source_file, record->source_begin} < sought;
                                        });
    const auto last = std::upper_bound(first, by_source_.end(), key,
                                       [](const auto& sought, const auto* record) {
                                           return sought < std::pair{record->source_file, record->source_begin};
                                       });
    return {first == by_source_.end() ? nullptr : &*first,
            static_cast<std::size_t>(last - first)};
}

bool ast_query_index_v1::visit_matching(ast_matcher_v1 matcher,
                                        ast_query_visitor_v1 visitor,
                                        void* context) const noexcept {
    if (!visitor) return false;
    const auto matches = [&](const auto& record) {
        return (!matcher.kind || record.kind == *matcher.kind) &&
               (!matcher.form || record.form == *matcher.form) &&
               (!matcher.source_file || record.source_file == *matcher.source_file);
    };
    if (matcher.kind) {
        for (const auto* record : by_kind_[kind_index(*matcher.kind)])
            if (matches(*record) && !visitor(*record, context)) return false;
    } else {
        for (const auto& record : records_)
            if (matches(record) && !visitor(record, context)) return false;
    }
    return true;
}

std::optional<ast_query_index_v1>
freeze_ast_query_index_v1(ast_arena_id_v1 arena,
                          std::vector<ast_query_record_v1> records,
                          std::string* error) {
    auto fail = [&](std::string message) -> std::optional<ast_query_index_v1> {
        if (error) *error = std::move(message);
        return std::nullopt;
    };
    if (arena == 0) return fail("query index requires an AST arena");
    std::sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
        return left.node.slot < right.node.slot;
    });
    for (std::size_t index = 0; index < records.size(); ++index) {
        const auto& record = records[index];
        if (record.node.arena != arena || !record.node.valid() ||
            (index && records[index - 1].node.slot == record.node.slot) ||
            kind_index(record.kind) >= 5 || record.semantic_identity == 0 ||
            record.source_file == 0 || record.source_begin > record.source_end)
            return fail("invalid query record");
    }

    ast_query_index_v1 index;
    index.records_ = std::move(records);
    index.by_source_.reserve(index.records_.size());
    for (const auto& record : index.records_) {
        index.by_kind_[kind_index(record.kind)].push_back(&record);
        index.by_source_.push_back(&record);
    }
    const auto stable_order = [](const auto* left, const auto* right) {
        if (left->semantic_identity != right->semantic_identity)
            return left->semantic_identity < right->semantic_identity;
        return left->node.slot < right->node.slot;
    };
    for (auto& kind : index.by_kind_) std::sort(kind.begin(), kind.end(), stable_order);
    std::sort(index.by_source_.begin(), index.by_source_.end(), [](const auto* left, const auto* right) {
        if (left->source_file != right->source_file) return left->source_file < right->source_file;
        if (left->source_begin != right->source_begin) return left->source_begin < right->source_begin;
        return left->node.slot < right->node.slot;
    });
    if (error) error->clear();
    return index;
}

} // namespace Cellerator::compiler::ast
