#include <Cellerator/compiler/ast/implement_ast_visitors_matchers_and_queries_v1.hh>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>

using namespace Cellerator::compiler::ast;

static bool count_record(const ast_query_record_v1& record, void* context) noexcept {
    *static_cast<std::uint64_t*>(context) += record.semantic_identity != 0;
    return true;
}

int main() {
    constexpr std::uint32_t node_count = 1'000'000;
    std::vector<ast_query_record_v1> records;
    records.reserve(node_count);
    for (std::uint32_t slot = 0; slot < node_count; ++slot) {
        const auto kind = static_cast<ast_query_kind_v1>((slot % 4) + 1);
        records.push_back({{91, slot}, kind, static_cast<std::uint16_t>(slot % 17),
                           std::uint64_t{slot} + 1, 7, std::uint64_t{slot} * 4,
                           std::uint64_t{slot} * 4 + 3});
    }
    std::string error;
    auto index = freeze_ast_query_index_v1(91, std::move(records), &error);
    assert(index && error.empty() && index->size() == node_count);
    assert(index->records(ast_query_kind_v1::field).size() == node_count / 4);

    std::uint64_t visited = 0;
    assert(index->visit_matching({ast_query_kind_v1::effect, {}, 7}, count_record, &visited));
    assert(visited == node_count / 4);

    const auto started = std::chrono::steady_clock::now();
    std::uint64_t checksum = 0;
    constexpr std::uint32_t query_count = 200'000;
    for (std::uint32_t query = 0; query < query_count; ++query) {
        const auto slot = (std::uint64_t{query} * 48271U) % node_count;
        const auto* record = index->find({91, static_cast<std::uint32_t>(slot)});
        assert(record);
        checksum += record->semantic_identity;
        const auto source = index->at_source(7, slot * 4);
        assert(source.size() == 1 && source[0].node.slot == slot);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    assert(checksum != 0);
    std::cout << "nodes=" << node_count << " queries=" << query_count
              << " ns_per_handle_and_source_query=" << elapsed / query_count << '\n';
}
